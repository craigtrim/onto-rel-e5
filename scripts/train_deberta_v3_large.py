#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a DeBERTa-v3-large cross-encoder for directional course relationships.

- Loads:
    data/processed/Ground Truth-Train.jsonl
    data/processed/Ground Truth-Validation.jsonl
- Assumes single-GPU (DGX Spark GB10), uses bf16 if supported else fp16.
- Uses gradient checkpointing, cosine schedule, early stopping on macro-F1.
- Saves best checkpoint to: models/fine_tuned/deberta-v3-large-v1/

Run:
    python scripts/train_deberta_v3_large.py
"""

import os
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from torch import nn

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Paths & config
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "Ground Truth-Train.jsonl"
VAL_PATH   = DATA_DIR / "Ground Truth-Validation.jsonl"

MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = ROOT / "models" / "fine_tuned" / "deberta-v3-large"
CONFIG_DIR = ROOT / "config"
LABELS_JSON = CONFIG_DIR / "labels.json"
ID2LABEL_JSON = CONFIG_DIR / "id2label.json"

SEED = 42
MAX_LENGTH = 196
LR = 1.5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 8
WARMUP_RATIO = 0.06
GRADIENT_ACCUM_STEPS = 1
PER_DEVICE_TRAIN_BS = 48
PER_DEVICE_EVAL_BS  = 96

# -----------------------------
# Label maps
# -----------------------------
DEFAULT_LABEL2ID = {
    "A_is_subclass_of_B": 0,
    "B_is_subclass_of_A": 1,
    "equivalent": 2,
    "unrelated": 3,
}
DEFAULT_ID2LABEL = {str(v): k for k, v in DEFAULT_LABEL2ID.items()}

if LABELS_JSON.exists():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        LABEL2ID = json.load(f)
else:
    LABEL2ID = DEFAULT_LABEL2ID

if ID2LABEL_JSON.exists():
    with open(ID2LABEL_JSON, "r", encoding="utf-8") as f:
        ID2LABEL = json.load(f)
else:
    ID2LABEL = DEFAULT_ID2LABEL

NUM_LABELS = len(LABEL2ID)

# -----------------------------
# Helpers
# -----------------------------
def split_text_pair(text: str) -> Tuple[str, str]:
    """
    Training data uses the format: "A: <a_text>\\nB: <b_text>"
    Parse robustly; fall back to single sequence if format differs.
    """
    if not isinstance(text, str):
        return str(text), ""
    # Expect "A: ...\nB: ..."
    parts = text.split("\n")
    if len(parts) == 2 and parts[0].startswith("A:") and parts[1].startswith("B:"):
        a = parts[0][2:].strip()  # after "A:"
        b = parts[1][2:].strip()  # after "B:"
        return a, b
    # Fallback: try simple split on "B:"
    if "\nB:" in text:
        a, b = text.split("\nB:", 1)
        a = a.replace("A:", "").strip()
        b = b.strip()
        return a, b
    # As last resort, feed entire thing as first segment
    return text, ""

def device_capabilities():
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available()
    return bf16_ok, fp16_ok

def compute_metrics_fn(label_ids: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(label_ids, preds)
    macro_f1 = f1_score(label_ids, preds, average="macro")
    micro_f1 = f1_score(label_ids, preds, average="micro")
    return {"accuracy": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}

# -----------------------------
# Load datasets
# -----------------------------
if not TRAIN_PATH.exists() or not VAL_PATH.exists():
    raise FileNotFoundError("Expected train/val JSONL at data/processed/. "
                            "Run the converter to create Ground Truth-Train.jsonl and -Validation.jsonl.")

raw_ds = DatasetDict({
    "train": load_dataset("json", data_files=str(TRAIN_PATH))["train"],
    "validation": load_dataset("json", data_files=str(VAL_PATH))["train"],
})

# -----------------------------
# Model / tokenizer
# -----------------------------
set_seed(SEED)

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={int(k): v for k, v in ID2LABEL.items()},
    label2id=LABEL2ID,
)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
    a_texts, b_texts = [], []
    for t in batch["text"]:
        a, b = split_text_pair(t)
        a_texts.append(a)
        b_texts.append(b)
    enc = tokenizer(
        a_texts,
        b_texts,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    # Map string labels -> ids
    enc["labels"] = [LABEL2ID[lbl] for lbl in batch["label"]]
    return enc

proc_ds = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# Precision & perf knobs
# -----------------------------
bf16_ok, fp16_ok = device_capabilities()
fp16 = False
bf16 = False
if bf16_ok:
    bf16 = True
elif fp16_ok:
    fp16 = True

# Allow optimized matmul on GB10
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# -----------------------------
# Model init
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
)

# Gradient checkpointing to reduce activation memory
# model.config.use_cache = False  # must be False when using gradient checkpointing
# model.gradient_checkpointing_enable()

# Optional: a tiny label-smoothing if desired (commented out)
# model.config.label_smoothing = 0.0

# -----------------------------
# Training args
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    logging_dir=str(OUTPUT_DIR / "logs"),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    seed=SEED,
    dataloader_num_workers=16,
    bf16=True,
    tf32=True,                    # Use Tensor Float-32
    fp16=fp16,
    report_to=[],
    logging_steps=50,
    save_total_limit=4,
    label_smoothing_factor=0.1,  # Prevent overconfidence
    dataloader_pin_memory=True,   # Pin memory for faster transfers
    group_by_length=True,         # Group similar lengths for efficiency
    # gradient_checkpointing=True,  # Now you can afford it with larger batches
    fp16_opt_level="O2",         # Most aggressive mixed precision
)

# -----------------------------
# Metrics callback
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_fn(labels, preds)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc_ds["train"],
    eval_dataset=proc_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# -----------------------------
# Train
# -----------------------------
train_output = trainer.train()

# -----------------------------
# Final eval & reports
# -----------------------------
eval_metrics = trainer.evaluate()
print("\n=== Validation metrics (HF) ===")
for k, v in eval_metrics.items():
    if isinstance(v, (float, int)):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

# Detailed classification report & confusion matrix
preds = trainer.predict(proc_ds["validation"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=-1)

# Build ordered target names
ordered_labels = [LABEL2ID["A_is_subclass_of_B"], LABEL2ID["B_is_subclass_of_A"],
                  LABEL2ID["equivalent"], LABEL2ID["unrelated"]]
target_names = ["A_is_subclass_of_B", "B_is_subclass_of_A", "equivalent", "unrelated"]

print("\n=== Validation classification report ===")
print(classification_report(y_true, y_pred, labels=ordered_labels, target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
print("=== Validation confusion matrix (rows=true, cols=pred) ===")
print(cm)

# -----------------------------
# Save label maps & final artifacts
# -----------------------------
(OUTPUT_DIR / "label2id.json").write_text(json.dumps(LABEL2ID, indent=2))
(OUTPUT_DIR / "id2label.json").write_text(json.dumps({int(k): v for k, v in ID2LABEL.items()}, indent=2))
(OUTPUT_DIR / "metrics_val.json").write_text(json.dumps({
    **{k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
}, indent=2))

print(f"\n✅ Training complete. Best checkpoint saved in: {trainer.state.best_model_checkpoint or OUTPUT_DIR}")
