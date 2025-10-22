#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the (balanced or unbalanced) Ground Truth CSV into
two JSONL files (train/validation) for model fine-tuning.

Usage:
    python scripts/convert_ground_truth_to_jsonl.py
"""

import csv
import json
import random
from pathlib import Path
import sys
from collections import defaultdict

# ---------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

BALANCED_CSV = INTERIM_DIR / "Ground Truth-Balanced.csv"
UNBALANCED_CSV = INTERIM_DIR / "Ground Truth.csv"

# prefer balanced version if available
if BALANCED_CSV.exists():
    INPUT_CSV = BALANCED_CSV
    print(f"📄 Using balanced dataset: {INPUT_CSV.name}")
elif UNBALANCED_CSV.exists():
    INPUT_CSV = UNBALANCED_CSV
    print(f"📄 Using unbalanced dataset: {INPUT_CSV.name}")
else:
    sys.exit("❌ No input CSV found in data/interim/")

# ---------------------------------------------------------------------
# Load CSV manually (no pandas)
# ---------------------------------------------------------------------
records = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    required = ["A", "B", "A->B", "B->A"]
    missing = [c for c in required if c not in reader.fieldnames]
    if missing:
        sys.exit(f"❌ Missing required columns: {missing}")

    for i, row in enumerate(reader, start=2):
        a = row["A"].strip()
        b = row["B"].strip()
        a_to_b = row["A->B"].strip().lower() in ("true", "t", "yes", "1")
        b_to_a = row["B->A"].strip().lower() in ("true", "t", "yes", "1")

        if not a or not b:
            print(f"⚠️ Skipping blank row {i}: A='{a}', B='{b}'")
            continue

        text = f"A: {a}\nB: {b}"

        if a_to_b and b_to_a:
            label = "equivalent"
        elif a_to_b and not b_to_a:
            label = "A_is_subclass_of_B"
        elif not a_to_b and b_to_a:
            label = "B_is_subclass_of_A"
        else:
            label = "unrelated"

        records.append({"text": text, "label": label})

if not records:
    sys.exit("❌ No valid records found in input CSV.")

# ---------------------------------------------------------------------
# Stratified deterministic 80/20 split
# ---------------------------------------------------------------------
train_records, val_records = [], []
label_groups = defaultdict(list)
for r in records:
    label_groups[r["label"]].append(r)

random.seed(42)  # deterministic split
for label, group in label_groups.items():
    random.shuffle(group)
    split_idx = int(len(group) * 0.8)
    train_records.extend(group[:split_idx])
    val_records.extend(group[split_idx:])

# ---------------------------------------------------------------------
# Write train / validation JSONL files
# ---------------------------------------------------------------------
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_path = PROCESSED_DIR / "Ground Truth-Train.jsonl"
val_path = PROCESSED_DIR / "Ground Truth-Validation.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for r in val_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
def summarize(records, split_name):
    counts = defaultdict(int)
    for r in records:
        counts[r["label"]] += 1
    total = len(records)
    print(f"\n📊 {split_name} set ({total:,} samples):")
    for label, count in counts.items():
        pct = (count / total) * 100
        print(f"  {label:<25} {count:>6}  ({pct:5.2f}%)")

print(f"\n✅ Training/validation files written to: {PROCESSED_DIR}")
summarize(train_records, "Training")
summarize(val_records, "Validation")
print(f"\nExample training record:\n{json.dumps(train_records[0], indent=2)}")
