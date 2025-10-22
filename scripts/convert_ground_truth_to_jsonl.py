#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the validated Ground Truth CSV into a JSONL file for model fine-tuning.
Usage:
    python scripts/convert_ground_truth_to_jsonl.py
"""

import csv
import json
from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "data" / "interim" / "Ground Truth.csv"
OUTPUT_JSONL = ROOT_DIR / "data" / "processed" / "Ground Truth.jsonl"

# ---------------------------------------------------------------------
# Verify input exists
# ---------------------------------------------------------------------
if not INPUT_CSV.exists():
    sys.exit(f"❌ Input CSV not found: {INPUT_CSV}")

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

        # Compose text pair
        text = f"A: {a}\nB: {b}"

        # Relationship label (4-class setup)
        if a_to_b and b_to_a:
            label = "equivalent"
        elif a_to_b and not b_to_a:
            label = "A_is_subclass_of_B"
        elif not a_to_b and b_to_a:
            label = "B_is_subclass_of_A"
        else:
            label = "unrelated"

        records.append({"text": text, "label": label})

# ---------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Training file written to: {OUTPUT_JSONL}")
print(f"📊 Samples: {len(records):,}")
if records:
    print("Example:")
    print(json.dumps(records[0], indent=2))
