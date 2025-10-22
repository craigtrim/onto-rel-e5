#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balance the Ground Truth dataset by augmenting underrepresented relationships.
Reads the validated CSV and creates a balanced version with inverted examples.

Usage:
    python scripts/balance_ground_truth.py
"""

import csv
from pathlib import Path
import sys
from collections import Counter
import random

# ---------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "data" / "interim" / "Ground Truth.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "interim" / "Ground Truth-Balanced.csv"

# ---------------------------------------------------------------------
# Verify input exists
# ---------------------------------------------------------------------
if not INPUT_CSV.exists():
    sys.exit(f"❌ Input file not found: {INPUT_CSV}")

# ---------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------
records = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    required = ["A", "B", "A->B", "B->A"]
    missing = [c for c in required if c not in reader.fieldnames]
    if missing:
        sys.exit(f"❌ Missing required columns: {missing}")

    records = list(reader)

print(f"📥 Loaded {len(records):,} rows from {INPUT_CSV.name}")

# ---------------------------------------------------------------------
# Label assignment helper
# ---------------------------------------------------------------------
def classify(row):
    a_to_b = row["A->B"].strip().lower() in ("true", "t", "yes", "1")
    b_to_a = row["B->A"].strip().lower() in ("true", "t", "yes", "1")
    if a_to_b and b_to_a:
        return "equivalent"
    elif a_to_b and not b_to_a:
        return "A_is_subclass_of_B"
    elif not a_to_b and b_to_a:
        return "B_is_subclass_of_A"
    return "unrelated"

# ---------------------------------------------------------------------
# Compute original distribution
# ---------------------------------------------------------------------
label_counts = Counter(classify(r) for r in records)
total = sum(label_counts.values())

print("\n🔎 Original label distribution:")
print(f"{'Label':<25} | {'Count':>8} | {'Percent':>8}")
print("-" * 48)
for label, count in label_counts.items():
    pct = (count / total) * 100
    print(f"{label:<25} | {count:>8} | {pct:>7.2f}%")

# ---------------------------------------------------------------------
# Generate synthetic inversions
# ---------------------------------------------------------------------
needed = label_counts["A_is_subclass_of_B"] - label_counts["B_is_subclass_of_A"]
if needed <= 0:
    print("\n✅ Dataset already balanced or reversed class larger — no augmentation needed.")
    sys.exit(0)

print(f"\n🧩 Augmenting {needed:,} synthetic inversions to balance classes...")

a_sub_b_rows = [r for r in records if classify(r) == "A_is_subclass_of_B"]
random.shuffle(a_sub_b_rows)
augment = a_sub_b_rows[:needed]

augmented_records = records.copy()
for r in augment:
    augmented_records.append({
        "A": r["B"],            # swap direction
        "B": r["A"],
        "A->B": "False",         # invert flags
        "B->A": "True"
    })

print(f"✅ Added {len(augment):,} inverted examples.")

# ---------------------------------------------------------------------
# Save balanced CSV
# ---------------------------------------------------------------------
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["A", "B", "A->B", "B->A"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(augmented_records)

# ---------------------------------------------------------------------
# Verify new distribution
# ---------------------------------------------------------------------
label_counts_new = Counter(classify(r) for r in augmented_records)
total_new = sum(label_counts_new.values())

print(f"\n💾 Balanced dataset written to: {OUTPUT_CSV}")
print(f"📊 Total rows: {total_new:,}")
print("\n🔎 New label distribution:")
print(f"{'Label':<25} | {'Count':>8} | {'Percent':>8}")
print("-" * 48)
for label, count in label_counts_new.items():
    pct = (count / total_new) * 100
    print(f"{label:<25} | {count:>8} | {pct:>7.2f}%")

# ---------------------------------------------------------------------
# Balance check
# ---------------------------------------------------------------------
a_sub_b = label_counts_new["A_is_subclass_of_B"]
b_sub_a = label_counts_new["B_is_subclass_of_A"]

if b_sub_a == 0:
    print("\n⚠️ Something went wrong: no B_is_subclass_of_A examples after augmentation.")
    sys.exit(1)

ratio = a_sub_b / b_sub_a
tolerance = 1.05  # 5% tolerance for "balanced"
print("\n⚖️ Balance check:")
print(f"A_is_subclass_of_B: {a_sub_b:,}")
print(f"B_is_subclass_of_A: {b_sub_a:,}")
print(f"Ratio: {ratio:.2f} (1.00 = perfect balance)")

if 1 / tolerance <= ratio <= tolerance:
    print("✅ Classes are now balanced within 5% tolerance.")
elif ratio > tolerance:
    print(f"⚠️ Still imbalanced: A_is_subclass_of_B is {ratio:.2f}× larger.")
else:
    print(f"⚠️ Still imbalanced: B_is_subclass_of_A is {1/ratio:.2f}× larger.")

print("\n✅ Balance operation complete.")
