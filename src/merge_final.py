# ======================================================
# File: merge_final.py
# Project: Agentic ICU Delay Detection
# Changes: is_weekend REMOVED from all features
# ======================================================

import pandas as pd
import numpy as np
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")

print("=" * 55)
print("Merge Final Dataset")
print("=" * 55)

# ── Load delay table ──────────────────────────────────
print("\n[1] Loading delay table...")

delay_df = pd.read_csv(
    os.path.join(DATA_PATH, "delay_per_trigger_final.csv"),
    parse_dates=["trigger_time"],
    low_memory=False
)

print(f"Delay table rows: {len(delay_df)}")
print(f"Columns: {list(delay_df.columns)}")

# ── Fix datatype warnings ─────────────────────────────
numeric_cols = [
    "hour_of_day", "is_night",
    "patient_age", "gender_encoded"
    # is_weekend REMOVED
]

for col in numeric_cols:
    if col in delay_df.columns:
        delay_df[col] = pd.to_numeric(delay_df[col], errors="coerce")



# ── Load trigger table ────────────────────────────────
print("\n[2] Loading trigger table...")

trigger_df = pd.read_csv(
    os.path.join(DATA_PATH, "trigger_table.csv"),
    parse_dates=["trigger_time"],
    low_memory=False
)

print(f"Trigger table rows: {len(trigger_df)}")

# ── Merge vitals if necessary ─────────────────────────
vital_cols = ["HR", "MAP", "RR", "SPO2"]

vitals_in_delay = all(c in delay_df.columns for c in vital_cols)

if vitals_in_delay:
    print("\n[3] Vitals already present — skipping merge")
    merged = delay_df.copy()
else:
    print("\n[3] Merging vitals from trigger table...")
    merge_keys  = ["subject_id", "hadm_id", "stay_id", "trigger_time"]
    trigger_cols = merge_keys + vital_cols
    trigger_cols = [c for c in trigger_cols if c in trigger_df.columns]

    merged = delay_df.merge(
        trigger_df[trigger_cols],
        on=merge_keys,
        how="left"
    )
    print(f"Rows after merge: {len(merged)}")

# ── Ensure Option 3 ───────────────────────────────────
print("\n[4] Validating Option 3 compliance...")

if "episode_type" in merged.columns:
    merged = merged.drop(columns=["episode_type"])
    print("Dropped episode_type")

if "trigger_type" not in merged.columns:
    raise ValueError("trigger_type column missing")

# ── Encode trigger types ──────────────────────────────
print("\n[5] Encoding trigger_type...")

trigger_map = {
    "HR_HIGH": 0, "MAP_LOW": 1,
    "SPO2_LOW": 2, "RR_HIGH": 3,
}

merged["trigger_encoded"] = merged["trigger_type"].map(
    trigger_map
).fillna(-1).astype(int)

print(merged["trigger_encoded"].value_counts())

# ── Sort ──────────────────────────────────────────────
merged = merged.sort_values(["stay_id", "trigger_time"])

# ── Vital Imputation ──────────────────────────────────
print("\n[6] Filling missing vitals...")

merged["MAP"] = merged.groupby("stay_id")["MAP"].ffill()
merged["MAP"] = merged.groupby("stay_id")["MAP"].bfill()
print("MAP filled using patient forward/backward fill")

def safe_fill(series):
    if series.notna().sum() == 0:
        return series
    return series.fillna(series.median())

for col in ["HR", "RR", "SPO2"]:
    before = merged[col].isna().sum()
    merged[col] = merged.groupby("stay_id")[col].transform(safe_fill)
    merged[col] = merged.groupby("trigger_type")[col].transform(safe_fill)
    merged[col] = merged[col].fillna(merged[col].median())
    after  = merged[col].isna().sum()
    print(f"{col}: {before} → {after}")

# ── Fill demographic features ─────────────────────────
print("\n[7] Filling demographic NaNs...")

demo_fills = {
    "patient_age":    merged["patient_age"].median(),
    "gender_encoded": -1,
    "hour_of_day":    0,
    "is_night":       0,
    # is_weekend REMOVED
}

for col, val in demo_fills.items():
    if col in merged.columns:
        missing = merged[col].isna().sum()
        merged[col] = merged[col].fillna(val)
        if missing > 0:
            print(f"{col}: filled {missing}")

# ── Compute time gap ──────────────────────────────────
print("\n[8] Computing time gap between triggers...")

merged["time_gap_min"] = (
    merged.groupby("stay_id")["trigger_time"]
    .diff()
    .dt.total_seconds() / 60
).fillna(0)

# ── Clip extreme vitals ───────────────────────────────
print("\n[9] Clipping extreme vitals...")

merged["HR"]   = merged["HR"].clip(30, 220)
merged["MAP"]  = merged["MAP"].clip(30, 150)
merged["RR"]   = merged["RR"].clip(5, 60)
merged["SPO2"] = merged["SPO2"].clip(70, 100)

# ── Final column order ────────────────────────────────
print("\n[10] Ordering columns...")

final_cols = [
    "subject_id", "hadm_id", "stay_id", "trigger_time",
    "trigger_type", "trigger_encoded",
    "delay_hours",
    "HR", "MAP", "RR", "SPO2",
    "time_gap_min",
    "hour_of_day", "is_night",
    # is_weekend REMOVED
    "patient_age", "gender_encoded",
]

final_cols = [c for c in final_cols if c in merged.columns]
merged     = merged[final_cols]

# ── Validation ────────────────────────────────────────
print("\n[11] Dataset validation...")
print("Missing values:")
print(merged.isna().sum())
print("\nVital ranges:")
for col in ["HR", "MAP", "RR", "SPO2"]:
    print(f"{col}: {merged[col].min()} - {merged[col].max()}")

print("\nUnique ICU stays:  ", merged["stay_id"].nunique())
print("Unique patients:   ", merged["subject_id"].nunique())

# ── Summary ───────────────────────────────────────────
print("\n" + "=" * 55)
print("DATASET SUMMARY")
print("=" * 55)
print("Total rows:     ", len(merged))
print("Rows with delay:", merged["delay_hours"].notna().sum())
print("\nDelay statistics:")
print(merged["delay_hours"].describe())
print("\nTrigger distribution:")
print(merged["trigger_type"].value_counts())

# ── Save ──────────────────────────────────────────────
print("\n[12] Saving dataset...")

output_path = os.path.join(DATA_PATH, "lstm_final_dataset_v3.csv")
merged.to_csv(output_path, index=False)

print("\nSaved: lstm_final_dataset_v3.csv")
print("=" * 55)