# ======================================================
# File: run_reactive.py
# Changes: is_weekend REMOVED
#          10 features consistent with training
#          eligibility check BEFORE dropna
#          ONE-TIME RUN on last valid trigger
# ======================================================

import pandas as pd
import numpy as np
from agentic.graph import graph
import os

SEQ_LEN      = 5
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "lstm_final_dataset_v3.csv"
)

# ── Load dataset ──────────────────────────────────────
print("Loading dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=["trigger_time"])
print("Columns:", list(df.columns))

# ── Feature preparation ───────────────────────────────

# trigger_encoded
if "trigger_encoded" not in df.columns:
    trigger_map = {
        "HR_HIGH": 0, "MAP_LOW": 1,
        "SPO2_LOW": 2, "RR_HIGH": 3
    }
    df["trigger_encoded"] = df["trigger_type"].map(
        trigger_map
    ).fillna(-1).astype(int)

# Vitals group-wise fill
vitals = ["HR", "MAP", "RR", "SPO2"]
for col in vitals:
    df[col] = df.groupby("trigger_type")[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())

df = df.sort_values(["stay_id", "trigger_time"])

# Time gap
df["time_gap_min"] = (
    df.groupby("stay_id")["trigger_time"]
    .diff().dt.total_seconds() / 60
).fillna(0).clip(0, 1440)

# Time features
if "hour_of_day" not in df.columns:
    df["hour_of_day"] = df["trigger_time"].dt.hour
if "is_night" not in df.columns:
    df["is_night"] = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 22 or h <= 6) else 0
    )

# Demographics
if "patient_age" not in df.columns:
    df["patient_age"] = 65
else:
    df["patient_age"] = df["patient_age"].fillna(
        df["patient_age"].median()
    )
if "gender_encoded" not in df.columns:
    df["gender_encoded"] = -1
else:
    df["gender_encoded"] = df["gender_encoded"].fillna(-1)

# ── Verify features ───────────────────────────────────
FEATURES = [
    "trigger_encoded", "time_gap_min",
    "HR", "MAP", "RR", "SPO2",
    "hour_of_day", "is_night",
    "patient_age", "gender_encoded",
]

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print("WARNING — missing features:", missing)
else:
    print("All 10 features present ✅")

# ── Eligibility BEFORE dropna ─────────────────────────
eligible = df.groupby("stay_id").filter(
    lambda x: len(x) > SEQ_LEN
)["stay_id"].unique()

# NOW drop NaN delay rows
df = df.dropna(subset=["delay_hours"])

# ── Select stay ───────────────────────────────────────
stay_id = 30002654

if stay_id not in eligible:
    stay_id = eligible[0]
    print(f"Using stay: {stay_id}")

print(f"\nSimulating Stay: {stay_id}")

stay_df = df[df["stay_id"] == stay_id].reset_index(drop=True)

print(f"Total triggers with valid delay: {len(stay_df)}\n")

# ── Fallback for short delay-only stays ───────────────
if len(stay_df) <= SEQ_LEN:
    print("Not enough delay rows — using full stay for sequence.")

    full_df      = pd.read_csv(DATA_PATH, parse_dates=["trigger_time"])
    stay_full_df = full_df[
        full_df["stay_id"] == stay_id
    ].sort_values("trigger_time").reset_index(drop=True)

    if "trigger_encoded" not in stay_full_df.columns:
        trigger_map = {"HR_HIGH":0,"MAP_LOW":1,"SPO2_LOW":2,"RR_HIGH":3}
        stay_full_df["trigger_encoded"] = stay_full_df[
            "trigger_type"
        ].map(trigger_map).fillna(-1).astype(int)

    for col in vitals:
        stay_full_df[col] = stay_full_df[col].fillna(
            stay_full_df[col].median()
        )

    stay_full_df["time_gap_min"] = (
        stay_full_df["trigger_time"]
        .diff().dt.total_seconds() / 60
    ).fillna(0).clip(0, 1440)

    stay_full_df["hour_of_day"] = stay_full_df["trigger_time"].dt.hour
    stay_full_df["is_night"]    = stay_full_df["hour_of_day"].apply(
        lambda h: 1 if (h >= 22 or h <= 6) else 0
    )

    if "is_weekend" in stay_full_df.columns:
        stay_full_df = stay_full_df.drop(columns=["is_weekend"])

    stay_full_df["patient_age"]    = stay_full_df.get(
        "patient_age", pd.Series([65]*len(stay_full_df))
    ).fillna(65)
    stay_full_df["gender_encoded"] = stay_full_df.get(
        "gender_encoded", pd.Series([-1]*len(stay_full_df))
    ).fillna(-1)

    stay_df = stay_full_df
    print(f"Using full stay: {len(stay_df)} triggers\n")

# ── Run graph ONCE on last valid trigger ──────────────
i = len(stay_df) - 1

initial_state = {
    "stay_id": stay_id,
    "stay_df": stay_df,
    "index":   i,
}

graph.invoke(initial_state)

print("\nReactive Monitoring Complete ✅")