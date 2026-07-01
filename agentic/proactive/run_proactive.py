# ======================================================
# File: run_proactive.py
# Changes: is_weekend REMOVED
#          10 features consistent with training
# ======================================================

import os
import pandas as pd
from datetime import timedelta
from agentic.proactive.graph import graph

print("Running Proactive Monitoring...\n")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "lstm_final_dataset_v3.csv")
LAB_PATH  = os.path.join(PROJECT_ROOT, "data", "filtered_labevents.csv")

# ── Load dataset ──────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["trigger_time"])
df = df.sort_values(["stay_id", "trigger_time"])

print("Columns:", list(df.columns))

# ── Feature preparation ───────────────────────────────

# trigger_encoded — create if missing
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

# ── Select stay ───────────────────────────────────────
counts      = df.groupby("stay_id").size()
valid_stays = counts[counts >= 5].index

stay_id =30002654
if stay_id not in valid_stays:
    stay_id = valid_stays[0]
    print(f"Using stay: {stay_id}")

stay_df    = df[df["stay_id"] == stay_id].copy().reset_index(drop=True)
current_df = stay_df.iloc[:5].copy()

print(f"\nSimulating Stay: {stay_id}")
print(f"Total triggers:  {len(stay_df)}\n")

# ── Load labs ─────────────────────────────────────────
hadm_id   = current_df.iloc[-1]["hadm_id"]
labs_list = []

for chunk in pd.read_csv(
    LAB_PATH,
    parse_dates=["charttime"],
    chunksize=500_000,
    low_memory=False
):
    filtered = chunk[chunk["hadm_id"] == hadm_id]
    if not filtered.empty:
        labs_list.append(filtered)

if labs_list:
    labs_df = pd.concat(labs_list)
    print(f"Labs loaded: {len(labs_df)} rows")
else:
    labs_df = pd.DataFrame(columns=["hadm_id", "charttime"])
    print("No labs found")

# ── Simulate current time ─────────────────────────────
trigger_time           = current_df.iloc[-1]["trigger_time"]
simulated_current_time = trigger_time + timedelta(hours=5)

print(f"Trigger time: {trigger_time}")
print(f"Current time: {simulated_current_time}")

# ── Run graph ─────────────────────────────────────────
initial_state = {
    "stay_df":     current_df,
    "labs_df":     labs_df,
    "current_time": simulated_current_time,
}

graph.invoke(initial_state)

print("\nProactive Monitoring Complete ✅")