# ======================================================
# File: compute_delay_final.py
# Project: Agentic ICU Delay Detection
# Changes: is_weekend REMOVED from features
#          Option 3 trigger_type direct
#          minimum 2hr window
#          3-level lab fallback
# ======================================================

import pandas as pd
import numpy as np
import os
import time

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")

print("=" * 55)
print("  ICU Delay Calculation — Final Version")
print("=" * 55)

# ── Load core files ───────────────────────────────────
print("\n[1] Loading files...")

trigger  = pd.read_csv(os.path.join(DATA_PATH, "trigger_table.csv"))
trigger["trigger_time"] = pd.to_datetime(trigger["trigger_time"])

icustays = pd.read_csv(os.path.join(DATA_PATH, "filtered_icustays.csv"))
icustays["outtime"] = pd.to_datetime(icustays["outtime"])
icustays["intime"]  = pd.to_datetime(icustays["intime"])

patients = pd.read_csv(os.path.join(DATA_PATH, "patients.csv"))

print(f"   Triggers loaded:   {len(trigger)}")
print(f"   ICU stays loaded:  {len(icustays)}")
print(f"   Patients loaded:   {len(patients)}")

# ── Time features (hour and night only) ──────────────
print("\n[2] Adding time features...")

trigger["hour_of_day"] = trigger["trigger_time"].dt.hour
trigger["is_night"]    = trigger["hour_of_day"].apply(
    lambda h: 1 if (h >= 22 or h <= 6) else 0
)
# is_weekend REMOVED

# ── Merge ICU stay info ───────────────────────────────
print("[3] Merging ICU stay info...")

trigger = trigger.merge(
    icustays[["stay_id", "outtime"]],
    on="stay_id",
    how="left"
)

# ── Merge patient demographics ────────────────────────
print("[4] Merging patient demographics...")

trigger = trigger.merge(
    patients[["subject_id", "anchor_age", "gender"]],
    on="subject_id",
    how="left"
)

trigger["gender_encoded"] = trigger["gender"].map(
    {"M": 0, "F": 1}
).fillna(-1).astype(int)

print(f"   Rows with demographics: {len(trigger)}")

# ── Lab ID definitions ────────────────────────────────
cardiac_lab_ids = [
    51003, 50911, 50908, 50910, 50963,
    50813, 50912, 50971, 50983,
]

resp_lab_ids = [
    50801, 50802, 50803, 50804, 50805, 50806,
    50808, 50809, 50810, 50811, 50812,
    50815, 50816, 50817, 50818, 50819,
    50820, 50821, 50822, 50823, 50824,
    50825, 50826, 50827, 50828,
]

common_lab_ids = [
    50912, 50971, 50983, 51006, 51221,
    51222, 51265, 51301, 50802, 50820,
    51478, 50813,
]

trigger_to_lab = {
    "HR_HIGH":  "cardiac",
    "MAP_LOW":  "cardiac",
    "SPO2_LOW": "respiratory",
    "RR_HIGH":  "respiratory",
}

all_lab_ids = list(set(
    cardiac_lab_ids + resp_lab_ids + common_lab_ids
))

# ── Load lab events ───────────────────────────────────
print("\n[5] Loading lab events (chunked)...")

filtered_chunks = []

for chunk in pd.read_csv(
    os.path.join(DATA_PATH, "filtered_labevents.csv"),
    chunksize=500_000,
    low_memory=False
):
    chunk = chunk[chunk["itemid"].isin(all_lab_ids)]
    filtered_chunks.append(chunk)

labevents = pd.concat(filtered_chunks, ignore_index=True)
labevents["charttime"] = pd.to_datetime(labevents["charttime"])
labevents = labevents.sort_values(["hadm_id", "charttime"])

print(f"   Relevant lab rows: {len(labevents)}")

lab_dict = {
    hadm_id: group.reset_index(drop=True)
    for hadm_id, group in labevents.groupby("hadm_id")
}
print(f"   Unique admissions with labs: {len(lab_dict)}")

# ── Main delay computation ────────────────────────────
print("\n[6] Computing delays...")

trigger     = trigger.sort_values(["stay_id", "trigger_time"])
results     = []
start_time  = time.time()
total_stays = trigger["stay_id"].nunique()
nan_count   = 0
found_count = 0

for stay_index, (stay_id, group) in enumerate(
    trigger.groupby("stay_id")
):
    group   = group.sort_values("trigger_time").reset_index(drop=True)
    outtime = group.loc[0, "outtime"]

    if pd.isna(outtime):
        continue

    for i in range(len(group)):
        row          = group.loc[i]
        T1           = row["trigger_time"]
        hadm_id      = row["hadm_id"]
        trigger_type = row["trigger_type"]
        lab_group    = trigger_to_lab.get(trigger_type, "common")

        # Minimum 2hr window
        if i < len(group) - 1:
            next_trigger = group.loc[i + 1, "trigger_time"]
            strict_end   = min(next_trigger, outtime)
        else:
            strict_end   = outtime

        min_window_end = T1 + pd.Timedelta(hours=2)
        window_end     = max(strict_end, min_window_end)
        window_start   = T1 + pd.Timedelta(minutes=5)

        delay = np.nan

        if hadm_id in lab_dict:
            labs        = lab_dict[hadm_id]
            window_labs = labs[
                (labs["charttime"] > window_start) &
                (labs["charttime"] <= window_end)
            ]

            # Level 1: trigger-specific labs
            if lab_group == "cardiac":
                specific = window_labs[
                    window_labs["itemid"].isin(cardiac_lab_ids)
                ]
            elif lab_group == "respiratory":
                specific = window_labs[
                    window_labs["itemid"].isin(resp_lab_ids)
                ]
            else:
                specific = pd.DataFrame()

            # Level 2: common labs fallback
            if specific.empty:
                specific = window_labs[
                    window_labs["itemid"].isin(common_lab_ids)
                ]

            # Level 3: any lab in window
            if specific.empty:
                specific = window_labs

            if not specific.empty:
                first_time   = specific.iloc[0]["charttime"]
                delay        = (first_time - T1).total_seconds() / 3600
                found_count += 1
            else:
                nan_count += 1

        results.append({
            "subject_id":     row["subject_id"],
            "hadm_id":        hadm_id,
            "stay_id":        stay_id,
            "trigger_time":   T1,
            "trigger_type":   trigger_type,
            "delay_hours":    delay,
            "HR":             row.get("HR",   np.nan),
            "MAP":            row.get("MAP",  np.nan),
            "RR":             row.get("RR",   np.nan),
            "SPO2":           row.get("SPO2", np.nan),
            "hour_of_day":    row["hour_of_day"],
            "is_night":       row["is_night"],
            # is_weekend REMOVED
            "patient_age":    row.get("anchor_age",     np.nan),
            "gender_encoded": row.get("gender_encoded", -1),
        })

    if stay_index % 500 == 0:
        elapsed = round(time.time() - start_time, 1)
        print(f"   Stays: {stay_index}/{total_stays} | "
              f"Found: {found_count} | "
              f"NaN: {nan_count} | "
              f"Time: {elapsed}s")

# ── Save ──────────────────────────────────────────────
print("\n[7] Saving results...")

delay_df    = pd.DataFrame(results)
output_path = os.path.join(DATA_PATH, "delay_per_trigger_final.csv")
delay_df.to_csv(output_path, index=False)

total    = len(delay_df)
nan_rate = round(delay_df["delay_hours"].isna().mean() * 100, 1)

print("\n" + "=" * 55)
print("  DELAY COMPUTATION COMPLETE")
print("=" * 55)
print(f"  Total rows:    {total}")
print(f"  Has delay:     {delay_df['delay_hours'].notna().sum()}")
print(f"  NaN delay:     {total - delay_df['delay_hours'].notna().sum()}")
print(f"  NaN rate:      {nan_rate}%")
print(f"  Runtime (sec): {round(time.time()-start_time, 1)}")
print(f"  Saved to:      delay_per_trigger_final.csv")
print("=" * 55)

print("\nTrigger type distribution:")
print(delay_df["trigger_type"].value_counts())

print("\nDelay stats by trigger type:")
print(delay_df.groupby("trigger_type")["delay_hours"].agg(
    ["count", "mean", "median", "std"]
).round(2))