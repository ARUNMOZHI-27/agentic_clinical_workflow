# ======================================================
# File: filter_large_tables.py
# Purpose: Filter procedureevents, labevents, labitems
# Using filtered_icustays cohort
# ======================================================

import pandas as pd
import os

import os

# Get current file directory (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up ONE level to project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Define data path
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

print("Project root:", PROJECT_ROOT)
print("Data path:", DATA_PATH)

print("Loading filtered ICU cohort...")

filtered_icustays = pd.read_csv(
    os.path.join(DATA_PATH, "filtered_icustays.csv")
)

target_hadm = filtered_icustays["hadm_id"].unique()
target_stay = filtered_icustays["stay_id"].unique()

print("Total target hadm_id:", len(target_hadm))
print("Total target stay_id:", len(target_stay))


# ======================================================
# 2. FILTER PROCEDUREEVENTS (ICU LEVEL)
# ======================================================

print("\nFiltering procedureevents...")

proc_path = os.path.join(DATA_PATH, "procedureevents.csv")
proc_out_path = os.path.join(DATA_PATH, "filtered_procedureevents.csv")

chunks = []

for chunk in pd.read_csv(proc_path, chunksize=500000):
    filtered_chunk = chunk[
        chunk["stay_id"].isin(target_stay)
    ]
    chunks.append(filtered_chunk)

filtered_procedures = pd.concat(chunks, ignore_index=True)
filtered_procedures.to_csv(proc_out_path, index=False)

print("Filtered procedureevents saved.")
print("Shape:", filtered_procedures.shape)


# ======================================================
# 3. FILTER LABEVENTS (HOSPITAL LEVEL)
# ======================================================

print("\nFiltering labevents...")

lab_path = os.path.join(DATA_PATH, "labevents.csv")
lab_out_path = os.path.join(DATA_PATH, "filtered_labevents.csv")

chunks = []

for chunk in pd.read_csv(lab_path, chunksize=500000):
    filtered_chunk = chunk[
        chunk["hadm_id"].isin(target_hadm)
    ]
    chunks.append(filtered_chunk)

filtered_labs = pd.concat(chunks, ignore_index=True)
filtered_labs.to_csv(lab_out_path, index=False)

print("Filtered labevents saved.")
print("Shape:", filtered_labs.shape)


# ======================================================
# 4. FILTER D_LABITEMS (Keep Only Used Lab Items)
# ======================================================

print("\nFiltering d_labitems...")

labitems = pd.read_csv(os.path.join(DATA_PATH, "d_labitems.csv"))

used_itemids = filtered_labs["itemid"].unique()

filtered_labitems = labitems[
    labitems["itemid"].isin(used_itemids)
]

filtered_labitems.to_csv(
    os.path.join(DATA_PATH, "filtered_d_labitems.csv"),
    index=False
)

print("Filtered labitems saved.")
print("Shape:", filtered_labitems.shape)


print("\nALL FILTERING COMPLETE ✅")