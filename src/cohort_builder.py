# ======================================================
# File: cohort_builder.py
# Project: Agentic ICU Delay Detection
# Purpose: Build Cardiac + Respiratory ICU Cohort
# ======================================================

import pandas as pd
import os

# ------------------------------------------------------
# 1. Define Data Path
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")

print("Loading files from:", DATA_PATH)

# ------------------------------------------------------
# 2. Load Required Tables
# ------------------------------------------------------
icustays = pd.read_csv(os.path.join(DATA_PATH, "icustays.csv"))
admissions = pd.read_csv(os.path.join(DATA_PATH, "admissions.csv"))
diagnoses = pd.read_csv(os.path.join(DATA_PATH, "diagnoses_icd.csv"))

print("Files loaded successfully!")

# ------------------------------------------------------
# 3. Keep Only Necessary Columns
# ------------------------------------------------------
icustays = icustays[[
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime"
]]

admissions = admissions[[
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime"
]]

diagnoses = diagnoses[[
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version"
]]

# Convert datetime columns
icustays["intime"] = pd.to_datetime(icustays["intime"])
icustays["outtime"] = pd.to_datetime(icustays["outtime"])

# ------------------------------------------------------
# 4. Filter Cardiac + Respiratory ICD Codes
# ------------------------------------------------------

diagnoses["icd_code"] = diagnoses["icd_code"].astype(str)

# Cardiac ICD-10 prefixes
cardiac_prefix = [
    "I20","I21","I22","I23","I24","I25",
    "I47","I48","I49","I50"
]

# Respiratory ICD-10 prefixes
resp_prefix = [
    "J44","J45","J96","J18"
]

def is_target_icd(code):
    return any(code.startswith(p) for p in cardiac_prefix + resp_prefix)

filtered_diag = diagnoses[diagnoses["icd_code"].apply(is_target_icd)]

print("Filtered diagnoses shape:", filtered_diag.shape)

# ------------------------------------------------------
# 5. Get Target Hospital Admissions
# ------------------------------------------------------

target_hadm_ids = filtered_diag["hadm_id"].unique()

print("Number of Cardiac/Resp Admissions:", len(target_hadm_ids))

# ------------------------------------------------------
# 6. Filter ICU Stays Based on Diagnosis
# ------------------------------------------------------

filtered_icustays = icustays[icustays["hadm_id"].isin(target_hadm_ids)]

print("Number of ICU stays after filtering:", filtered_icustays.shape)

# ------------------------------------------------------
# 7. Save Clean Cohort
# ------------------------------------------------------

output_path = os.path.join(DATA_PATH, "filtered_icustays.csv")
filtered_icustays.to_csv(output_path, index=False)

print("Filtered ICU cohort saved to:", output_path)

print("Cohort building completed successfully!")
