# import numpy as np

# SEQ_LEN = 5

# def feature_agent(state):
#     print("[Feature Agent]")

#     stay_df = state["stay_df"]
#     i = state["index"]

#     seq_df = stay_df.iloc[i-SEQ_LEN:i]

#     features = [
#         "episode_type",
#         "time_gap_min",
#         "HR",
#         "MAP",
#         "RR",
#         "SPO2"
#     ]

#     state["sequence"] = seq_df[features].values

#     return state

# ======================================================
# File: feature_agent.py  [REACTIVE]
# Changes: episode_type → trigger_encoded
#          + 5 new features (hour, night, weekend, age, gender)
#          features_len: 6 → 11
# ======================================================
# ======================================================
# File: feature_agent.py  [REACTIVE]
# Changes: is_weekend REMOVED — 10 features
# ======================================================

SEQ_LEN = 5

FEATURES = [
    "trigger_encoded",
    "time_gap_min",
    "HR", "MAP", "RR", "SPO2",
    "hour_of_day",
    "is_night",
    # is_weekend REMOVED
    "patient_age",
    "gender_encoded",
]

def feature_agent(state):
    print("[Feature Agent]")

    stay_df = state["stay_df"]
    i       = state["index"]

    seq_df = stay_df.iloc[i - SEQ_LEN : i]

    state["sequence"] = seq_df[FEATURES].values
    state["features"] = FEATURES

    return state