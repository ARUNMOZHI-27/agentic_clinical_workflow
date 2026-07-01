# import numpy as np

# SEQ_LEN = 5
# FEATURES = ["episode_type", "time_gap_min", "HR", "MAP", "RR", "SPO2"]

# def feature_agent(state):
#     print("[Feature Agent]")

#     stay_df = state["stay_df"]

#     if len(stay_df) < SEQ_LEN:
#         state["stop"] = True
#         return state
#     seq_df = stay_df.tail(SEQ_LEN).copy()

#     state["sequence_df"] = seq_df[FEATURES]
#     state["stop"] = False

#     return state

# ======================================================
# File: feature_agent.py  [PROACTIVE]
# Changes: episode_type → trigger_encoded
#          + 5 new features (hour, night, weekend, age, gender)
# ======================================================
# ======================================================
# File: feature_agent.py  [PROACTIVE]
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

    if len(stay_df) < SEQ_LEN:
        state["stop"] = True
        return state

    seq_df = stay_df.tail(SEQ_LEN).copy()

    state["sequence_df"] = seq_df[FEATURES]
    state["stop"]        = False

    return state