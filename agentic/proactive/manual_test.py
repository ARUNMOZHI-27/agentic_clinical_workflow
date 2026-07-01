

import pandas as pd
from datetime import datetime, timedelta
from agentic.proactive.graph import graph

print("Synthetic Manual Proactive Test\n")

# =====================================================
# CREATE SYNTHETIC PATIENT DATA
# =====================================================

data = {
    "stay_id": [999999]*5,
    "hadm_id": [888888]*5,
    "trigger_time": [
        datetime(2026,1,1,8,0),
        datetime(2026,1,1,10,0),
        datetime(2026,1,1,12,0),
        datetime(2026,1,1,14,0),
        datetime(2026,1,1,16,0),
    ],
    "episode_type": [0,0,0,0,0], # cardiac
    "HR": [110,115,120,118,122],
    "MAP": [65,60,58,55,52],
    "RR": [20,22,24,26,28],
    "SPO2": [95,94,93,92,90],
    "time_gap_min": [0,120,120,120,120]
}

synthetic_df = pd.DataFrame(data)

print("Synthetic Trigger Data:")
print(synthetic_df[["trigger_time","HR","MAP"]], "\n")

# =====================================================
# SIMULATE CURRENT TIME
# =====================================================

trigger_time = synthetic_df.iloc[-1]["trigger_time"]

# Simulate 6 hours later
simulated_current_time = trigger_time + timedelta(hours=6)

print("Trigger Time:", trigger_time)
print("Simulated Current Time:", simulated_current_time, "\n")

# =====================================================
# NO LAB EXISTS
# =====================================================

labs_df = pd.DataFrame(columns=["hadm_id","charttime"])

# =====================================================
# RUN GRAPH
# =====================================================

initial_state = {
    "stay_df": synthetic_df,
    "labs_df": labs_df,
    "current_time": simulated_current_time
}

graph.invoke(initial_state)

print("\nSynthetic Manual Test Complete ✅")