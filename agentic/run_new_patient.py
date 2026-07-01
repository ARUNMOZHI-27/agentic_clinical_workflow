import os
from datetime import datetime
from graph import graph
import pandas as pd
# ==============================
# Create Synthetic 5 Triggers
# ==============================

new_patient_df = pd.DataFrame([
    {"episode_type": 0, "time_gap_min": 0, "HR": 125, "MAP": 70, "RR": 22, "SPO2": 95},
    {"episode_type": 0, "time_gap_min": 30, "HR": 130, "MAP": 68, "RR": 24, "SPO2": 94},
    {"episode_type": 0, "time_gap_min": 45, "HR": 135, "MAP": 65, "RR": 25, "SPO2": 93},
    {"episode_type": 0, "time_gap_min": 60, "HR": 140, "MAP": 63, "RR": 26, "SPO2": 92},
    {"episode_type": 0, "time_gap_min": 90, "HR": 145, "MAP": 60, "RR": 28, "SPO2": 91}
])

print("\nRunning Agentic NEW Patient Monitoring...\n")

initial_state = {
    "stay_id": 999999,
    "stay_df": new_patient_df,
    "index": len(new_patient_df)-1, # simulate after 5 triggers
    "trigger_time": datetime.now()
}

graph.invoke(initial_state)

print("\nNew Patient Monitoring Complete ✅")