# def trigger_agent(state):
#     row = state["stay_df"].iloc[state["index"]]

#     state["trigger_time"] = row["trigger_time"]
#     state["episode_type"] = row["episode_type"]
#     state["actual_delay"] = row["delay_hours"]

#     print("\n[Trigger Agent]")
#     print("Trigger Time:", state["trigger_time"])

#     return state

# ======================================================
# File: trigger_agent.py  [REACTIVE]
# Changes: episode_type removed
#          trigger_type + trigger_encoded added
# ======================================================

def trigger_agent(state):
    print("[Trigger Agent]")

    row = state["stay_df"].iloc[state["index"]]

    state["trigger_time"]    = row["trigger_time"]
    state["trigger_type"]    = row["trigger_type"]      # NEW
    state["trigger_encoded"] = row["trigger_encoded"]   # NEW
    state["actual_delay"]    = row["delay_hours"]

    print("Trigger Time:", state["trigger_time"])
    print("Trigger Type:", state["trigger_type"])

    return state