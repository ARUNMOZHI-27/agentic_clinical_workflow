
# def trigger_agent(state):
#     print("[Trigger Agent]")

#     row = state["stay_df"].iloc[-1]

#     state["current_trigger_time"] = row["trigger_time"]
#     state["current_stay_id"] = row["stay_id"]

#     print("Trigger Time:", state["current_trigger_time"])

#     return state

# ======================================================
# File: trigger_agent.py  [PROACTIVE]
# Changes: No changes needed
# ======================================================

def trigger_agent(state):
    print("[Trigger Agent]")

    row = state["stay_df"].iloc[-1]

    state["current_trigger_time"] = row["trigger_time"]
    state["current_stay_id"]      = row["stay_id"]

    print("Trigger Time:", state["current_trigger_time"])

    return state