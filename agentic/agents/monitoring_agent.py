# def monitoring_agent(state):
#     print("[Reactive Monitoring Agent]")

#     trigger_time = state["trigger_time"]
#     predicted_delay = state["predicted_delay"]

#     # actual delay comes directly from dataset
#     actual_delay = state["actual_delay"]

#     deviation = abs(actual_delay - predicted_delay)

#     state["deviation"] = deviation
#     state["reactive_flag"] = True

#     print("Trigger Time:", trigger_time)
#     print("Predicted Delay (hrs):", round(predicted_delay, 2))
#     print("Actual Delay (hrs):", round(actual_delay, 2))
#     print("Deviation (hrs):", round(deviation, 2))

# ======================================================
# File: monitoring_agent.py  [REACTIVE]
# Changes: No feature changes needed
#          Works on delay values only
# ======================================================
# ======================================================
# File: monitoring_agent.py  [REACTIVE]
# No changes needed
# ======================================================

def monitoring_agent(state):
    print("[Reactive Monitoring Agent]")

    trigger_time    = state["trigger_time"]
    predicted_delay = state["predicted_delay"]
    actual_delay    = state["actual_delay"]

    deviation = abs(actual_delay - predicted_delay)

    state["deviation"]     = deviation
    state["reactive_flag"] = True

    print("Trigger Time:          ", trigger_time)
    print("Predicted Delay (hrs): ", round(predicted_delay, 2))
    print("Actual Delay (hrs):    ", round(actual_delay, 2))
    print("Deviation (hrs):       ", round(deviation, 2))

    return state