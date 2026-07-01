# def severity_agent(state):
#     print("[Severity Agent]")

#     predicted = state.get("predicted_delay_hours")
#     actual = state.get("actual_delay_hours")

#     if predicted is None:
#         print("Predicted delay missing.")
#         state["severity"] = None
#         return state

#     if actual is None:
#         print("Actual delay not available (lab not occurred yet).")
#         state["severity"] = "Pending"
#         return state

#     deviation = actual - predicted
#     state["deviation_hours"] = deviation

#     print("Predicted Delay (hrs):", round(predicted, 2))
#     print("Actual Delay (hrs):", round(actual, 2))
#     print("Deviation (hrs):", round(deviation, 2))

#     # Severity rules
#     if deviation <= 0:
#         severity = "No Delay"
#     elif deviation <= 1:
#         severity = "Mild"
#     elif deviation <= 3:
#         severity = "Moderate"
#     else:
#         severity = "Severe"

#     state["severity"] = severity

#     print("Severity Level:", severity)
#     print("-" * 50)

# ======================================================
# File: severity_agent.py  [PROACTIVE]
# Changes: No changes needed
# ======================================================

def severity_agent(state):
    print("[Severity Agent]")

    predicted = state.get("predicted_delay_hours")
    actual    = state.get("actual_delay_hours")

    if predicted is None:
        print("Predicted delay missing.")
        state["severity"] = None
        return state

    if actual is None:
        print("Actual delay not available (lab not occurred yet).")
        state["severity"] = "Pending"
        return state

    deviation               = actual - predicted
    state["deviation_hours"] = deviation

    print("Predicted Delay (hrs):", round(predicted, 2))
    print("Actual Delay (hrs):   ", round(actual, 2))
    print("Deviation (hrs):      ", round(deviation, 2))

    if deviation <= 0:
        severity = "No Delay"
    elif deviation <= 1:
        severity = "Mild"
    elif deviation <= 3:
        severity = "Moderate"
    else:
        severity = "Severe"

    state["severity"] = severity

    print("Severity Level:", severity)
    print("-" * 50)

    return state