# from datetime import timedelta

# def monitoring_agent(state):
#     print("[Proactive Monitoring Agent]")

#     trigger_time = state["current_trigger_time"]
#     predicted_delay = state["predicted_delay_hours"]
#     current_time = state["current_time"]
#     labs_df = state["labs_df"]
#     stay_df = state["stay_df"]

#     hadm_id = stay_df.iloc[-1]["hadm_id"]

#     expected_lab_time = trigger_time + timedelta(hours=predicted_delay)

#     print("Trigger Time:", trigger_time)
#     print("Predicted Delay (hrs):", round(predicted_delay, 2))
#     print("Expected Lab Time:", expected_lab_time)
#     print("Current Time:", current_time)

#     # Find actual first lab after trigger
#     future_labs = labs_df[
#         (labs_df["hadm_id"] == hadm_id) &
#         (labs_df["charttime"] > trigger_time)
#     ].sort_values("charttime")

#     if not future_labs.empty:
#         actual_lab_time = future_labs.iloc[0]["charttime"]
#         actual_delay = (actual_lab_time - trigger_time).total_seconds() / 3600
#         state["actual_delay_hours"] = actual_delay

#         print("Actual First Lab Time:", actual_lab_time)
#         print("Actual True Delay (hrs):", round(actual_delay, 2))
#     else:
#         state["actual_delay_hours"] = None
#         print("No lab found after trigger.")

#     # Check if lab already happened by current_time
#     patient_labs = labs_df[
#         (labs_df["hadm_id"] == hadm_id) &
#         (labs_df["charttime"] > trigger_time) &
#         (labs_df["charttime"] <= current_time)
#     ]

#     if not patient_labs.empty:
#         print("Lab already completed before monitoring time.")
#         state["alert_flag"] = False
#         state["lab_taken"] = True
#     else:
#         tolerance = timedelta(minutes=15)

#         if current_time > expected_lab_time + tolerance:
#             print("Expected time exceeded.")
#             state["alert_flag"] = True
#         else:
#             print("Still within expected window.")
#             state["alert_flag"] = False

#         state["lab_taken"] = False

#     print("Proactive Alert:", state["alert_flag"])
#     print("-" * 50)

#     return state

# ======================================================
# File: monitoring_agent.py  [PROACTIVE]
# Changes: No changes needed
# ======================================================
# ======================================================
# File: monitoring_agent.py  [PROACTIVE]
# No changes needed
# ======================================================

from datetime import timedelta

def monitoring_agent(state):
    print("[Proactive Monitoring Agent]")

    trigger_time    = state["current_trigger_time"]
    predicted_delay = state["predicted_delay_hours"]
    current_time    = state["current_time"]
    labs_df         = state["labs_df"]
    stay_df         = state["stay_df"]

    hadm_id           = stay_df.iloc[-1]["hadm_id"]
    expected_lab_time = trigger_time + timedelta(hours=predicted_delay)

    print("Trigger Time:          ", trigger_time)
    print("Predicted Delay (hrs): ", round(predicted_delay, 2))
    print("Expected Lab Time:     ", expected_lab_time)
    print("Current Time:          ", current_time)

    future_labs = labs_df[
        (labs_df["hadm_id"] == hadm_id) &
        (labs_df["charttime"] > trigger_time)
    ].sort_values("charttime")

    if not future_labs.empty:
        actual_lab_time             = future_labs.iloc[0]["charttime"]
        actual_delay                = (
            actual_lab_time - trigger_time
        ).total_seconds() / 3600
        state["actual_delay_hours"] = actual_delay
        print("Actual First Lab Time:  ", actual_lab_time)
        print("Actual True Delay (hrs):", round(actual_delay, 2))
    else:
        state["actual_delay_hours"] = None
        print("No lab found after trigger.")

    patient_labs = labs_df[
        (labs_df["hadm_id"] == hadm_id) &
        (labs_df["charttime"] > trigger_time) &
        (labs_df["charttime"] <= current_time)
    ]

    if not patient_labs.empty:
        print("Lab already completed before monitoring time.")
        state["alert_flag"] = False
        state["lab_taken"]  = True
    else:
        tolerance = timedelta(minutes=15)
        if current_time > expected_lab_time + tolerance:
            print("Expected time exceeded.")
            state["alert_flag"] = True
        else:
            print("Still within expected window.")
            state["alert_flag"] = False
        state["lab_taken"] = False

    print("Proactive Alert:", state["alert_flag"])
    print("-" * 50)

    return state