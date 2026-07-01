def alert_agent(state):
    print("[Alert Agent]")

    if state.get("lab_taken", False):
        print("No alert — lab already completed.")
        return state

    if state.get("alert_flag", False):
        print("🚨 ALERT: Followup action Needed!")
    else:
        print("No proactive alert.")

    return state