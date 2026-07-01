# def alert_agent(state):
#     print("[Alert Agent]")

#     print("Actual Delay:", round(state["actual_delay"], 2))
#     print("Deviation:", round(state["deviation"], 2))

#     print("Proactive Alert:",
#           state["proactive_flag"])

#     print("Reactive Flag:",
#           state["reactive_flag"])

#     print("-" * 50)

#     return state

def alert_agent(state):
    print("[Alert Agent]")

    print("Actual Delay:   ", round(state["actual_delay"], 2))
    print("Deviation:      ", round(state["deviation"], 2))
    print("Proactive Alert:", state.get("proactive_flag", False))  # safe default
    print("Reactive Flag:  ", state.get("reactive_flag", False))

    print("-" * 50)
    return state