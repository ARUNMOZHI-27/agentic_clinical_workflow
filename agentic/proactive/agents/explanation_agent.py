

from langchain_ollama import ChatOllama

def explanation_agent(state):
    print("[Explanation Agent - LLM]")

    predicted = state.get("predicted_delay_hours")
    actual = state.get("actual_delay_hours")
    severity = state.get("severity")
    stay_df = state.get("stay_df")

    if predicted is None:
        print("No prediction available.")
        return state

    last_row = stay_df.iloc[-1]

    hr = last_row.get("HR")
    map_val = last_row.get("MAP")
    rr = last_row.get("RR")
    spo2 = last_row.get("SPO2")
    episode = last_row.get("episode_type")

    episode_label = "cardiac" if episode == 0 else "respiratory"

    prompt = f"""
You are a clinical workflow monitoring assistant.

Episode type: {episode_label}
Predicted expected lab delay: {predicted:.2f} hours
Actual lab delay: {actual}
Severity classification: {severity}

Latest vitals:
Heart Rate: {hr}
MAP: {map_val}
Respiratory Rate: {rr}
SpO2: {spo2}

Explain clearly:
- What this delay means
- Whether workflow inefficiency exists
- Potential risks
Be professional and concise.
"""

    llm = ChatOllama(
        model="llama3.2",
        temperature=0.2
    )

    response = llm.invoke(prompt)

    explanation = response.content

    print("\nLLM Explanation:\n")
    print(explanation)
    print("-" * 50)

    state["llm_explanation"] = explanation

    return state
