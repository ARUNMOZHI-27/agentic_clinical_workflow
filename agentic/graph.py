from langgraph.graph import StateGraph
from agentic.agents.trigger_agent import trigger_agent
from agentic.agents.feature_agent import feature_agent
from agentic.agents.prediction_agent import prediction_agent
from agentic.agents.monitoring_agent import monitoring_agent
from agentic.agents.severity_agent import severity_agent
from agentic.agents.alert_agent import alert_agent
from agentic.agents.explanation_agent import explanation_agent

builder = StateGraph(dict)

builder.add_node("trigger", trigger_agent)
builder.add_node("feature", feature_agent)
builder.add_node("predict", prediction_agent)
builder.add_node("monitor", monitoring_agent)
builder.add_node("severity", severity_agent)
builder.add_node("explain", explanation_agent)
builder.add_node("alert", alert_agent)

builder.set_entry_point("trigger")

builder.add_edge("trigger", "feature")
builder.add_edge("feature", "predict")
builder.add_edge("predict", "monitor")
builder.add_edge("monitor", "severity")
builder.add_edge("severity", "explain")
builder.add_edge("explain", "alert")

graph = builder.compile()