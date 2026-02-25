from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.data_researcher import data_researcher_node
from graph.nodes.trading_strategy import strategy_node
from graph.nodes.financial_analyst import analyst_node
from graph.nodes.decision_maker import decision_node
from graph.nodes.risk_manager import risk_node
from graph.nodes.quant_trader import quant_trader_node
from graph.nodes.quant_auditor import quant_auditor_node
from graph.nodes.etf_rotation_ai import etf_rotation_ai_node
from graph.nodes.supervisor import supervisor_node, members

# --- 构建图 (Build Graph) ---

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Data_Researcher", data_researcher_node)
workflow.add_node("Trading_Strategy", strategy_node)
workflow.add_node("Financial_Analyst", analyst_node)
workflow.add_node("Decision_Maker", decision_node)
workflow.add_node("Risk_Manager", risk_node)
workflow.add_node("Quant_Trader", quant_trader_node)
workflow.add_node("Quant_Auditor", quant_auditor_node)
workflow.add_node("ETF_Rotation_AI", etf_rotation_ai_node)

for member in members:
    workflow.add_edge(member, "Supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("Supervisor")

graph = workflow.compile()
