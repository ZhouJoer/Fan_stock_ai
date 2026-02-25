from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from graph.state import AgentState
from llm import llm

# 复用清洗逻辑（或者简单复制一份，避免循环依赖utils）
def _clean_history_for_reasoner(messages):
    cleaned = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            cleaned.append(msg)
        elif isinstance(msg, HumanMessage):
            cleaned.append(msg)
        elif isinstance(msg, ToolMessage):
            cleaned.append(HumanMessage(content=f"[数据上下文] {msg.content}"))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                if msg.content:
                    cleaned.append(AIMessage(content=msg.content))
            else:
                cleaned.append(msg)
    return cleaned

# --- 2.5 金融分析师 Agent (Financial Analyst)
analyst_prompt = (
    "你是一个资深的金融分析师。"
    "根据对话历史中提供的数据（股票行情、基本面）以及 交易策略专家 的建议，对股票进行综合分析。"
    "分析趋势、估值情况，并给出最终投资建议（买入、持有或卖出，并说明理由）。"
    "不仅要看技术面，还要结合基本面。"
)

def analyst_node(state: AgentState):
    # 此节点直接使用 LLM
    messages = state['messages'] 
    clean_messages = _clean_history_for_reasoner(messages)
    response = llm.invoke([{"role": "system", "content": analyst_prompt}] + clean_messages)
    return {"messages": [response]}
