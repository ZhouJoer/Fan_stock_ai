from graph.state import AgentState
from llm import llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 风控经理 Agent (Risk Manager)
risk_prompt = (
    "你是一个风控经理。"
    "你的任务是审核'交易决策者'的建议。"
    "1. 如果建议是'买入'，请检查潜在风险（如近期涨幅过大、技术指标背离、大盘环境差等）。如果不符合风控标准，请'拒绝'。"
    "2. 如果建议是'卖出'，确认是否需要'清仓'或'减仓'。"
    "3. 如果建议是'持有'，评估是否需要设置止损点。"
    "输出最终的风控裁决：【通过】 或 【拒绝】 并说明原因。"
)

def _clean_history_for_reasoner(messages):
    """
    清理消息历史，适配 DeepSeek R1 (Reasoner) 模型：
    1. 移除 Reasoner 不支持的 ToolMessage
    2. 将带有 tool_calls 的 AIMessage 转换为纯文本 HumanMessage (包含工具调用信息)
    3. 确保传递给 Reasoner 的只有 System, Human, AI (纯文本) 消息
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            cleaned.append(msg)
        elif isinstance(msg, HumanMessage):
            cleaned.append(msg)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                content = f"[AI Tool Request]: {msg.tool_calls}"
                cleaned.append(HumanMessage(content=content))
            else:
                cleaned.append(msg)
        elif msg.type == 'tool': 
            content = f"[Tool Result]: {msg.content}"
            cleaned.append(HumanMessage(content=content))
        else:
            cleaned.append(HumanMessage(content=str(msg.content)))
    return cleaned

def risk_node(state: AgentState):
    messages = state['messages']
    
    # 清理历史记录
    clean_messages = _clean_history_for_reasoner(messages)
    
    final_input = [SystemMessage(content=risk_prompt)] + clean_messages
    
    response = llm.invoke(final_input)
    return {"messages": [response]}
