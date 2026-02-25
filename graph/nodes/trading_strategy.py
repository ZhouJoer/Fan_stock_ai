from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from graph.state import AgentState
from llm import llm

# --- 2. 交易策略专家 Agent (Trading Strategy Expert)
strategy_prompt = (
    "你是一位量化交易策略专家。"
    "根据提供的数据（特别是 MACD, EMA, BOLL 等技术指标），判断当前的市场状态。"
    "1. 判断当前是'趋势状态'还是'震荡状态'。"
    "2. 给出具体的交易策略建议（例如趋势跟踪、均值回归）。"
    "请基于数据给出明确的信号判断。"
)

def _clean_history_for_reasoner(messages):
    """
    清洗消息历史，适配 DeepSeek Reasoner (R1)。
    R1 不支持 Tool Call，也不支持历史记录中包含 tool_calls 字段的消息。
    我们将 ToolMessage 转换为普通的 System/Human 消息，并将之前的 Tool Call 消息过滤或转化为文本。
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            cleaned.append(msg)
        elif isinstance(msg, HumanMessage):
            cleaned.append(msg)
        elif isinstance(msg, ToolMessage):
            # 将工具输出（数据）转换为 HumanMessage，以便模型作为上下文读取
            # 添加前缀说明这是数据
            cleaned.append(HumanMessage(content=f"[系统数据] {msg.content}"))
        elif isinstance(msg, AIMessage):
            # 检查是否有 tool_calls
            if msg.tool_calls:
                # 只有 tool_calls 没有 content 的消息，R1 可能会报错
                # 我们将其转换为文本说明，或者直接忽略（因为下面的 ToolMessage 才是关键数据）
                # 这里选择忽略纯 Tool Call 消息，或者保留其 content 部分（如果有）
                if msg.content:
                    cleaned.append(AIMessage(content=msg.content))
            else:
                cleaned.append(msg)
        else:
            # 其他类型消息直接转文本
            if hasattr(msg, "content") and msg.content:
                cleaned.append(HumanMessage(content=str(msg.content)))
    return cleaned

def strategy_node(state: AgentState):
    messages = state['messages']
    # 清洗历史消息，移除工具调用痕迹，只保留数据内容
    clean_messages = _clean_history_for_reasoner(messages)
    
    # 构造请求
    response = llm.invoke([{"role": "system", "content": strategy_prompt}] + clean_messages)
    return {"messages": [response]}
