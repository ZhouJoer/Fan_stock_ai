from graph.state import AgentState
from llm import llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 交易决策者 Agent (Decision Maker)
decision_prompt = (
    "你是一个交易决策者。"
    "你的任务是综合'交易策略专家'和'金融分析师'的意见，生成最终的交易信号。"
    "输出必须包含："
    "1. 决策：买入 / 卖出 / 持有 "
    "2. 理由：简要说明原因"
    "3. 信心分数：0-10分"
    "请给出明确的行动指令。"
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
            # 如果是 AI 消息但包含 tool_calls，R1 可能无法解析，转为文本描述
            if msg.tool_calls:
                content = f"[AI Tool Request]: {msg.tool_calls}"
                cleaned.append(HumanMessage(content=content))
            else:
                cleaned.append(msg)
        # ToolMessage (工具执行结果) 转为 HumanMessage
        elif msg.type == 'tool': 
            content = f"[Tool Result]: {msg.content}"
            cleaned.append(HumanMessage(content=content))
        else:
            # 其他类型转化文本
            cleaned.append(HumanMessage(content=str(msg.content)))
    return cleaned

def decision_node(state: AgentState):
    messages = state['messages']
    
    # 清理历史记录，防止 DeepSeek R1 报错
    clean_messages = _clean_history_for_reasoner(messages)
    
    # 重新构建 prompt 上下文
    prompt_msg = HumanMessage(content=decision_prompt)
    
    # 注意：这里我们把 clean_messages放在前面作为背景，prompt放在最后作为当前指令
    # 或者把 system prompt 放在最前。DeepSeek R1 对 System Message 支持良好，但为了保险，
    # 我们可以把历史作为 context 放在 HumanMessage 里，或者保持 list 结构。
    # 这里保持 invoke([system] + history) 的结构，但 history 已经 clean 过。
    
    final_input = [SystemMessage(content=decision_prompt)] + clean_messages
    
    response = llm.invoke(final_input)
    return {"messages": [response]}
