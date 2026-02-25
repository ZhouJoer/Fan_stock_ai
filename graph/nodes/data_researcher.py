from graph.state import AgentState
from graph.utils import create_agent
from llm import llm, tool_llm
from tools.akshare_tools import get_stock_code, get_stock_history, get_stock_info

# 1. 数据研究员 Agent (Data Researcher)
data_researcher_tools = [get_stock_code, get_stock_history, get_stock_info]
data_researcher_prompt = (
    "你是一个股票数据研究员。"
    "你的任务是使用工具获取用户请求的股票的相关数据。"
    "首先确定股票代码，然后获取历史行情和基本信息。"
    "获取到数据后，确保输出包含具体的数据内容，不要只是说'已获取'。"
)
# 注意：使用 tool_llm (deepseek-chat) 避免 reasoner 模型在工具调用时出现格式问题
data_researcher_agent = create_agent(tool_llm, data_researcher_tools, data_researcher_prompt)

def data_researcher_node(state: AgentState):
    output = data_researcher_agent.invoke(state)
    # create_react_agent 管理完整的状态交互。
    # 我们希望返回由 agent 生成的 *新* 消息。
    # output['messages'] 包含 (历史 + 新消息)。
    # 我们需要过滤并只返回新消息，以避免在主图中 'operator.add' 合并时重复。
    
    # 确定新消息从哪里开始。
    # 输入的 state['messages'] 是我们传入的。
    # 我们可以检查长度。
    original_count = len(state['messages'])
    new_messages = output['messages'][original_count:]
    
    return {"messages": new_messages}
