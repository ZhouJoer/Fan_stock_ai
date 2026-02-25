"""新闻 Agent 核心实现（仅支持中国 A 股与港股）。

说明：
- 参照附件的分层结构（core/prompts/tools）。
- 使用 LangGraph 的 StateGraph 构建标准的 Agent Graph。
- 当前不接入 workflow，仅提供独立调用入口 run_news_agent。
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm import tool_llm

from .prompts import NEWS_AGENT_INSTRUCTIONS
from .tools import (
    get_ashare_market_news,
    get_hk_market_news,
    get_ashare_stock_news,
    get_hk_stock_news,
)


news_agent_tools = [
    get_ashare_market_news,
    get_hk_market_news,
    get_ashare_stock_news,
    get_hk_stock_news,
]


# 定义 Agent State
class NewsAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 绑定工具到 LLM
llm_with_tools = tool_llm.bind_tools(news_agent_tools)


# Agent 节点：调用 LLM
def call_news_model(state: NewsAgentState):
    messages = state["messages"]
    
    # 如果第一条消息不包含系统提示，添加它
    if not any(m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=NEWS_AGENT_INSTRUCTIONS)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 路由函数：决定继续调用工具还是结束
def should_continue(state: NewsAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果 LLM 返回了工具调用，继续到工具节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则结束
    return END


# 构建 LangGraph
workflow = StateGraph(NewsAgentState)

# 添加节点
workflow.add_node("agent", call_news_model)
workflow.add_node("tools", ToolNode(news_agent_tools))

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# 工具执行后回到 agent
workflow.add_edge("tools", "agent")

# 编译图
news_agent_graph = workflow.compile()


async def run_news_agent(query: str) -> str:
    """运行新闻 Agent 并返回最终文本结果（异步版本，支持并发）。"""
    state = {"messages": [HumanMessage(content=query)]}
    output = await news_agent_graph.ainvoke(state)
    # output['messages'] 为"历史 + 新消息"，这里取最后一条作为最终回答
    last = output["messages"][-1]
    return getattr(last, "content", str(last))
