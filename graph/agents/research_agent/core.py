"""研究 Agent 核心实现（仅支持中国 A 股与港股）。

说明：
- 参照附件的分层结构（core/prompts/tools）。
- 使用 LangGraph 的 StateGraph 构建标准的 Agent Graph。
- 当前不接入 workflow，仅提供独立调用入口 run_research_agent。
"""

from __future__ import annotations

import asyncio
from typing import Annotated, AsyncIterator, Callable
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm import tool_llm

from tools.cninfo_tools import fetch_cninfo_filings

from tools.akshare_tools import (
    get_stock_code,
    get_stock_history,
    get_stock_info,
    get_hk_stock_code,
    get_hk_stock_history,
    get_hk_stock_info,
)

from .prompts import RESEARCH_AGENT_INSTRUCTIONS
from .knowledge import (
    search_knowledge_base,
    save_important_info,
    analyze_pdf_document,
    analyze_markdown_document,
)


research_agent_tools = [
    # A 股
    get_stock_code,
    get_stock_history,
    get_stock_info,
    # A 股公告/定期报告（CNINFO）
    fetch_cninfo_filings,
    # 港股
    get_hk_stock_code,
    get_hk_stock_history,
    get_hk_stock_info,
    # 知识库工具
    search_knowledge_base,
    save_important_info,
    # 文档分析工具（新增）
    analyze_pdf_document,
    analyze_markdown_document,
]


# 定义 Agent State
class ResearchAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 绑定工具到 LLM
llm_with_tools = tool_llm.bind_tools(research_agent_tools)


# Agent 节点：调用 LLM
def call_research_model(state: ResearchAgentState):
    messages = state["messages"]
    
    # 如果第一条消息不包含系统提示，添加它
    if not any(m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=RESEARCH_AGENT_INSTRUCTIONS)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 路由函数：决定继续调用工具还是结束
def should_continue(state: ResearchAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果 LLM 返回了工具调用，继续到工具节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则结束
    return END


# 构建 LangGraph
workflow = StateGraph(ResearchAgentState)

# 添加节点
workflow.add_node("agent", call_research_model)
workflow.add_node("tools", ToolNode(research_agent_tools))

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
research_agent_graph = workflow.compile()


# 工具名到中文描述的映射，用于进度与步骤展示
TOOL_NAME_LABELS: dict[str, str] = {
    "get_stock_code": "根据名称查A股代码",
    "get_stock_history": "获取历史行情",
    "get_stock_info": "获取股票基本信息",
    "fetch_cninfo_filings": "获取CNINFO公告/定期报告",
    "get_hk_stock_code": "根据名称查港股代码",
    "get_hk_stock_history": "获取港股历史行情",
    "get_hk_stock_info": "获取港股基本信息",
    "search_knowledge_base": "知识库语义搜索",
    "save_important_info": "保存重要信息到知识库",
    "analyze_pdf_document": "分析PDF文档",
    "analyze_markdown_document": "分析Markdown文档",
}


def _step_description_from_update(update: dict) -> str:
    """从 astream(stream_mode='updates') 的单个 update 中提取可读步骤描述。"""
    messages = update.get("messages") or []
    if not messages:
        return "执行中…"
    msg = messages[-1]
    if isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            names = [tc.get("name", "unknown") for tc in tool_calls]
            labels = [TOOL_NAME_LABELS.get(n, n) for n in names]
            return "准备调用：" + "、".join(labels)
        if (getattr(msg, "content", None) or "").strip():
            return "Agent 回答中…"
        return "Agent 思考中…"
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "tool")
        return "工具返回：" + TOOL_NAME_LABELS.get(name, name)
    return "执行中…"


def _messages_to_steps(messages: list, user_query: str) -> list[dict]:
    """将消息列表转为可展示的步骤列表。"""
    from langchain_core.messages import HumanMessage

    steps = []
    step_num = 0

    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            step_num += 1
            content = (getattr(msg, "content", None) or "") if hasattr(msg, "content") else str(msg)
            steps.append({
                "step": step_num,
                "type": "user",
                "title": "用户提问",
                "detail": content[:500] + ("..." if len(content) > 500 else ""),
            })
        elif isinstance(msg, AIMessage):
            content = (getattr(msg, "content", None) or "") if hasattr(msg, "content") else ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    step_num += 1
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    args_str = str(args)[:300] + ("..." if len(str(args)) > 300 else "")
                    steps.append({
                        "step": step_num,
                        "type": "tool_call",
                        "title": f"调用工具：{name}",
                        "detail": f"参数：{args_str}",
                    })
            elif content.strip():
                step_num += 1
                steps.append({
                    "step": step_num,
                    "type": "agent",
                    "title": "Agent 回答",
                    "detail": content[:2000] + ("..." if len(content) > 2000 else ""),
                })
        elif isinstance(msg, ToolMessage):
            step_num += 1
            name = getattr(msg, "name", "tool")
            content = (getattr(msg, "content", None) or "") if hasattr(msg, "content") else str(msg)
            preview = content[:400].replace("\n", " ").strip() + ("..." if len(content) > 400 else "")
            steps.append({
                "step": step_num,
                "type": "tool_result",
                "title": f"工具返回：{name}",
                "detail": preview,
            })

    return steps


# 研究 Agent 可能多轮调用工具（查代码、行情、公告、知识库等），默认 25 步易触顶，提高到 50
RESEARCH_RECURSION_LIMIT = 50


async def run_research_agent(query: str, include_steps: bool = True) -> dict:
    """
    运行研究 Agent 并返回结果与步骤（异步版本）。

    Returns:
        dict: {"result": str, "steps": list[dict]}，若 include_steps=False 则 steps 为空列表。
    """
    state = {"messages": [HumanMessage(content=query)]}
    config = {"recursion_limit": RESEARCH_RECURSION_LIMIT}
    output = await research_agent_graph.ainvoke(state, config=config)
    messages = output.get("messages", [])
    last = messages[-1] if messages else None
    result = getattr(last, "content", str(last)) if last else ""

    steps = _messages_to_steps(messages, query) if include_steps else []
    return {"result": result or "", "steps": steps}


async def run_research_agent_stream(
    query: str,
    *,
    get_cancel_flag: Callable[[], bool] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """
    流式运行研究 Agent，每步 yield ("progress", {...})，结束时 yield ("done", {result, steps}) 或 ("cancelled", {}) / ("error", {detail})。
    get_cancel_flag: 可选的无参可调用对象，返回 True 时中止并 yield ("cancelled", {})。
    """
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    async def produce() -> None:
        try:
            state = {"messages": [HumanMessage(content=query)]}
            config = {"recursion_limit": RESEARCH_RECURSION_LIMIT}
            step = 0
            last_state: dict | None = None
            async for chunk in research_agent_graph.astream(
                state, config, stream_mode=["updates", "values"]
            ):
                if get_cancel_flag and get_cancel_flag():
                    await queue.put(("cancelled", {}))
                    return
                # 多 mode 时 chunk 为 (mode, payload)
                if isinstance(chunk, (list, tuple)) and len(chunk) == 2:
                    mode, payload = chunk[0], chunk[1]
                    if mode == "updates" and isinstance(payload, dict):
                        for _node_name, node_update in payload.items():
                            if isinstance(node_update, dict):
                                step += 1
                                desc = _step_description_from_update(node_update)
                                await queue.put((
                                    "progress",
                                    {"step": step, "message": desc, "total_hint": "多轮工具调用中"},
                                ))
                    elif mode == "values" and isinstance(payload, dict):
                        last_state = payload
                else:
                    # 兼容单 mode 的 yield
                    last_state = chunk if isinstance(chunk, dict) else last_state
            if last_state:
                messages = last_state.get("messages", [])
                last = messages[-1] if messages else None
                result = getattr(last, "content", str(last)) if last else ""
                steps = _messages_to_steps(messages, query)
                await queue.put(("done", {"result": result or "", "steps": steps}))
            else:
                await queue.put(("done", {"result": "", "steps": []}))
        except asyncio.CancelledError:
            await queue.put(("cancelled", {}))
        except Exception as e:
            await queue.put(("error", {"detail": str(e)}))

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                if get_cancel_flag and get_cancel_flag():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    yield ("cancelled", {})
                    return
                continue
            kind, payload = item
            yield (kind, payload)
            if kind in ("done", "error", "cancelled"):
                return
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
