"""Research、News、Health 路由"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph.agents.news_agent.core import run_news_agent
from graph.agents.research_agent.core import run_research_agent, run_research_agent_stream


router = APIRouter(prefix="/api", tags=["research_news"])


class ResearchRequest(BaseModel):
    query: str


class NewsRequest(BaseModel):
    query: str


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/research")
async def research(body: ResearchRequest = Body(..., embed=False)):
    """研究 Agent：分析 CNINFO 等数据，返回研究结果与执行步骤。请求体 JSON: {"query": "问题"}"""
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    try:
        data = await run_research_agent(query, include_steps=True)
        return {
            "result": data.get("result") or "",
            "steps": data.get("steps") or [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/stream")
async def research_stream(request: Request, body: ResearchRequest = Body(..., embed=False)):
    """研究 Agent 流式接口：SSE 推送进度与最终结果，客户端断开即视为取消（不设自动超时）。"""
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    cancel = asyncio.Event()

    async def check_disconnect() -> None:
        while not cancel.is_set():
            await asyncio.sleep(0.3)
            if await request.is_disconnected():
                cancel.set()
                return

    def get_cancel_flag() -> bool:
        return cancel.is_set()

    async def event_generator():
        disconnect_task = asyncio.create_task(check_disconnect())
        try:
            async for kind, payload in run_research_agent_stream(query, get_cancel_flag=get_cancel_flag):
                yield f"data: {json.dumps({'type': kind, **payload})}\n\n"
                if kind in ("done", "error", "cancelled"):
                    break
        finally:
            cancel.set()
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/news")
async def news(body: NewsRequest = Body(..., embed=False)):
    """新闻 Agent：抓取市场/个股新闻。请求体 JSON: {"query": "问题"}"""
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    try:
        result = await run_news_agent(query)
        return {"result": result if result is not None else ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
