"""量化回测、Agent、AI 决策等路由"""
from __future__ import annotations

import logging
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph.workflow import graph as workflow_graph
from modules.qmt import get_qmt_tools


router = APIRouter(prefix="/api", tags=["quant"])


class QuantBacktestRequest(BaseModel):
    stock_code: str
    strategy: str = "multi_factor"
    strategy_type: str = "trend"
    risk_preference: str = "balanced"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000


class QuantAgentRequest(BaseModel):
    query: str


class AITradingDecisionRequest(BaseModel):
    stock_code: str
    has_position: bool = False
    entry_price: Optional[float] = None
    days_held: Optional[int] = None
    strategy_type: str = "trend"
    risk_preference: str = "balanced"


class AIDecisionBacktestRequest(BaseModel):
    stock_code: str
    initial_capital: float = 100000
    days: int = 252
    strategy_type: str = "trend"
    risk_preference: str = "balanced"
    use_llm_signals: bool = False
    llm_sample_rate: int = 10


@router.post("/quant/backtest")
async def quant_backtest(req: QuantBacktestRequest):
    """普通量化回测（不走AI Agent，直接返回回测结果）"""
    try:
        from tools.stock_data import get_stock_data
        from modules.strategy_config import get_strategy_config
        import pandas as pd

        qmt = get_qmt_tools()
        strategy_config = get_strategy_config(req.strategy_type, req.risk_preference)
        days = 365
        stock_data = get_stock_data(req.stock_code, days)
        df = pd.DataFrame(stock_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        start_date = df.index[0].strftime("%Y%m%d")
        end_date = df.index[-1].strftime("%Y%m%d")
        df = qmt.calculate_technical_indicators(df)
        effective_strategy = req.strategy
        if req.strategy in ['trend', 'mean_reversion', 'chanlun']:
            effective_strategy = req.strategy
        elif req.strategy_type in ['trend', 'mean_reversion', 'chanlun'] and req.strategy == 'multi_factor':
            effective_strategy = req.strategy_type
        df = qmt.generate_trading_signals(df, effective_strategy, strategy_config)
        backtest_result = qmt.backtest(df, initial_capital=req.initial_capital, strategy_config=strategy_config)
        chart_base64 = qmt.generate_backtest_charts(df, backtest_result)
        backtest_result.update({
            "stock_code": req.stock_code,
            "strategy": effective_strategy,
            "strategy_type": req.strategy_type,
            "risk_preference": req.risk_preference,
            "strategy_config": {
                "name": strategy_config.get('full_name', ''),
                "position_size": f"{strategy_config.get('position_size', 0.7)*100:.0f}%",
                "stop_loss": f"{strategy_config.get('stop_loss_pct', 0.05)*100:.1f}%",
                "take_profit": f"{strategy_config.get('take_profit_pct', 0.12)*100:.1f}%",
            },
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": req.initial_capital,
            "chart": chart_base64
        })
        return {"result": backtest_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quant/agent")
async def quant_agent(req: QuantAgentRequest):
    """AI Agent量化分析（走完整的量化工作流）"""
    query = (req.query or "").strip()
    if not query:
        return {"error": "query 不能为空"}
    try:
        if not any(kw in query for kw in ["量化", "回测", "策略"]):
            query = f"量化分析：{query}"
        state = {"messages": [HumanMessage(content=query)]}
        output = await workflow_graph.ainvoke(state)
        all_messages = []
        for msg in output["messages"]:
            content = getattr(msg, "content", str(msg))
            name = getattr(msg, "name", "User")
            all_messages.append({"name": name, "content": content})
        return {
            "result": all_messages[-1]["content"] if all_messages else "",
            "full_workflow": all_messages
        }
    except Exception as e:
        logging.exception("quant/agent 执行失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quant/ai-decision")
async def ai_trading_decision(req: AITradingDecisionRequest):
    """AI 实时交易决策（基于技术指标综合判断）"""
    try:
        from tools.technical_indicators import calculate_technical_indicators
        from prompts import ai_trading_strategy
        from tools.stock_data import get_stock_data

        stock_data = get_stock_data(req.stock_code, days=120)
        indicators = calculate_technical_indicators(stock_data)
        historical_trades = []
        if req.has_position and req.entry_price:
            historical_trades.append({
                'status': 'open',
                'entry_price': req.entry_price,
                'days': req.days_held or 0
            })
        decision, ai_analysis = ai_trading_strategy(
            req.stock_code,
            indicators,
            historical_trades,
            strategy_type=req.strategy_type,
            risk_preference=req.risk_preference
        )
        return {
            "result": {
                "stock_code": req.stock_code,
                "strategy_type": req.strategy_type,
                "risk_preference": req.risk_preference,
                "indicators": indicators,
                "decision": decision,
                "ai_analysis": ai_analysis
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quant/ai-backtest")
async def ai_decision_backtest_api(req: AIDecisionBacktestRequest):
    """AI 决策策略回测（验证AI策略在历史数据上的表现）"""
    try:
        from modules.backtest import ai_decision_backtest

        result = ai_decision_backtest(
            stock_code=req.stock_code,
            initial_capital=req.initial_capital,
            days=req.days,
            strategy_type=req.strategy_type,
            risk_preference=req.risk_preference,
            use_llm_signals=req.use_llm_signals,
            llm_sample_rate=req.llm_sample_rate
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
