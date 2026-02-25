"""ETF 轮动、ETF 模拟盘、ETF 实盘路由"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from api.auth import get_current_user, get_user_etf_sim_dir
from graph.workflow import graph as workflow_graph

# 按 session 隔离的停止信号，支持多用户同时回测
_etf_abort_events: dict[str, threading.Event] = {}
_etf_abort_lock = threading.Lock()


router = APIRouter(prefix="/api", tags=["etf"])


def _count_trading_days_since(date_str: Optional[str]) -> int:
    """计算自指定日期以来的交易日数（简化：自然日减周末）"""
    if not date_str:
        return 999999
    try:
        last_date = datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0]).date()
        today = datetime.now().date()
        days_diff = (today - last_date).days
        weekend_days = (days_diff // 7) * 2
        if last_date.weekday() >= 5:
            weekend_days += 1
        if today.weekday() >= 5:
            weekend_days += 1
        return max(0, days_diff - weekend_days)
    except Exception as e:
        print(f"[ETF模拟盘] 计算交易日失败: {e}")
        return 0


# ---------- 请求模型 ----------

class ETFRotationBacktestRequest(BaseModel):
    etf_codes: list[str]
    initial_capital: float = 100000
    days: int = 252
    rotation_interval: int = 5
    lookback_days: int = 20
    commission_rate: float = 0.0003
    slippage: float = 0.001
    top_k: int = 1
    score_weights: Optional[dict[str, float]] = None
    rebalance_interval: Optional[int] = None
    min_score_threshold: float = 20.0
    use_ai: bool = False
    position_strategy: str = "equal"  # "equal"=等权重, "kelly"=凯利公式


class ETFRotationSuggestionRequest(BaseModel):
    etf_codes: list[str]
    lookback_days: int = 20
    top_k: int = 1
    score_weights: Optional[dict[str, float]] = None
    min_score_threshold: float = 20.0


class ETFRotationAIRequest(BaseModel):
    etf_codes: list[str]
    lookback_days: int = 20
    top_k: int = 1
    min_score_threshold: float = 20.0


class ETFSimAccountCreateRequest(BaseModel):
    account_id: str
    initial_capital: float = 100000


class ETFSimTradeRequest(BaseModel):
    account_id: str
    etf_code: str
    action: str
    shares: Optional[int] = None
    price: Optional[float] = None
    reason: str = ""


class ETFSimAutoTradeRequest(BaseModel):
    account_id: str
    etf_codes: list[str]
    lookback_days: int = 20
    top_k: int = 1
    score_weights: Optional[dict[str, float]] = None
    min_score_threshold: float = 20.0
    rotation_interval: Optional[int] = None
    rebalance_interval: Optional[int] = None
    use_ai: bool = False


class ETFRealTradeRequest(BaseModel):
    etf_code: str
    action: str
    shares: int
    price_type: str = "market"
    price: Optional[float] = None
    reason: str = ""


# ---------- ETF 轮动 ----------

@router.post("/etf-rotation/backtest")
async def etf_rotation_backtest(req: ETFRotationBacktestRequest):
    """ETF轮动策略回测"""
    try:
        from tools.etf_rotation import backtest_etf_rotation
        result = backtest_etf_rotation(
            etf_codes=req.etf_codes,
            initial_capital=req.initial_capital,
            days=req.days,
            rotation_interval=req.rotation_interval,
            lookback_days=req.lookback_days,
            commission_rate=req.commission_rate,
            slippage=req.slippage,
            top_k=req.top_k,
            score_weights=req.score_weights,
            rebalance_interval=req.rebalance_interval,
            min_score_threshold=req.min_score_threshold,
            use_ai=req.use_ai,
            position_strategy=req.position_strategy or "equal"
        )
        if result.get('error'):
            return {"error": result['error'], "result": result}
        chart_base64 = result.pop('chart', None)
        chart_url = chart_base64 if chart_base64 and chart_base64.startswith('data:image') else (f"data:image/png;base64,{chart_base64}" if chart_base64 else None)
        return {"result": result, "chart": chart_url}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _get_etf_abort_event(session_id: str) -> threading.Event:
    with _etf_abort_lock:
        if session_id not in _etf_abort_events:
            _etf_abort_events[session_id] = threading.Event()
        ev = _etf_abort_events[session_id]
        ev.clear()
        return ev


def _cleanup_etf_abort(session_id: str):
    with _etf_abort_lock:
        _etf_abort_events.pop(session_id, None)


@router.get("/etf-rotation/backtest/stream")
async def etf_rotation_backtest_stream(
    session_id: str = Query(..., description="回测会话ID，由前端生成，用于停止时定位"),
    etf_codes: str = Query(..., description="ETF代码，逗号分隔，如 510300,510500,159915"),
    initial_capital: float = Query(100000),
    days: int = Query(252),
    rotation_interval: int = Query(5),
    lookback_days: int = Query(20),
    commission_rate: float = Query(0.0003),
    slippage: float = Query(0.001),
    top_k: int = Query(1),
    rebalance_interval: Optional[int] = Query(None),
    min_score_threshold: float = Query(20.0),
    use_ai: bool = Query(False),
    position_strategy: str = Query("equal", description="仓位策略: equal=等权重, kelly=凯利公式"),
    score_weights: Optional[str] = Query(None, description="得分权重 JSON"),
):
    """ETF轮动回测 - SSE 流式响应，支持多用户同时回测"""
    etf_list = [c.strip() for c in (etf_codes or "").split(",") if c.strip()]
    if not etf_list:
        raise HTTPException(status_code=400, detail="etf_codes 不能为空")
    if not session_id or len(session_id) > 64:
        raise HTTPException(status_code=400, detail="session_id 无效")
    score_weights_dict = None
    if score_weights and score_weights.strip():
        try:
            score_weights_dict = json.loads(score_weights)
        except Exception:
            pass
    abort_ev = _get_etf_abort_event(session_id)
    msg_queue = queue.Queue()

    def progress_callback(event_type: str, data: dict):
        msg_queue.put({"type": event_type, "data": data})

    def abort_check():
        return abort_ev.is_set()

    def run_backtest_thread():
        try:
            from tools.etf_rotation import backtest_etf_rotation
            result = backtest_etf_rotation(
                etf_codes=etf_list,
                initial_capital=initial_capital,
                days=days,
                rotation_interval=rotation_interval,
                lookback_days=lookback_days,
                commission_rate=commission_rate,
                slippage=slippage,
                top_k=top_k,
                score_weights=score_weights_dict,
                rebalance_interval=rebalance_interval,
                min_score_threshold=min_score_threshold,
                use_ai=use_ai,
                position_strategy=position_strategy or "equal",
                progress_callback=progress_callback,
                abort_check=abort_check,
            )
            if result.get("error"):
                msg_queue.put({"type": "error", "data": {"message": result["error"]}})
                return
            chart_base64 = result.pop("chart", None)
            chart_url = chart_base64 if chart_base64 and chart_base64.startswith("data:image") else (
                f"data:image/png;base64,{chart_base64}" if chart_base64 else None
            )
            msg_queue.put({
                "type": "complete",
                "data": {"result": result, "chart": chart_url},
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg_queue.put({"type": "error", "data": {"message": str(e)}})

    async def event_generator():
        thread = threading.Thread(target=run_backtest_thread)
        thread.daemon = True
        thread.start()
        try:
            while True:
                try:
                    try:
                        msg = msg_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not thread.is_alive():
                            break
                        if abort_ev.is_set():
                            # 用户停止：等待回测线程结束并获取其 complete 消息，再断开
                            thread.join(timeout=120)
                            try:
                                while True:
                                    msg = msg_queue.get_nowait()
                                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                                    if msg["type"] in ("complete", "error"):
                                        break
                            except queue.Empty:
                                yield f"data: {json.dumps({'type': 'complete', 'data': {'result': {'aborted': True, 'aborted_message': '回测已停止，以下为局部结果', 'error': '未获取到局部结果'}}}, ensure_ascii=False)}\n\n"
                            break
                        await asyncio.sleep(0.1)
                        continue
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                    if msg["type"] in ("complete", "error"):
                        break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}}, ensure_ascii=False)}\n\n"
                    break
        finally:
            if thread.is_alive():
                thread.join(timeout=3)
            _cleanup_etf_abort(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class BacktestStopRequest(BaseModel):
    session_id: str


@router.post("/etf-rotation/backtest/stop")
async def etf_rotation_backtest_stop(req: BacktestStopRequest):
    """停止指定会话的 ETF 轮动流式回测"""
    session_id = req.session_id or ""
    with _etf_abort_lock:
        ev = _etf_abort_events.get(session_id)
    if ev:
        ev.set()
    return {"ok": True, "message": "已发送停止信号"}


@router.get("/etf-rotation/default-etfs")
async def get_default_etfs():
    """获取默认ETF列表"""
    try:
        from tools.etf_rotation import DEFAULT_ETF_LIST
        return {"result": [{"code": code, "name": name} for code, name in DEFAULT_ETF_LIST.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etf-rotation/suggestion")
async def get_etf_rotation_suggestion(req: ETFRotationSuggestionRequest):
    """获取当前调仓建议"""
    try:
        from tools.etf_rotation import get_current_rotation_suggestion
        result = get_current_rotation_suggestion(
            etf_codes=req.etf_codes,
            lookback_days=req.lookback_days,
            top_k=req.top_k,
            score_weights=req.score_weights,
            min_score_threshold=req.min_score_threshold
        )
        if result.get('error'):
            return {"error": result['error'], "result": result}
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etf-rotation/ai")
async def etf_rotation_ai(req: ETFRotationAIRequest):
    """ETF AI轮动分析（使用LLM工作流）"""
    try:
        etf_codes_str = ",".join(req.etf_codes)
        query = f"""请进行ETF AI轮动分析。\nETF代码: {etf_codes_str}\n回看天数: {req.lookback_days}\n持仓数量: {req.top_k}\n最低得分阈值: {req.min_score_threshold}"""
        state = {"messages": [HumanMessage(content=query)]}
        output = await workflow_graph.ainvoke(state)
        last_message = output["messages"][-1]
        result_content = getattr(last_message, "content", str(last_message))
        structured_data = {}
        tech_match = re.search(r'技术指标推荐:\s*\[([^\]]+)\]', result_content)
        if tech_match:
            structured_data["technical_recommended"] = [e.strip().strip('"\'') for e in tech_match.group(1).split(',')]
        ai_match = re.search(r'AI推荐:\s*\[([^\]]+)\]', result_content)
        if ai_match:
            structured_data["ai_recommended"] = [e.strip().strip('"\'') for e in ai_match.group(1).split(',')]
        final_match = re.search(r'最终推荐:\s*\[([^\]]+)\]', result_content)
        if final_match:
            structured_data["final_recommended"] = [e.strip().strip('"\'') for e in final_match.group(1).split(',')]
        scores_match = re.search(r'所有ETF得分:\s*(\{[^}]+\})', result_content, re.DOTALL)
        if scores_match:
            try:
                structured_data["etf_scores"] = json.loads(scores_match.group(1))
            except Exception:
                pass
        return {"result": {"report": result_content, "structured_data": structured_data}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- ETF 模拟盘 ----------

@router.post("/etf-sim/account/create")
async def create_etf_sim_account(
    req: ETFSimAccountCreateRequest,
    user: dict = Depends(get_current_user),
):
    """创建ETF模拟盘账户"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        data_dir = get_user_etf_sim_dir(user["user_id"])
        os.makedirs(data_dir, exist_ok=True)
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.create_account(req.account_id, req.initial_capital)
        return {"result": account.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-sim/account/{account_id}")
async def get_etf_sim_account(
    account_id: str,
    user: dict = Depends(get_current_user),
):
    """获取ETF模拟盘账户信息"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        from tools.stock_data import get_stock_data
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        current_prices = {}
        for etf_code in account.positions.keys():
            try:
                stock_data = get_stock_data(etf_code, days=60, use_cache=True)
                if stock_data and len(stock_data) > 0:
                    current_prices[etf_code] = stock_data[-1]['close']
            except Exception as e:
                print(f"[ETF模拟盘] 获取账户持仓{etf_code}价格失败: {e}")
        total_equity = account.get_total_equity(current_prices)
        positions_value = account.get_positions_value(current_prices)
        profit_loss = total_equity - account.initial_capital
        profit_loss_pct = (profit_loss / account.initial_capital * 100) if account.initial_capital > 0 else 0
        positions_detail = []
        for etf_code, pos in account.positions.items():
            current_price = current_prices.get(etf_code, pos['entry_price'])
            market_value = pos['shares'] * current_price
            profit_loss_pos = (current_price - pos['entry_price']) * pos['shares']
            profit_loss_pct_pos = (current_price / pos['entry_price'] - 1) * 100 if pos['entry_price'] > 0 else 0
            positions_detail.append({
                'etf_code': etf_code, 'shares': pos['shares'], 'entry_price': pos['entry_price'],
                'current_price': current_price, 'market_value': market_value,
                'profit_loss': profit_loss_pos, 'profit_loss_pct': profit_loss_pct_pos, 'entry_date': pos['entry_date']
            })
        return {
            "result": {
                "account": account.to_dict(),
                "statistics": {
                    "total_equity": round(total_equity, 2), "cash": round(account.cash, 2),
                    "positions_value": round(positions_value, 2), "profit_loss": round(profit_loss, 2),
                    "profit_loss_pct": round(profit_loss_pct, 2), "initial_capital": account.initial_capital
                },
                "positions_detail": positions_detail,
                "current_prices": current_prices
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etf-sim/trade")
async def execute_etf_sim_trade(
    req: ETFSimTradeRequest,
    user: dict = Depends(get_current_user),
):
    """执行ETF模拟盘交易"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        from tools.stock_data import get_stock_data
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(req.account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{req.account_id}不存在")
        if req.price is None:
            try:
                stock_data = get_stock_data(req.etf_code, days=60, use_cache=True)
                if not stock_data or len(stock_data) == 0:
                    raise HTTPException(status_code=400, detail=f"无法获取{req.etf_code}的当前价格")
                current_price = stock_data[-1]['close']
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"获取价格失败: {str(e)}")
        else:
            current_price = req.price
        if req.action == "buy":
            if req.shares is None or req.shares <= 0:
                raise HTTPException(status_code=400, detail="买入时必须指定股数")
            result = account.buy(req.etf_code, req.shares, current_price, reason=req.reason)
        elif req.action == "sell":
            result = account.sell(req.etf_code, req.shares, current_price, reason=req.reason)
        else:
            raise HTTPException(status_code=400, detail=f"无效的交易类型: {req.action}")
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "交易失败"))
        manager.update_account(account)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etf-sim/auto-trade")
async def execute_etf_sim_auto_trade(
    req: ETFSimAutoTradeRequest,
    user: dict = Depends(get_current_user),
):
    """根据调仓建议自动执行ETF模拟盘交易"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        from tools.etf_rotation import get_current_rotation_suggestion
        from tools.stock_data import get_stock_data
        from tools.risk_control import audit_risk, RiskAuditInput, DEFAULT_RISK_CONFIG
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(req.account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{req.account_id}不存在")
        should_rotate = True
        rotation_warnings = []
        if req.rotation_interval is not None and req.rotation_interval > 0:
            days_since_rotation = _count_trading_days_since(account.last_rotation_date)
            if days_since_rotation < req.rotation_interval:
                should_rotate = False
                rotation_warnings.append(
                    f"未到轮动时间：上次轮动于{account.last_rotation_date or '从未'}，已过{days_since_rotation}个交易日，需要{req.rotation_interval}个交易日"
                )
        should_rebalance = True
        rebalance_warnings = []
        if req.rebalance_interval is not None and req.rebalance_interval > 0:
            days_since_rebalance = _count_trading_days_since(account.last_rebalance_date)
            if days_since_rebalance < req.rebalance_interval:
                should_rebalance = False
                rebalance_warnings.append(
                    f"未到再平衡时间：上次再平衡于{account.last_rebalance_date or '从未'}，已过{days_since_rebalance}个交易日，需要{req.rebalance_interval}个交易日"
                )
        if not should_rotate and not should_rebalance:
            return {
                "result": {
                    "suggestion": None, "trades_executed": [], "errors": [],
                    "warnings": rotation_warnings + rebalance_warnings,
                    "account_after": account.to_dict()
                }
            }
        etf_codes_to_use = account.etf_pool if account.etf_pool else req.etf_codes
        if not etf_codes_to_use:
            raise HTTPException(status_code=400, detail="ETF池为空，请先添加ETF到池中")
        suggestion = get_current_rotation_suggestion(
            etf_codes=etf_codes_to_use,
            lookback_days=req.lookback_days,
            top_k=req.top_k,
            score_weights=req.score_weights,
            min_score_threshold=req.min_score_threshold,
            use_ai=req.use_ai
        )
        if suggestion.get("error"):
            error_msg = suggestion["error"]
            if "无法加载ETF数据" in error_msg or "无法获取" in error_msg:
                error_msg = f"{error_msg}。请检查：1) ETF代码是否正确（如：510300, 510500, 159915）；2) 网络连接是否正常；3) 数据源是否可用。"
            raise HTTPException(status_code=400, detail=error_msg)
        current_prices = {}
        price_errors = []
        for etf_code in etf_codes_to_use:
            try:
                stock_data = get_stock_data(etf_code, days=60, use_cache=True)
                if stock_data and len(stock_data) > 0:
                    current_prices[etf_code] = stock_data[-1]['close']
                else:
                    price_errors.append(f"{etf_code}(数据为空)")
            except Exception as e:
                price_errors.append(f"{etf_code}({str(e)})")
        target_etfs = set(suggestion.get("recommended_etfs", []))
        current_etfs = set(account.positions.keys())
        if len(target_etfs) == 0:
            return {
                "result": {
                    "suggestion": suggestion, "trades_executed": [],
                    "errors": ["没有推荐的ETF（所有ETF得分都低于阈值或数据不足）"],
                    "account_after": account.to_dict()
                }
            }
        etfs_to_sell = (current_etfs - target_etfs) if should_rotate else set()
        etfs_to_buy = (target_etfs - current_etfs) if should_rotate else set()
        etfs_to_keep = current_etfs & target_etfs
        trades_executed = []
        errors = []
        warnings = rotation_warnings + rebalance_warnings
        if len(current_prices) == 0:
            error_msg = "无法获取ETF价格数据"
            if price_errors:
                error_msg += f"。失败的ETF: {', '.join(price_errors)}"
            errors.append(error_msg)
            return {"result": {"suggestion": suggestion, "trades_executed": trades_executed, "errors": errors, "account_after": account.to_dict()}}
        missing_prices = [c for c in target_etfs if c not in current_prices]
        if missing_prices:
            errors.append(f"推荐的ETF中以下ETF无法获取价格: {', '.join(missing_prices)}")
        total_equity = account.get_total_equity(current_prices)
        positions_value = account.get_positions_value(current_prices)
        positions_for_risk = {code: {'market_value': account.positions[code]['shares'] * current_prices[code]} for code in account.positions if code in current_prices}
        peak_equity = getattr(account, 'peak_equity', None) or total_equity
        risk_input = RiskAuditInput(
            cash=account.cash, total_value=total_equity, initial_capital=account.initial_capital,
            positions_value=positions_value, positions=positions_for_risk,
            recent_daily_returns=[], peak_value=peak_equity,
        )
        risk_result = audit_risk(risk_input)
        risk_would_block = not risk_result.pass_audit and (risk_result.action in ('stop', 'pause_trading') or (risk_result.action == 'reduce_position' and 'over_weight_code' not in risk_result.details))
        # 模拟盘：风控触发时记录警告供用户人工确认，但不阻断交易
        risk_warning = None
        if risk_would_block:
            warnings.append(f"【风控警告】{risk_result.reason}")
            risk_warning = {"reason": risk_result.reason, "action": risk_result.action, "details": risk_result.details}
        for etf_code in etfs_to_sell:
            if etf_code in current_prices:
                result = account.sell(etf_code, None, current_prices[etf_code], reason=f"轮动卖出（不在top-{req.top_k}）")
                if result.get("success"):
                    trades_executed.append(result["trade"])
                else:
                    errors.append(f"卖出{etf_code}失败: {result.get('error')}")
        total_equity_after_sell = account.get_total_equity(current_prices)
        max_single_pct = DEFAULT_RISK_CONFIG.get('max_single_position_pct', 0.45)
        if should_rebalance and req.rebalance_interval is not None and len(etfs_to_keep) > 0:
            target_weight = min(0.95 / len(target_etfs), max_single_pct) if len(target_etfs) > 0 else 0
            for etf_code in etfs_to_keep:
                if etf_code in current_prices:
                    current_price = current_prices[etf_code]
                    current_shares = account.positions[etf_code]['shares']
                    target_value = total_equity_after_sell * target_weight
                    target_shares = int(target_value / current_price)
                    if target_shares != current_shares:
                        diff_shares = target_shares - current_shares
                        if diff_shares > 0:
                            result = account.buy(etf_code, diff_shares, current_price, reason="再平衡加仓")
                        else:
                            result = account.sell(etf_code, abs(diff_shares), current_price, reason="再平衡减仓")
                        if result.get("success"):
                            trades_executed.append(result["trade"])
                        else:
                            errors.append(f"调仓{etf_code}失败: {result.get('error')}")
        if len(etfs_to_buy) > 0:
            if req.rebalance_interval is not None:
                total_equity_after_adjust = account.get_total_equity(current_prices)
                target_weight = min(0.95 / len(target_etfs), max_single_pct)
                for etf_code in etfs_to_buy:
                    if etf_code not in current_prices:
                        errors.append(f"无法获取{etf_code}的价格数据")
                        continue
                    current_price = current_prices[etf_code]
                    if current_price <= 0:
                        errors.append(f"{etf_code}的价格无效: {current_price}")
                        continue
                    target_value = total_equity_after_adjust * target_weight
                    shares = int(target_value / (current_price * 1.001 * 1.0003))
                    if shares > 0:
                        result = account.buy(etf_code, shares, current_price, reason=f"轮动买入（top-{req.top_k}）")
                        if result.get("success"):
                            trades_executed.append(result["trade"])
                        else:
                            errors.append(f"买入{etf_code}失败: {result.get('error')}")
            else:
                available_cash = account.cash
                total_equity = account.get_total_equity(current_prices)
                num_to_buy = len(etfs_to_buy)
                target_value_per_etf = min(available_cash / num_to_buy, total_equity * max_single_pct) if num_to_buy > 0 else 0
                for etf_code in etfs_to_buy:
                    if etf_code not in current_prices:
                        errors.append(f"无法获取{etf_code}的价格数据")
                        continue
                    current_price = current_prices[etf_code]
                    if current_price <= 0:
                        errors.append(f"{etf_code}的价格无效: {current_price}")
                        continue
                    shares = int(target_value_per_etf / (current_price * 1.001 * 1.0003))
                    if shares > 0:
                        result = account.buy(etf_code, shares, current_price, reason=f"轮动买入（top-{req.top_k}）")
                        if result.get("success"):
                            trades_executed.append(result["trade"])
                        else:
                            errors.append(f"买入{etf_code}失败: {result.get('error')}")
        current_date = datetime.now().date().isoformat()
        if should_rotate and (len(etfs_to_sell) > 0 or len(etfs_to_buy) > 0):
            account.last_rotation_date = current_date
        if should_rebalance and req.rebalance_interval is not None:
            account.last_rebalance_date = current_date
        manager.update_account(account)
        result_data = {"suggestion": suggestion, "trades_executed": trades_executed, "errors": errors, "warnings": warnings, "account_after": account.to_dict()}
        if risk_warning is not None:
            result_data["risk_warning"] = risk_warning
        return {"result": result_data}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-sim/accounts")
async def list_etf_sim_accounts(user: dict = Depends(get_current_user)):
    """列出所有ETF模拟盘账户"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        data_dir = get_user_etf_sim_dir(user["user_id"])
        os.makedirs(data_dir, exist_ok=True)
        manager = ETFSimAccountManager(data_dir=data_dir)
        account_ids = manager.list_accounts()
        return {"result": account_ids}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/etf-sim/account/{account_id}")
async def delete_etf_sim_account(
    account_id: str,
    user: dict = Depends(get_current_user),
):
    """删除ETF模拟盘账户"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        success = manager.delete_account(account_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        return {"result": {"success": True, "message": f"账户{account_id}已删除"}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-sim/account/{account_id}/trades")
async def get_etf_sim_account_trades(
    account_id: str,
    page: int = 1,
    page_size: int = 20,
    user: dict = Depends(get_current_user),
):
    """获取账户的交易记录（分页）"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        all_trades = sorted(account.trades, key=lambda x: x.get('date', ''), reverse=True)
        total = len(all_trades)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        trades_page = all_trades[start_idx:start_idx + page_size]
        return {"result": {"trades": trades_page, "pagination": {"page": page, "page_size": page_size, "total": total, "total_pages": total_pages}}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/etf-sim/account/{account_id}/etf-pool/add")
async def add_etf_to_pool(
    account_id: str,
    etf_code: str,
    user: dict = Depends(get_current_user),
):
    """向账户的ETF池添加ETF"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        etf_code = etf_code.strip().upper()
        if not etf_code:
            raise HTTPException(status_code=400, detail="ETF代码不能为空")
        if etf_code in account.etf_pool:
            raise HTTPException(status_code=400, detail=f"ETF {etf_code} 已在池中")
        account.etf_pool.append(etf_code)
        account.last_updated = datetime.now().isoformat()
        manager.update_account(account)
        return {"result": {"success": True, "message": f"ETF {etf_code} 已添加到池中", "etf_pool": account.etf_pool}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/etf-sim/account/{account_id}/etf-pool/{etf_code}")
async def remove_etf_from_pool(
    account_id: str,
    etf_code: str,
    user: dict = Depends(get_current_user),
    auto_sell: bool = True,
):
    """从账户的ETF池移除ETF，可选自动清仓"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        from tools.stock_data import get_stock_data
        data_dir = get_user_etf_sim_dir(user["user_id"])
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        etf_code = etf_code.strip().upper()
        if etf_code not in account.etf_pool:
            raise HTTPException(status_code=400, detail=f"ETF {etf_code} 不在池中")
        trades_executed = []
        errors = []
        if auto_sell and etf_code in account.positions:
            try:
                stock_data = get_stock_data(etf_code, days=60, use_cache=True)
                if stock_data and len(stock_data) > 0:
                    current_price = stock_data[-1]['close']
                    result = account.sell(etf_code, None, current_price, reason="从ETF池移除，自动清仓")
                    if result.get("success"):
                        trades_executed.append(result["trade"])
                    else:
                        errors.append(f"清仓{etf_code}失败: {result.get('error')}")
                else:
                    errors.append(f"无法获取{etf_code}的价格数据，无法清仓")
            except Exception as e:
                errors.append(f"清仓{etf_code}时出错: {str(e)}")
        account.etf_pool.remove(etf_code)
        account.last_updated = datetime.now().isoformat()
        manager.update_account(account)
        return {"result": {"success": True, "message": f"ETF {etf_code} 已从池中移除" + ("，已清仓" if trades_executed else ""), "trades_executed": trades_executed, "errors": errors, "etf_pool": account.etf_pool}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-sim/account/{account_id}/suggestion")
async def get_etf_sim_account_suggestion(
    account_id: str,
    etf_codes: str,
    user: dict = Depends(get_current_user),
    lookback_days: int = 20,
    top_k: int = 1,
    min_score_threshold: float = 20.0,
    rebalance_interval: Optional[str] = Query(None),
    score_weights: Optional[str] = Query(None),
):
    """获取账户的详细调仓建议（包含具体买卖金额和股数）"""
    try:
        from modules.etf_rotation.etf_sim_account import ETFSimAccountManager
        from tools.etf_rotation import get_current_rotation_suggestion
        from tools.risk_control import DEFAULT_RISK_CONFIG
        from tools.stock_data import get_stock_data
        data_dir = get_user_etf_sim_dir(user["user_id"])
        max_single_pct = DEFAULT_RISK_CONFIG.get('max_single_position_pct', 0.45)
        manager = ETFSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        etf_code_list = [code.strip() for code in etf_codes.split(',') if code.strip()]
        if not etf_code_list:
            raise HTTPException(status_code=400, detail="ETF代码列表不能为空")
        rebalance_interval_int = None
        if rebalance_interval and rebalance_interval.strip():
            try:
                rebalance_interval_int = int(rebalance_interval)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的再平衡间隔: {rebalance_interval}")
        score_weights_dict = None
        if score_weights:
            try:
                score_weights_dict = json.loads(score_weights)
            except Exception:
                pass
        suggestion = get_current_rotation_suggestion(
            etf_codes=etf_code_list,
            lookback_days=lookback_days,
            top_k=top_k,
            score_weights=score_weights_dict,
            min_score_threshold=min_score_threshold
        )
        if suggestion.get("error"):
            raise HTTPException(status_code=400, detail=suggestion["error"])
        current_prices = {}
        for etf_code in etf_code_list:
            try:
                stock_data = get_stock_data(etf_code, days=60, use_cache=True)
                if stock_data and len(stock_data) > 0:
                    current_prices[etf_code] = stock_data[-1]['close']
            except Exception as e:
                print(f"[ETF模拟盘] 获取{etf_code}价格失败: {e}")
        target_etfs = set(suggestion.get("recommended_etfs", []))
        current_etfs = set(account.positions.keys())
        etfs_to_sell = current_etfs - target_etfs
        etfs_to_buy = target_etfs - current_etfs
        etfs_to_keep = current_etfs & target_etfs
        total_equity = account.get_total_equity(current_prices)
        trading_plan = {"to_sell": [], "to_buy": [], "to_adjust": []}
        for etf_code in etfs_to_sell:
            if etf_code in account.positions and etf_code in current_prices:
                pos = account.positions[etf_code]
                current_shares = pos['shares']
                current_price = current_prices[etf_code]
                sell_price = current_price * (1 - 0.001)
                estimated_revenue = current_shares * sell_price * (1 - 0.0003)
                trading_plan["to_sell"].append({
                    "etf_code": etf_code, "current_shares": current_shares, "sell_shares": current_shares,
                    "current_price": round(current_price, 2), "estimated_revenue": round(estimated_revenue, 2)
                })
        if rebalance_interval_int is not None and len(etfs_to_keep) > 0:
            target_weight = min(0.95 / len(target_etfs), max_single_pct) if len(target_etfs) > 0 else 0
            for etf_code in etfs_to_keep:
                if etf_code in account.positions and etf_code in current_prices:
                    pos = account.positions[etf_code]
                    current_shares = pos['shares']
                    current_price = current_prices[etf_code]
                    target_value = total_equity * target_weight
                    target_shares = int(target_value / current_price)
                    if target_shares != current_shares:
                        diff_shares = target_shares - current_shares
                        adjust_value = abs(diff_shares) * current_price * (1 + 0.001) * (1 + 0.0003) if diff_shares > 0 else abs(diff_shares) * current_price * (1 - 0.001) * (1 - 0.0003)
                        trading_plan["to_adjust"].append({
                            "etf_code": etf_code, "current_shares": current_shares, "target_shares": target_shares,
                            "adjust_shares": diff_shares, "adjust_value": round(adjust_value, 2),
                            "current_price": round(current_price, 2), "action": "加仓" if diff_shares > 0 else "减仓"
                        })
        if len(etfs_to_buy) > 0:
            if rebalance_interval_int is not None:
                target_weight = min(0.95 / len(target_etfs), max_single_pct)
                for etf_code in etfs_to_buy:
                    if etf_code in current_prices:
                        current_price = current_prices[etf_code]
                        target_value = total_equity * target_weight
                        target_shares = int(target_value / (current_price * 1.001 * 1.0003))
                        estimated_cost = target_shares * current_price * (1 + 0.001) * (1 + 0.0003)
                        trading_plan["to_buy"].append({
                            "etf_code": etf_code, "target_value": round(target_value, 2), "target_shares": target_shares,
                            "current_price": round(current_price, 2), "estimated_cost": round(estimated_cost, 2),
                            "target_weight": round(target_weight * 100, 2)
                        })
            else:
                available_cash = account.cash
                num_to_buy = len(etfs_to_buy)
                target_weight = 1.0 / num_to_buy if num_to_buy > 0 else 0
                for etf_code in etfs_to_buy:
                    if etf_code in current_prices:
                        current_price = current_prices[etf_code]
                        target_value = available_cash * target_weight
                        target_shares = int(target_value / (current_price * 1.001 * 1.0003))
                        estimated_cost = target_shares * current_price * (1 + 0.001) * (1 + 0.0003)
                        trading_plan["to_buy"].append({
                            "etf_code": etf_code, "target_value": round(target_value, 2), "target_shares": target_shares,
                            "current_price": round(current_price, 2), "estimated_cost": round(estimated_cost, 2),
                            "target_weight": round(target_weight * 100, 2)
                        })
        return {
            "result": {
                "suggestion": suggestion,
                "trading_plan": trading_plan,
                "account_info": {
                    "total_equity": round(total_equity, 2),
                    "cash": round(account.cash, 2),
                    "positions_value": round(account.get_positions_value(current_prices), 2)
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- ETF 实盘 ----------

@router.post("/etf-real/trade")
async def execute_etf_real_trade(req: ETFRealTradeRequest):
    """执行ETF实盘交易（调用QMT接口）"""
    try:
        from modules.qmt import get_qmt_tools
        from tools.stock_data import get_stock_data
        qmt = get_qmt_tools()
        if not qmt.is_connected:
            if not qmt.connect():
                raise HTTPException(status_code=500, detail="QMT连接失败，请检查QMT客户端是否已启动")
        if req.price_type == "market" and req.price is None:
            try:
                stock_data = get_stock_data(req.etf_code, days=60, use_cache=True)
                if not stock_data or len(stock_data) == 0:
                    raise HTTPException(status_code=400, detail=f"无法获取{req.etf_code}的当前价格")
                current_price = stock_data[-1]['close']
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"获取价格失败: {str(e)}")
        else:
            current_price = req.price or 0
        order_type = "buy" if req.action == "buy" else "sell"
        order_result = qmt.place_order(stock_code=req.etf_code, order_type=order_type, price=current_price, volume=req.shares)
        if not order_result.get("success"):
            raise HTTPException(status_code=400, detail=order_result.get("error", "下单失败"))
        return {"result": {"order": order_result, "message": f"实盘{req.action}订单已提交: {req.etf_code} {req.shares}股 @ {current_price:.2f}元"}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-real/positions")
async def get_etf_real_positions():
    """获取实盘持仓（调用QMT接口）"""
    try:
        from modules.qmt import get_qmt_tools
        qmt = get_qmt_tools()
        if not qmt.is_connected:
            if not qmt.connect():
                raise HTTPException(status_code=500, detail="QMT连接失败，请检查QMT客户端是否已启动")
        positions = qmt.get_positions()
        return {"result": positions}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-real/account")
async def get_etf_real_account():
    """获取实盘账户信息（调用QMT接口）"""
    try:
        from modules.qmt import get_qmt_tools
        qmt = get_qmt_tools()
        if not qmt.is_connected:
            if not qmt.connect():
                raise HTTPException(status_code=500, detail="QMT连接失败，请检查QMT客户端是否已启动")
        account_info = qmt.get_account_info()
        return {"result": account_info}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
