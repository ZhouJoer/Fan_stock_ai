"""选股池与选股池模拟仓路由"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
import queue
import re
import threading
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import (
    get_current_user,
    get_user_pool_sim_dir,
    get_user_pools_dir,
)

# 按 session 隔离的停止信号，支持多用户同时回测
_pool_abort_events: dict[str, threading.Event] = {}
_pool_abort_lock = threading.Lock()

# 深度因子搜索：按 session 隔离的进度队列与停止信号
_deep_search_queues: dict[str, queue.Queue] = {}
_deep_search_abort_events: dict[str, threading.Event] = {}
_deep_search_lock = threading.Lock()

# 因子回测流式进度
_factor_backtest_queues: dict[str, queue.Queue] = {}
_factor_backtest_lock = threading.Lock()


def _get_pool_abort_event(session_id: str) -> threading.Event:
    with _pool_abort_lock:
        if session_id not in _pool_abort_events:
            _pool_abort_events[session_id] = threading.Event()
        ev = _pool_abort_events[session_id]
        ev.clear()
        return ev


def _cleanup_pool_abort(session_id: str):
    with _pool_abort_lock:
        _pool_abort_events.pop(session_id, None)


def _build_go_live_recommendation(evaluation: dict | None) -> dict:
    e = evaluation or {}
    reasons: list[str] = []
    level = "pass"
    ic_ir = float(e.get("ic_ir", 0.0) or 0.0)
    ic_pos = float(e.get("ic_positive_ratio", 0.0) or 0.0)
    spread = float(e.get("spread_mean", 0.0) or 0.0)
    turnover = float(e.get("selection_turnover_mean", 0.0) or 0.0)
    fail_streak = int(e.get("max_consecutive_fail_days", 0) or 0)
    records = int(e.get("records", 0) or 0)

    if records < 12:
        level = "warn"
        reasons.append("评估样本天数偏少，稳定性结论可信度有限")
    if ic_ir < 0.2:
        level = "warn"
        reasons.append("ICIR 偏低，排序稳定性不足")
    if ic_pos < 0.55:
        level = "warn"
        reasons.append("IC 正值占比不足 55%")
    if spread <= 0:
        level = "fail"
        reasons.append("分层收益差非正，因子区分度不足")
    if fail_streak >= 8:
        level = "fail"
        reasons.append("连续失效天数过长")
    if turnover > 0.75:
        if level == "pass":
            level = "warn"
        reasons.append("换手偏高，实盘摩擦成本可能放大")

    if not reasons:
        reasons.append("稳定性与解释性指标满足上线参考阈值")
    return {
        "level": level,
        "reasons": reasons,
        "thresholds": {
            "ic_ir_min": 0.2,
            "ic_positive_ratio_min": 0.55,
            "spread_mean_min": 0.0,
            "max_consecutive_fail_days_max": 7,
            "selection_turnover_mean_warn": 0.75,
        },
        "actual": {
            "records": records,
            "ic_ir": round(ic_ir, 6),
            "ic_positive_ratio": round(ic_pos, 6),
            "spread_mean": round(spread, 6),
            "max_consecutive_fail_days": fail_streak,
            "selection_turnover_mean": round(turnover, 6),
        },
    }


router = APIRouter(prefix="/api", tags=["pool"])


class StockPoolSignalsRequest(BaseModel):
    stocks: list[str]
    risk_preference: str = "balanced"


class StockPoolAllocationRequest(BaseModel):
    stocks: list[str]
    total_capital: float = 1000000
    allocation_method: str = "signal_strength"
    risk_preference: str = "balanced"


class StockPoolBacktestRequest(BaseModel):
    stocks: list[str] = []
    initial_capital: float = 1000000
    days: int = 252
    strategy_type: str = "adaptive"
    risk_preference: str = "balanced"
    allocation_method: str = "signal_strength"
    rebalance_interval: int = 5
    use_llm_signals: bool = False
    llm_sample_rate: int = 5
    universe_source: str = "manual"
    universe_index: str = ""
    industry_list: list[str] | None = None
    leaders_per_industry: int = 1
    selection_mode: str = "none"
    selection_top_n: int = 10
    selection_interval: int = 0
    score_weights: dict[str, float] | None = None
    factor_set: str = "hybrid"
    weight_source: str = "manual"  # manual | learned
    model_name: str = ""
    no_lookahead: bool = False
    start_date: str = ""


class StockPoolReportRequest(BaseModel):
    stocks: list[str]
    total_capital: float = 1000000
    risk_preference: str = "balanced"


def _resolve_backtest_learned_weights(
    user: dict,
    weight_source: str,
    model_name: str,
) -> tuple[dict | None, str]:
    if (weight_source or "manual") != "learned":
        return None, ""
    from modules.factor_mining import ensure_model_dir, load_factor_model

    models_dir = ensure_model_dir(os.path.join(get_user_pools_dir(user["user_id"]), "factor_models"))
    learned_model, resolved_model_name = load_factor_model(models_dir, model_name or "")
    if learned_model is None:
        if (model_name or "").strip():
            raise HTTPException(status_code=400, detail=f"学习模型不存在：{model_name}")
        raise HTTPException(status_code=400, detail="未找到可用学习模型，请先训练权重")
    return (learned_model or {}).get("learned_weights", {}), (resolved_model_name or "")


class StockPoolSaveRequest(BaseModel):
    name: str
    stocks: list[str]
    initial_capital: float = 1000000
    strategy_type: str = "adaptive"
    risk_preference: str = "balanced"
    allocation_method: str = "signal_strength"
    selection_mode: str = "none"
    selection_top_n: int = 10
    selection_interval: int = 0
    score_weights: dict[str, float] | None = None
    factor_set: str = "hybrid"
    strategy_meta: dict | None = None
    factor_profile: dict | None = None


class FactorDeepSearchRequest(BaseModel):
    stocks: list[str] = []
    universe_source: str = "manual"
    universe_index: str = ""
    industry_list: list[str] | None = None
    leaders_per_industry: int = 1
    max_stocks: int = 60
    days: int = 252
    factor_sets: str = "style,trading,hybrid"
    horizons: str = "1,3,5"
    feature_counts: str = "3,4,5"
    val_ratios: str = "0.15,0.2,0.25"
    epochs: int = 220
    lr: float = 0.05
    l2_lambda: float = 0.0001
    exclude_kechuang: bool = False
    exclude_small_cap: bool = False
    small_cap_max_billion: float = 30.0
    factor_mode: str = "multi"
    benchmark_code: str = "510300"
    label_horizon: int = 5
    rebalance_freq: int = 1
    top_n: int = 10
    max_combos: int = 15
    orchestrate_tasks: bool = False
    orchestrate_user_preference: str = ""
    cap_scope: str = "none"
    small_cap_threshold_billion: float = 30.0
    n_trials: int | None = None


class StockPoolSimAccountCreateRequest(BaseModel):
    account_id: str
    initial_capital: float = 1000000
    stock_pool: list[str] = []


class StockPoolSimRebalanceRequest(BaseModel):
    account_id: str
    risk_preference: str = "balanced"
    use_llm: bool = False


class PoolBacktestStopRequest(BaseModel):
    session_id: str


class FactorBacktestRequest(BaseModel):
    """因子策略仅回测：复用股票池参数 + 因子组合与权重。robustness_check 为真时额外跑多组 TopN/调仓周期并返回稳健性结果。"""
    stocks: list[str] = []
    universe_source: str = "manual"
    universe_index: str = ""
    industry_list: list[str] | None = None
    leaders_per_industry: int = 1
    max_stocks: int = 60
    days: int = 252
    label_horizon: int = 5
    rebalance_freq: int = 1
    benchmark_code: str = "510300"
    top_n: int = 10
    factor_combo: list[str] = []
    weights: dict[str, float] = {}
    exclude_kechuang: bool = False
    exclude_small_cap: bool = False
    small_cap_max_billion: float = 30.0
    cap_scope: str = "none"
    position_weight_method: str = "equal"
    robustness_check: bool = False


@router.post("/pool/signals")
async def pool_signals(req: StockPoolSignalsRequest):
    """获取选股池中所有股票的交易信号"""
    try:
        from tools.stock_pool import get_pool_signals
        signals = get_pool_signals(stocks=req.stocks, risk_preference=req.risk_preference)
        return {"result": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/allocation")
async def pool_allocation(req: StockPoolAllocationRequest):
    """获取选股池的仓位配置建议"""
    try:
        from tools.stock_pool import get_pool_allocation
        allocation = get_pool_allocation(
            stocks=req.stocks,
            total_capital=req.total_capital,
            allocation_method=req.allocation_method,
            risk_preference=req.risk_preference
        )
        return {"result": allocation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/backtest")
async def pool_backtest(
    req: StockPoolBacktestRequest,
    user: dict = Depends(get_current_user),
):
    """选股池组合回测"""
    try:
        from tools.stock_pool import backtest_stock_pool
        learned_weights, resolved_model_name = _resolve_backtest_learned_weights(
            user,
            getattr(req, "weight_source", "manual"),
            getattr(req, "model_name", ""),
        )
        if getattr(req, "no_lookahead", False) and not (getattr(req, "start_date", "") or "").strip():
            raise HTTPException(status_code=400, detail="无前视模式下必须提供 start_date")
        result = backtest_stock_pool(
            stocks=req.stocks or [],
            initial_capital=req.initial_capital,
            days=req.days,
            strategy_type=req.strategy_type,
            risk_preference=req.risk_preference,
            allocation_method=req.allocation_method,
            rebalance_interval=req.rebalance_interval,
            use_llm_signals=req.use_llm_signals,
            llm_sample_rate=req.llm_sample_rate,
            universe_source=getattr(req, "universe_source", "manual"),
            universe_index=getattr(req, "universe_index", "") or "",
            industry_list=getattr(req, "industry_list", None),
            leaders_per_industry=getattr(req, "leaders_per_industry", 1),
            selection_mode=getattr(req, "selection_mode", "none"),
            selection_top_n=getattr(req, "selection_top_n", 10),
            selection_interval=getattr(req, "selection_interval", 0),
            score_weights=getattr(req, "score_weights", None),
            factor_set=getattr(req, "factor_set", "hybrid"),
            weight_source=getattr(req, "weight_source", "manual"),
            model_name=resolved_model_name,
            learned_weights=learned_weights,
            no_lookahead=getattr(req, "no_lookahead", False),
            start_date=(getattr(req, "start_date", "") or "").strip(),
        )
        chart_base64 = result.pop('chart', None)
        return {
            "result": result,
            "chart": f"data:image/png;base64,{chart_base64}" if chart_base64 else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/backtest/stream")
async def pool_backtest_stream(
    session_id: str = Query(..., description="回测会话ID，由前端生成"),
    stocks: str = Query("", description="股票代码，逗号分隔（universe_source=manual 时必填）"),
    initial_capital: float = Query(1000000),
    days: int = Query(252),
    strategy_type: str = Query("adaptive"),
    risk_preference: str = Query("balanced"),
    allocation_method: str = Query("signal_strength"),
    rebalance_interval: int = Query(5),
    use_llm_signals: bool = Query(False),
    llm_sample_rate: int = Query(5, description="LLM采样频率：1=每天,3=每3天,5=每5天,10=每10天"),
    high_win_rate_mode: bool = Query(False, description="高胜率模式"),
    universe_source: str = Query("manual", description="股票来源：manual | index | industry"),
    universe_index: str = Query("", description="指数代码，如 000300"),
    industry_list: str = Query("", description="行业名称，逗号分隔，如 银行,电力,白酒"),
    leaders_per_industry: int = Query(1, description="每行业龙头数量"),
    selection_mode: str = Query("none", description="选股模式：none | factor_top_n"),
    selection_top_n: int = Query(10, description="因子选股TopN"),
    selection_interval: int = Query(0, description="因子重选间隔(交易日),0=仅初选"),
    score_weights: str = Query("", description="因子权重JSON"),
    factor_set: str = Query("hybrid", description="因子集合：style | trading | hybrid（风格与估值|情绪与交易）"),
    weight_source: str = Query("manual", description="权重来源：manual | learned"),
    model_name: str = Query("", description="学习模型名（weight_source=learned 时可选）"),
    no_lookahead: bool = Query(False, description="是否启用无前视模式"),
    start_date: str = Query("", description="无前视模式的回测起始日期，YYYY-MM-DD"),
    user: dict = Depends(get_current_user),
):
    """选股池组合回测 - SSE流式响应，支持多用户同时回测"""
    if not session_id or len(session_id) > 64:
        raise HTTPException(status_code=400, detail="session_id 无效")
    stock_list = [s.strip() for s in (stocks or "").split(",") if s.strip()]
    industry_list_parsed = [s.strip() for s in (industry_list or "").split(",") if s.strip()] or None
    score_weights_parsed = None
    if score_weights:
        try:
            score_weights_parsed = json.loads(score_weights)
        except Exception:
            score_weights_parsed = None
    learned_weights, resolved_model_name = _resolve_backtest_learned_weights(
        user,
        weight_source,
        model_name,
    )
    if no_lookahead and not (start_date or "").strip():
        raise HTTPException(status_code=400, detail="无前视模式下必须提供 start_date")
    abort_ev = _get_pool_abort_event(session_id)
    msg_queue = queue.Queue()

    def decision_callback(event_type: str, data: dict):
        msg_queue.put({'type': event_type, 'data': data})

    def abort_check():
        return abort_ev.is_set()

    def run_backtest_thread():
        try:
            from tools.stock_pool import backtest_stock_pool
            result = backtest_stock_pool(
                stocks=stock_list,
                initial_capital=initial_capital,
                days=days,
                strategy_type=strategy_type,
                risk_preference=risk_preference,
                allocation_method=allocation_method,
                rebalance_interval=rebalance_interval,
                use_llm_signals=use_llm_signals,
                llm_sample_rate=llm_sample_rate,
                high_win_rate_mode=high_win_rate_mode,
                decision_callback=decision_callback,
                abort_check=abort_check,
                universe_source=universe_source,
                universe_index=universe_index or "",
                industry_list=industry_list_parsed,
                leaders_per_industry=leaders_per_industry,
                selection_mode=selection_mode,
                selection_top_n=selection_top_n,
                selection_interval=selection_interval,
                score_weights=score_weights_parsed,
                factor_set=factor_set,
                weight_source=weight_source,
                model_name=resolved_model_name,
                learned_weights=learned_weights,
                no_lookahead=no_lookahead,
                start_date=(start_date or "").strip(),
            )
            chart_base64 = result.pop('chart', None)
            msg_queue.put({
                'type': 'complete',
                'data': {
                    'result': result,
                    'chart': f"data:image/png;base64,{chart_base64}" if chart_base64 else None
                }
            })
        except Exception as e:
            msg_queue.put({'type': 'error', 'data': {'message': str(e)}})

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
                            # 用户停止：等待回测线程结束并获取 complete 消息后再断开
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
                    if msg['type'] in ['complete', 'error']:
                        break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}}, ensure_ascii=False)}\n\n"
                    break
        finally:
            if thread.is_alive():
                thread.join(timeout=3)
            _cleanup_pool_abort(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/pool/backtest/stop")
async def pool_backtest_stop(req: PoolBacktestStopRequest):
    """停止指定会话的选股池流式回测"""
    session_id = req.session_id or ""
    with _pool_abort_lock:
        ev = _pool_abort_events.get(session_id)
    if ev:
        ev.set()
    return {"ok": True, "message": "已发送停止信号"}


def _n_trials_from_req(req: FactorDeepSearchRequest) -> int | None:
    """迭代次数：1–150，未传则返回 None（由下游用默认）。"""
    raw = getattr(req, "n_trials", None)
    if raw is None:
        return None
    n = max(1, min(150, int(raw or 0)))
    return n if n > 0 else None


def _max_combos_from_req(req: FactorDeepSearchRequest) -> int:
    """尝试组数：与迭代次数一致（n_trials 有值时用 n_trials 且上限 150），否则用 max_combos。"""
    n_trials = _n_trials_from_req(req)
    if n_trials is not None:
        return n_trials
    return max(1, min(150, int(getattr(req, "max_combos", 15) or 15)))


def _build_deep_search_params(req: FactorDeepSearchRequest) -> dict:
    universe_source = (req.universe_source or "manual").strip() or "manual"
    universe_index = (req.universe_index or "").strip()
    if universe_source == "index" and not universe_index:
        universe_index = "000300"
    return {
        "universe_source": universe_source,
        "universe_index": universe_index,
        "industry_list": ",".join(req.industry_list) if isinstance(req.industry_list, list) else (req.industry_list or ""),
        "leaders_per_industry": max(1, int(req.leaders_per_industry or 1)),
        "stocks": ",".join(req.stocks) if isinstance(req.stocks, list) else (req.stocks or ""),
        "max_stocks": int(req.max_stocks or 60),
        "days": int(req.days or 252),
        "factor_sets": (req.factor_sets or "style,trading,hybrid").strip(),
        "horizons": (req.horizons or "1,3,5").strip(),
        "feature_counts": (req.feature_counts or "3,4,5").strip(),
        "val_ratios": (req.val_ratios or "0.2").strip(),
        "epochs": int(req.epochs or 220),
        "lr": float(req.lr or 0.05),
        "l2_lambda": float(req.l2_lambda or 0.0001),
        "exclude_kechuang": bool(req.exclude_kechuang),
        "exclude_small_cap": bool(req.exclude_small_cap),
        "small_cap_max_billion": float(req.small_cap_max_billion or 30),
        "factor_mode": (req.factor_mode or "multi").strip().lower() or "multi",
        "benchmark_code": (req.benchmark_code or "510300").strip() or "510300",
        "label_horizon": max(1, int(req.label_horizon or 5)),
        "rebalance_freq": max(1, int(getattr(req, "rebalance_freq", 1) or 1)),
        "top_n": max(1, min(int(getattr(req, "top_n", 10) or 10), 50)),
        "max_combos": _max_combos_from_req(req),
        "orchestrate_tasks": bool(getattr(req, "orchestrate_tasks", False)),
        "orchestrate_user_preference": (getattr(req, "orchestrate_user_preference", None) or "").strip() or "",
        "cap_scope": (req.cap_scope or "none").strip() or "none",
        "small_cap_threshold_billion": float(getattr(req, "small_cap_threshold_billion", 30)),
        "n_trials": _n_trials_from_req(req),
    }


@router.post("/pool/factor-deep-search")
async def pool_factor_deep_search(
    req: FactorDeepSearchRequest,
    user: dict = Depends(get_current_user),
):
    """深度因子组合搜索（同步，无进度）：建议使用 start+stream+stop 以支持进度与停止"""
    def _run():
        from tools.deep_factor_search import run_search_from_params
        return run_search_from_params(_build_deep_search_params(req))

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/factor-deep-search/start")
async def pool_factor_deep_search_start(
    req: FactorDeepSearchRequest,
    user: dict = Depends(get_current_user),
):
    """启动深度因子组合搜索（后台运行），返回 session_id，前端用此 id 连接 stream 并可选调用 stop"""
    import uuid
    session_id = str(uuid.uuid4())
    msg_queue = queue.Queue()
    abort_ev = threading.Event()

    with _deep_search_lock:
        _deep_search_queues[session_id] = msg_queue
        _deep_search_abort_events[session_id] = abort_ev

    def run_search_thread():
        try:
            from tools.deep_factor_search import run_search_from_params
            params = _build_deep_search_params(req)
            def event_sink(role: str, phase: str, content: str):
                msg_queue.put({"type": "agent_log", "role": role, "phase": phase, "content": content})
            result = run_search_from_params(
                params,
                progress_callback=lambda cur, total, msg: msg_queue.put({"type": "progress", "current": cur, "total": total, "message": msg}),
                abort_check=lambda: abort_ev.is_set(),
                event_sink=event_sink,
            )
            msg_queue.put({"type": "complete", "result": result})
        except Exception as e:
            msg_queue.put({"type": "error", "message": str(e)})
        finally:
            with _deep_search_lock:
                _deep_search_queues.pop(session_id, None)
                _deep_search_abort_events.pop(session_id, None)

    threading.Thread(target=run_search_thread, daemon=True).start()
    return {"session_id": session_id}


@router.get("/pool/factor-deep-search/stream")
async def pool_factor_deep_search_stream(
    session_id: str = Query(..., description="由 /factor-deep-search/start 返回的 session_id"),
    user: dict = Depends(get_current_user),
):
    """SSE 流：推送深度搜索进度与结果（无超时），支持配合 stop 中止"""
    with _deep_search_lock:
        q = _deep_search_queues.get(session_id)
    if not q:
        raise HTTPException(status_code=404, detail="session 不存在或已结束")

    def _serialize_msg(m):
        """确保 result 等含 numpy/非 JSON 类型时可序列化，避免流中断导致前端收不到 reviewer。"""
        def _default(o):
            if hasattr(o, "item"):  # numpy scalar
                return o.item()
            if hasattr(o, "tolist"):  # numpy array
                return o.tolist()
            return str(o)
        return json.dumps(m, ensure_ascii=False, default=_default)

    async def event_gen():
        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=10))
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                continue
            yield f"data: {_serialize_msg(msg)}\n\n"
            if msg.get("type") in ("complete", "stopped", "error"):
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/pool/factor-deep-search/stop")
async def pool_factor_deep_search_stop(
    body: dict,
    user: dict = Depends(get_current_user),
):
    """请求停止正在进行的深度搜索（传入 session_id）"""
    session_id = (body or {}).get("session_id") or ""
    with _deep_search_lock:
        ev = _deep_search_abort_events.get(session_id)
    if ev:
        ev.set()
        return {"ok": True, "message": "已发送停止信号"}
    return {"ok": False, "message": "session 不存在或已结束"}


def _build_factor_backtest_params(req: FactorBacktestRequest) -> dict:
    universe_source = (req.universe_source or "manual").strip() or "manual"
    universe_index = (req.universe_index or "").strip()
    if universe_source == "index" and not universe_index:
        universe_index = "000300"
    return {
        "universe_source": universe_source,
        "universe_index": universe_index,
        "industry_list": ",".join(req.industry_list) if isinstance(req.industry_list, list) else (req.industry_list or ""),
        "leaders_per_industry": max(1, int(req.leaders_per_industry or 1)),
        "stocks": ",".join(req.stocks) if isinstance(req.stocks, list) else (req.stocks or ""),
        "max_stocks": int(req.max_stocks or 60),
        "days": int(req.days or 252),
        "label_horizon": max(1, int(req.label_horizon or 5)),
        "rebalance_freq": max(1, int(req.rebalance_freq or 1)),
        "benchmark_code": (req.benchmark_code or "510300").strip() or "510300",
        "top_n": max(1, min(int(req.top_n or 10), 50)),
        "factor_combo": req.factor_combo or [],
        "weights": req.weights or {},
        "exclude_kechuang": bool(req.exclude_kechuang),
        "exclude_small_cap": bool(req.exclude_small_cap),
        "small_cap_max_billion": float(req.small_cap_max_billion or 30),
        "cap_scope": (req.cap_scope or "none").strip() or "none",
        "position_weight_method": (req.position_weight_method or "equal").strip().lower() or "equal",
        "robustness_check": bool(req.robustness_check),
    }


@router.post("/pool/factor-backtest")
async def pool_factor_backtest(
    req: FactorBacktestRequest,
    user: dict = Depends(get_current_user),
):
    """因子策略仅回测（同步）：根据给定因子组合与权重运行回测。建议用 factor-backtest/start + stream 获取进度。"""
    try:
        from tools.deep_factor_search import run_factor_backtest_only
        result = run_factor_backtest_only(_build_factor_backtest_params(req))
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/factor-backtest/start")
async def pool_factor_backtest_start(
    req: FactorBacktestRequest,
    user: dict = Depends(get_current_user),
):
    """启动因子回测（后台），返回 session_id，前端用 stream 拉取进度与结果。"""
    import uuid
    session_id = str(uuid.uuid4())
    msg_queue = queue.Queue()
    with _factor_backtest_lock:
        _factor_backtest_queues[session_id] = msg_queue

    def run_backtest_thread():
        try:
            from tools.deep_factor_search import run_factor_backtest_only
            params = _build_factor_backtest_params(req)

            def progress_cb(phase: str, pct: float, message: str):
                msg_queue.put({"type": "progress", "phase": phase, "pct": pct, "message": message})

            result = run_factor_backtest_only(params, progress_callback=progress_cb)
            msg_queue.put({"type": "complete", "result": result})
        except Exception as e:
            msg_queue.put({"type": "error", "message": str(e)})
        finally:
            with _factor_backtest_lock:
                _factor_backtest_queues.pop(session_id, None)

    threading.Thread(target=run_backtest_thread, daemon=True).start()
    return {"session_id": session_id}


@router.get("/pool/factor-backtest/stream")
async def pool_factor_backtest_stream(
    session_id: str = Query(..., description="由 factor-backtest/start 返回的 session_id"),
    user: dict = Depends(get_current_user),
):
    """SSE 流：回测进度与结果。progress 消息含 phase, pct, message。"""
    with _factor_backtest_lock:
        q = _factor_backtest_queues.get(session_id)
    if not q:
        raise HTTPException(status_code=404, detail="session 不存在或已结束")

    async def event_gen():
        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=10))
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                continue
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


BACKTEST_SUMMARIES_FILENAME = "backtest_summaries.json"


def _get_backtest_summaries_path(user_id: str) -> str:
    base = get_user_pools_dir(user_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, BACKTEST_SUMMARIES_FILENAME)


def _load_backtest_summaries(user_id: str) -> list:
    path = _get_backtest_summaries_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_backtest_summaries(user_id: str, items: list) -> None:
    path = _get_backtest_summaries_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


class BacktestSummarySaveRequest(BaseModel):
    """保存当前回测摘要（因子、收益等）"""
    title: str = ""
    factor_combo: list[str] = []
    weights: dict[str, float] = {}
    backtest_stats: dict | None = None
    alpha: float | None = None
    beta: float | None = None
    annualized_alpha: float | None = None
    r_squared: float | None = None
    alpha_beta: dict | None = None
    position_weight_method: str = "equal"
    label_horizon: int = 5
    rebalance_freq: int = 1
    top_n: int = 10
    days: int = 252
    universe_source: str = ""
    universe_index: str = ""
    benchmark_code: str = ""
    strategy_logic: str = ""
    rotation_logic: str = ""
    rebalance_details_count: int = 0
    max_drawdown: float | None = None
    total_return: float | None = None
    sharpe_annual: float | None = None


@router.post("/pool/backtest-summaries")
async def pool_backtest_summary_save(
    req: BacktestSummarySaveRequest,
    user: dict = Depends(get_current_user),
):
    """保存回测摘要（因子组合、收益等）"""
    items = _load_backtest_summaries(user["user_id"])
    entry = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "title": (req.title or "").strip() or f"回测摘要 {len(items) + 1}",
        "factor_combo": req.factor_combo or [],
        "weights": req.weights or {},
        "backtest_stats": req.backtest_stats or {},
        "alpha": req.alpha,
        "beta": req.beta,
        "annualized_alpha": req.annualized_alpha,
        "r_squared": req.r_squared,
        "position_weight_method": req.position_weight_method or "equal",
        "label_horizon": req.label_horizon,
        "rebalance_freq": req.rebalance_freq,
        "top_n": req.top_n,
        "days": req.days,
        "universe_source": req.universe_source or "",
        "universe_index": req.universe_index or "",
        "benchmark_code": req.benchmark_code or "",
        "strategy_logic": (req.strategy_logic or "").strip(),
        "rotation_logic": (req.rotation_logic or "").strip(),
        "rebalance_details_count": req.rebalance_details_count or 0,
        "max_drawdown": req.max_drawdown,
        "total_return": req.total_return,
        "sharpe_annual": req.sharpe_annual,
        "alpha_beta": req.alpha_beta or ({"alpha": req.alpha, "beta": req.beta, "annualized_alpha": req.annualized_alpha, "r_squared": req.r_squared} if (req.alpha is not None or req.beta is not None) else None),
    }
    items.append(entry)
    _save_backtest_summaries(user["user_id"], items)
    return {"result": entry}


@router.get("/pool/backtest-summaries")
async def pool_backtest_summaries_list(
    user: dict = Depends(get_current_user),
):
    """列出已保存的回测摘要"""
    items = _load_backtest_summaries(user["user_id"])
    return {"result": items}


@router.delete("/pool/backtest-summaries/{summary_id}")
async def pool_backtest_summary_delete(
    summary_id: str,
    user: dict = Depends(get_current_user),
):
    """按 id 删除一条回测摘要"""
    items = _load_backtest_summaries(user["user_id"])
    before = len(items)
    items = [x for x in items if x.get("id") != summary_id]
    if len(items) < before:
        _save_backtest_summaries(user["user_id"], items)
        return {"ok": True, "message": "已删除"}
    return {"ok": False, "message": "未找到该摘要"}


@router.get("/pool/industry-names")
async def pool_industry_names():
    """获取东方财富行业板块名称列表（用于分行业选股）"""
    try:
        from tools.stock_pool import list_industry_names
        names = list_industry_names()
        return {"result": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/available-factors")
async def pool_available_factors():
    """获取因子挖掘可选因子列表：id、中文名、类别（风格与估值/情绪与交易及子类中文名）、描述"""
    try:
        from modules.factor_mining.factor_registry import get_available_factors
        factors = get_available_factors()
        return {"result": factors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/factor-storage/names")
async def pool_factor_storage_names():
    """获取数据库中已保存的因子名列表（关系型存储）"""
    try:
        from db.factor_storage import get_stored_factor_names
        names = get_stored_factor_names()
        return {"result": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/factor-storage/range")
async def pool_factor_storage_range(
    symbol: str = Query("", description="股票代码，可选"),
    factor_name: str = Query("", description="因子名，可选"),
):
    """获取已保存因子数据的日期范围 (min_date, max_date)"""
    try:
        from db.factor_storage import get_factor_date_range
        out = get_factor_date_range(
            symbol=symbol.strip() or None,
            factor_name=factor_name.strip() or None,
        )
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/industry-leaders")
async def pool_industry_leaders(
    industries: str = Query(..., description="行业名称，逗号分隔，如 银行,电力,白酒"),
    per: int = Query(1, description="每行业取龙头数量"),
):
    """分行业选龙头预览"""
    try:
        from tools.stock_pool import select_industry_leaders
        industry_list = [s.strip() for s in industries.split(",") if s.strip()]
        if not industry_list:
            return {"result": [], "code_to_industry": {}}
        codes, code_to_industry = select_industry_leaders(industry_list, leaders_per_industry=per)
        return {"result": codes, "code_to_industry": code_to_industry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/report")
async def pool_report(req: StockPoolReportRequest):
    """生成选股池分析报告"""
    try:
        from tools.stock_pool import generate_pool_report
        report = generate_pool_report(
            stocks=req.stocks,
            total_capital=req.total_capital,
            risk_preference=req.risk_preference
        )
        return {"result": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/sim/create")
async def create_pool_sim_account(
    req: StockPoolSimAccountCreateRequest,
    user: dict = Depends(get_current_user),
):
    """创建选股池模拟仓账户"""
    try:
        from modules.stock_pool.sim_account import StockPoolSimAccountManager
        if not req.account_id or not req.account_id.strip():
            raise HTTPException(status_code=400, detail="账户ID不能为空")
        account_id = req.account_id.strip()
        if not re.match(r'^[\w\u4e00-\u9fa5-]+$', account_id):
            raise HTTPException(status_code=400, detail="账户ID只能包含字母、数字、下划线、连字符和中文")
        data_dir = get_user_pool_sim_dir(user["user_id"])
        os.makedirs(data_dir, exist_ok=True)
        manager = StockPoolSimAccountManager(data_dir=data_dir)
        account = manager.create_account(account_id, req.initial_capital, req.stock_pool or [])
        return {"result": account.to_dict()}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/sim/list")
async def list_pool_sim_accounts(user: dict = Depends(get_current_user)):
    """列出所有选股池模拟仓账户"""
    try:
        from modules.stock_pool.sim_account import StockPoolSimAccountManager
        data_dir = get_user_pool_sim_dir(user["user_id"])
        os.makedirs(data_dir, exist_ok=True)
        manager = StockPoolSimAccountManager(data_dir=data_dir)
        account_ids = manager.list_accounts()
        accounts_info = []
        for acc_id in account_ids:
            account = manager.get_account(acc_id)
            if account:
                accounts_info.append({
                    "account_id": account.account_id,
                    "initial_capital": account.initial_capital,
                    "cash": account.cash,
                    "positions_count": len(account.positions),
                    "trades_count": len(account.trades),
                    "stock_pool": account.stock_pool,
                    "created_at": account.created_at,
                    "last_updated": account.last_updated
                })
        return {"result": accounts_info}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/sim/{account_id}")
async def get_pool_sim_account(
    account_id: str,
    user: dict = Depends(get_current_user),
):
    """获取选股池模拟仓账户信息"""
    try:
        from modules.stock_pool.sim_account import StockPoolSimAccountManager
        from tools.stock_data import get_latest_close
        data_dir = get_user_pool_sim_dir(user["user_id"])
        manager = StockPoolSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        current_prices = {}
        for stock_code in account.positions.keys():
            try:
                current_prices[stock_code] = get_latest_close(stock_code, use_cache=True)
            except Exception as e:
                print(f"[选股池模拟仓] 获取持仓{stock_code}价格失败: {e}")
        total_equity = account.get_total_equity(current_prices)
        positions_value = account.get_positions_value(current_prices)
        profit_loss = total_equity - account.initial_capital
        profit_loss_pct = (profit_loss / account.initial_capital * 100) if account.initial_capital > 0 else 0
        positions_detail = []
        for stock_code, pos in account.positions.items():
            current_price = current_prices.get(stock_code, pos['avg_cost'])
            market_value = pos['shares'] * current_price
            profit_loss_pos = (current_price - pos['avg_cost']) * pos['shares']
            profit_loss_pct_pos = (current_price / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] > 0 else 0
            positions_detail.append({
                'stock_code': stock_code,
                'shares': pos['shares'],
                'avg_cost': pos['avg_cost'],
                'current_price': current_price,
                'market_value': market_value,
                'profit_loss': profit_loss_pos,
                'profit_loss_pct': profit_loss_pct_pos,
                'entry_date': pos['entry_date']
            })
        return {
            "result": {
                "account": account.to_dict(),
                "statistics": {
                    "total_equity": round(total_equity, 2),
                    "cash": round(account.cash, 2),
                    "positions_value": round(positions_value, 2),
                    "profit_loss": round(profit_loss, 2),
                    "profit_loss_pct": round(profit_loss_pct, 2),
                    "initial_capital": account.initial_capital
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


@router.post("/pool/sim/{account_id}/rebalance")
async def rebalance_pool_sim_account(
    account_id: str,
    req: StockPoolSimRebalanceRequest,
    user: dict = Depends(get_current_user),
):
    """执行选股池模拟仓调仓"""
    try:
        from modules.stock_pool.sim_account import StockPoolSimAccountManager
        from tools.stock_pool import rebalance_account
        from tools.stock_data import get_latest_close
        from tools.risk_control import audit_risk, RiskAuditInput
        data_dir = get_user_pool_sim_dir(user["user_id"])
        manager = StockPoolSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        current_prices = {}
        all_stocks = set(account.stock_pool) | set(account.positions.keys())
        for stock_code in all_stocks:
            try:
                current_prices[stock_code] = get_latest_close(stock_code, use_cache=True)
            except Exception as e:
                print(f"[选股池模拟仓] 获取{stock_code}价格失败: {e}")
        if not current_prices:
            raise HTTPException(status_code=400, detail="无法获取股票价格数据")
        total_equity = account.get_total_equity(current_prices)
        positions_value = account.get_positions_value(current_prices)
        positions_for_risk = {
            code: {'market_value': account.positions[code]['shares'] * current_prices[code]}
            for code in account.positions if code in current_prices
        }
        peak_equity = getattr(account, 'peak_equity', None) or total_equity
        risk_input = RiskAuditInput(
            cash=account.cash,
            total_value=total_equity,
            initial_capital=account.initial_capital,
            positions_value=positions_value,
            positions=positions_for_risk,
            recent_daily_returns=[],
            peak_value=peak_equity,
        )
        risk_result = audit_risk(risk_input)
        if not risk_result.pass_audit and (risk_result.action in ('stop', 'pause_trading') or (risk_result.action == 'reduce_position' and 'over_weight_code' not in risk_result.details)):
            manager.update_account(account)
            return {"result": {"skipped": True, "reason": risk_result.reason, "details": risk_result.details}}
        result = rebalance_account(
            account=account,
            current_prices=current_prices,
            df_dict=None,
            risk_preference=req.risk_preference,
            use_llm=req.use_llm,
            llm=None
        )
        manager.update_account(account)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/sim/{account_id}/history")
async def get_pool_sim_account_history(
    account_id: str,
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    """获取选股池模拟仓交易历史"""
    try:
        from modules.stock_pool.sim_account import StockPoolSimAccountManager
        data_dir = get_user_pool_sim_dir(user["user_id"])
        manager = StockPoolSimAccountManager(data_dir=data_dir)
        account = manager.get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"账户{account_id}不存在")
        recent_trades = account.trades[-limit:] if len(account.trades) > limit else account.trades
        return {"result": {"trades": recent_trades, "total_trades": len(account.trades)}}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/list")
async def list_pools(user: dict = Depends(get_current_user)):
    """列出所有已保存的选股池"""
    try:
        from tools.stock_pool import list_saved_pools
        pools_dir = get_user_pools_dir(user["user_id"])
        os.makedirs(pools_dir, exist_ok=True)
        pools = list_saved_pools(directory=pools_dir)
        return {"result": pools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pool/save")
async def save_pool(
    req: StockPoolSaveRequest,
    user: dict = Depends(get_current_user),
):
    """保存选股池配置"""
    try:
        from tools.stock_pool import create_stock_pool, save_pool_config
        config = create_stock_pool(
            stocks=req.stocks,
            initial_capital=req.initial_capital,
            strategy_type=req.strategy_type,
            risk_preference=req.risk_preference,
            allocation_method=req.allocation_method,
            name=req.name,
            selection_mode=req.selection_mode,
            selection_top_n=req.selection_top_n,
            selection_interval=req.selection_interval,
            score_weights=req.score_weights,
            factor_set=req.factor_set,
            strategy_meta=req.strategy_meta,
            factor_profile=req.factor_profile,
        )
        pools_dir = get_user_pools_dir(user["user_id"])
        os.makedirs(pools_dir, exist_ok=True)
        filepath = os.path.join(pools_dir, f"{config.name.replace(' ', '_')}.json")
        save_pool_config(config, filepath=filepath)
        return {"result": {"filepath": filepath, "name": req.name}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/load/{pool_name}")
async def load_pool(
    pool_name: str,
    user: dict = Depends(get_current_user),
):
    """加载指定的选股池配置"""
    try:
        from tools.stock_pool import load_pool_config
        pools_dir = get_user_pools_dir(user["user_id"])
        filepath = os.path.join(pools_dir, f"{pool_name.replace(' ', '_')}.json")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"选股池 '{pool_name}' 不存在")
        config = load_pool_config(filepath)
        return {
            "result": {
                "name": config.name,
                "stocks": config.stocks,
                "initial_capital": config.initial_capital,
                "strategy_type": config.strategy_type,
                "risk_preference": config.risk_preference,
                "allocation_method": config.allocation_method,
                "selection_mode": getattr(config, "selection_mode", "none"),
                "selection_top_n": getattr(config, "selection_top_n", 10),
                "selection_interval": getattr(config, "selection_interval", 0),
                "score_weights": getattr(config, "score_weights", None),
                "factor_set": getattr(config, "factor_set", "hybrid"),
                "strategy_meta": getattr(config, "strategy_meta", {}),
                "factor_profile": getattr(config, "factor_profile", {}),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pool/delete/{pool_name}")
async def delete_pool(
    pool_name: str,
    user: dict = Depends(get_current_user),
):
    """删除指定的选股池配置"""
    try:
        pools_dir = get_user_pools_dir(user["user_id"])
        safe_name = os.path.basename(pool_name).replace("..", "")
        filepath = os.path.join(pools_dir, f"{safe_name.replace(' ', '_')}.json")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"选股池 '{pool_name}' 不存在")
        os.remove(filepath)
        return {"result": {"deleted": True, "name": pool_name}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
