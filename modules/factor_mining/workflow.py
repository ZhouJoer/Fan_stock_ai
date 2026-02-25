"""
因子挖掘 LangGraph 工作流：数据加载 → 因子计算与相关性 → LLM 选因子 → 钟形变换 → 训练 → Alpha/Beta → 策略/轮仓报告。
"""

from __future__ import annotations

import threading
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from langgraph.graph import END, StateGraph

# 线程局部：由调用方（如 deep_factor_search）在 invoke 前注入，供节点上报进度与响应中止
_factor_mining_tls = threading.local()


def set_factor_mining_progress(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> None:
    """在 invoke 前调用，注入当前线程的进度回调和中止检查（如 API 的 msg_queue / abort_ev）。"""
    _factor_mining_tls.progress_callback = progress_callback
    _factor_mining_tls.abort_check = abort_check


def get_factor_mining_progress() -> tuple:
    """返回 (progress_callback, abort_check)，未注入时为 (None, None)。"""
    return (
        getattr(_factor_mining_tls, "progress_callback", None),
        getattr(_factor_mining_tls, "abort_check", None),
    )

from .factor_definitions import FACTORS_SUGGEST_BELL
from .factor_eval import (
    backtest_stats,
    build_benchmark_horizon_returns,
    compute_alpha_beta,
    run_backtest,
)
from .factor_trainer import (
    ALL_FACTOR_KEYS as TRAINER_ALL_KEYS,
    TrainConfig,
    apply_bell_transforms,
    build_training_samples,
    train_linear_softmax_weights,
)
from .prompts import (
    NODE_ROLES,
    FactorSelectionOutput,
    get_agent_orchestration_prompt,
    get_factor_selection_prompt,
    get_report_generation_prompt,
)


class FactorMiningState(TypedDict, total=False):
    universe_codes: List[str]
    days: int
    label_horizon: int
    rebalance_freq: int  # 调仓周期（交易日数），1=每日调仓，5=约周频
    max_combos: int  # try_combinations 枚举组合数上限，默认 15
    mode: str
    data_dict: Dict[str, pd.DataFrame]
    factor_df: pd.DataFrame
    factor_corr_matrix: pd.DataFrame
    factor_names: List[str]
    selected_factors: List[str]
    bell_transforms: List[str]
    train_result: Dict[str, Any]
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    alpha: float
    beta: float
    alpha_beta: Dict[str, float]
    backtest_stats: Dict[str, float]
    strategy_logic: str
    rotation_logic: str
    benchmark_code: str
    error: str


def _get_llm():
    try:
        from llm import tool_llm
        return tool_llm
    except Exception:
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1)


def load_data_node(state: FactorMiningState) -> Dict[str, Any]:
    """加载股票行情；可选加载基准（如 510300）用于 alpha/beta。"""
    codes = state.get("universe_codes") or []
    days = max(252, state.get("days", 252) + 100)
    benchmark_code = state.get("benchmark_code") or "510300"
    out: Dict[str, Any] = {}
    progress_cb, _ = get_factor_mining_progress()
    if progress_cb:
        progress_cb(0, 1, f"load_data: 正在加载 {len(codes)} 只股票行情…")
    try:
        from tools.stock_data import get_stock_data
    except ImportError as e:
        out["error"] = f"tools.stock_data.get_stock_data 不可用: {e}"
        return out
    data_dict: Dict[str, pd.DataFrame] = {}
    min_rows = 60  # 与 get_stock_data 最低要求一致，便于更多标的通过
    for code in codes:
        try:
            raw = get_stock_data(code, days, use_cache=True)
            if raw and len(raw) >= min_rows:
                df = pd.DataFrame(raw)
                if {"date", "close", "high", "low", "volume"}.issubset(df.columns):
                    data_dict[code] = df.dropna().reset_index(drop=True)
        except Exception:
            continue
    out["data_dict"] = data_dict
    print(f"[factor_mining] load_data: 请求 {len(codes)} 只，有效加载 {len(data_dict)} 只（每只≥{min_rows} 条）")
    if progress_cb:
        progress_cb(0, 1, f"load_data: 已加载 {len(data_dict)}/{len(codes)} 只股票行情")
    if len(data_dict) < 8:
        out["error"] = (
            f"有效行情数据不足：请求 {len(codes)} 只，仅加载到 {len(data_dict)} 只（需至少 8 只且每只≥{min_rows} 条）。"
            "请检查股票池、网络或数据源；若仅见拉取打印而无成功条数，多为接口限流或单只数据不足。"
        )
    benchmark_df = None
    try:
        raw_b = get_stock_data(benchmark_code, days, use_cache=True)
        if raw_b and len(raw_b) >= 10:
            benchmark_df = pd.DataFrame(raw_b)
    except Exception:
        pass
    if benchmark_df is not None and "date" in benchmark_df.columns and "close" in benchmark_df.columns:
        bench = benchmark_df.sort_values("date")
        bench["ret"] = bench["close"].astype(float).pct_change()
        out["benchmark_returns"] = bench.set_index("date")["ret"].dropna()
    else:
        out["benchmark_returns"] = pd.Series(dtype=float)
    return out


def compute_factors_node(state: FactorMiningState) -> Dict[str, Any]:
    """构建训练样本并计算因子相关矩阵。"""
    progress_cb, _ = get_factor_mining_progress()
    data_dict = state.get("data_dict") or {}
    if not data_dict:
        print("[factor_mining] compute_factors: 无行情数据，跳过")
        return {"error": "无行情数据"}
    days = state.get("days", 252)
    label_horizon = state.get("label_horizon", 1)
    if progress_cb:
        progress_cb(0, 1, f"compute_factors: 正在用 {len(data_dict)} 只股票构建因子…")
    extra_by_code_date = {}
    start_d, end_d = "", ""
    try:
        from modules.factor_mining.cross_section_data import fetch_pe_turnover_for_codes
        dates = []
        for df in data_dict.values():
            if df is not None and not df.empty and "date" in df.columns:
                dates.extend(df["date"].astype(str).str[:10].tolist())
        if dates:
            start_d = min(dates)
            end_d = max(dates)
            extra_by_code_date = fetch_pe_turnover_for_codes(list(data_dict.keys()), start_d, end_d)
    except Exception:
        pass
    label_h = max(1, state.get("label_horizon", 1))
    factor_df = None
    if start_d and end_d:
        try:
            from db.factor_storage import load_factor_df
            loaded = load_factor_df(
                symbols=list(data_dict.keys()),
                start_date=start_d,
                end_date=end_d,
                factor_names=list(TRAINER_ALL_KEYS),
                label_horizon=label_h,
            )
            if loaded is not None and len(loaded) >= 80:
                factor_df = loaded
                print(f"[factor_mining] compute_factors: 从数据库加载因子 {len(factor_df)} 条", flush=True)
        except Exception as _e:
            pass
    if factor_df is None or factor_df.empty:
        try:
            factor_df = build_training_samples(
                data_dict=data_dict,
                days=days,
                label_horizon=label_h,
                max_window=120,
                extra_factors_by_code_date=extra_by_code_date if extra_by_code_date else None,
            )
        except Exception as e:
            print(f"[factor_mining] compute_factors: 构建样本异常: {e}")
            return {"error": str(e)}
        if factor_df is not None and not factor_df.empty:
            try:
                from db.factor_storage import save_factor_df
                factor_cols_to_save = [c for c in factor_df.columns if c not in ("date", "stock_code", "y")]
                if factor_cols_to_save:
                    save_factor_df(factor_df, factor_columns=factor_cols_to_save, label_horizon=label_h)
                    print(f"[factor_mining] compute_factors: 已保存因子到数据库", flush=True)
            except Exception as _e:
                pass
    if factor_df is None or factor_df.empty:
        print("[factor_mining] compute_factors: 训练样本为空")
        return {"error": "训练样本为空"}
    factor_cols = [c for c in factor_df.columns if c in TRAINER_ALL_KEYS]
    if not factor_cols:
        factor_cols = [c for c in factor_df.columns if c not in ("date", "stock_code", "y")]
    if not factor_cols:
        print("[factor_mining] compute_factors: 无因子列")
        return {"error": "无因子列"}
    corr = factor_df[factor_cols].astype(float).corr()
    print(f"[factor_mining] compute_factors: 样本 {len(factor_df)} 条，因子 {len(factor_cols)} 个", flush=True)
    if progress_cb:
        progress_cb(0, 1, f"compute_factors: 完成，{len(factor_df)} 条样本，{len(factor_cols)} 个因子")
    return {
        "factor_df": factor_df,
        "factor_corr_matrix": corr,
        "factor_names": factor_cols,
        "error": None,  # 成功则清除上游可能遗留的 error（如 load_data 的「有效数据不足」）
    }


def _enumerate_factor_combos(
    factor_names: List[str],
    corr_df: Optional[pd.DataFrame],
    mode: str,
    max_combos: int = 15,
    max_abs_corr: float = 0.6,
) -> List[List[str]]:
    """枚举低相关因子组合：dual 为所有 |r|<max_abs_corr 的二元组；multi 为 3～5 因子组合，限制数量。"""
    corr_empty = corr_df is None or (getattr(corr_df, "empty", True) if corr_df is not None else True)
    if not factor_names or corr_empty:
        return [factor_names[:1]] if factor_names else []
    mode = (mode or "single").lower()
    out: List[List[str]] = []

    def _max_pairwise_corr(combo: List[str]) -> float:
        m = -1.0
        for i, a in enumerate(combo):
            if a not in corr_df.columns:
                continue
            for b in combo[i + 1 :]:
                if b not in corr_df.columns:
                    continue
                r = corr_df.loc[a, b] if b in corr_df.index else corr_df.loc[b, a]
                if np.isfinite(r):
                    m = max(m, abs(float(r)))
        return m

    if mode == "dual" and len(factor_names) >= 2:
        for i, a in enumerate(factor_names):
            if a not in corr_df.columns:
                continue
            for b in factor_names[i + 1 :]:
                if b not in corr_df.columns:
                    continue
                r = corr_df.loc[a, b] if b in corr_df.index else corr_df.loc[b, a]
                if np.isfinite(r) and abs(float(r)) < max_abs_corr:
                    out.append([a, b])
        if not out:
            out = [factor_names[:2]]
    elif mode == "multi" and len(factor_names) >= 3:
        for k in range(3, min(6, len(factor_names) + 1)):
            for combo in combinations(factor_names, k):
                combo_list = list(combo)
                if _max_pairwise_corr(combo_list) >= max_abs_corr:
                    continue
                out.append(combo_list)
                if len(out) >= max_combos:
                    break
            if len(out) >= max_combos:
                break
        if not out:
            out = [factor_names[: min(5, len(factor_names))]]
    else:
        out = [factor_names[:1]] if mode == "single" else [factor_names[: min(5, len(factor_names))]]
    return out[:max_combos]


def _drop_high_correlation_factors(
    selected: List[str],
    corr_df: Optional[pd.DataFrame],
    max_abs_corr: float = 0.6,
) -> List[str]:
    """从已选因子中剔除高相关因子，保证任意两两 |r| < max_abs_corr。遇高相关对时保留列表中靠前的因子。"""
    corr_empty = corr_df is None or (getattr(corr_df, "empty", True) if corr_df is not None else True)
    if not selected or corr_empty or len(selected) <= 1:
        return selected
    remaining = list(selected)
    while True:
        dropped_any = False
        for i, a in enumerate(remaining):
            if a not in corr_df.columns:
                continue
            for b in remaining[i + 1 :]:
                if b not in corr_df.columns:
                    continue
                r = corr_df.loc[a, b] if a in corr_df.index and b in corr_df.columns else np.nan
                if np.isfinite(r) and abs(float(r)) > max_abs_corr:
                    remaining.remove(b)
                    dropped_any = True
                    break
            if dropped_any:
                break
        if not dropped_any:
            break
    return remaining


def select_factors_llm_node(state: FactorMiningState) -> Dict[str, Any]:
    """LLM 选取因子（单/双/多）并决定钟形变换；选后过滤高相关因子。"""
    mode = (state.get("mode") or "single").lower()
    _fn = state.get("factor_names")
    factor_names = list(_fn) if isinstance(_fn, (list, tuple)) else (list(_fn) if hasattr(_fn, "__iter__") and not isinstance(_fn, (str, pd.DataFrame)) else [])
    corr_df = state.get("factor_corr_matrix")
    if not factor_names:
        return {"error": "无候选因子"}
    corr_text = ""
    if corr_df is not None and not corr_df.empty:
        corr_text = corr_df.round(3).to_string()
    suggest_bell = list(FACTORS_SUGGEST_BELL)
    prompt = get_factor_selection_prompt(mode, factor_names, corr_text, suggest_bell)
    llm = _get_llm()
    selected: List[str] = []
    bell_transforms: List[str] = []
    if FactorSelectionOutput is not None and hasattr(llm, "with_structured_output"):
        try:
            chain = llm.with_structured_output(FactorSelectionOutput)
            result = chain.invoke(prompt)
            selected = getattr(result, "selected_factors", []) or []
            bell_transforms = getattr(result, "bell_transforms", []) or []
        except Exception:
            pass
    if not selected and (isinstance(factor_names, (list, tuple)) and len(factor_names) > 0):
        if mode == "single":
            selected = [factor_names[0]]
        elif mode == "dual" and len(factor_names) >= 2:
            selected = factor_names[:2]
        else:
            selected = factor_names[: min(5, len(factor_names))]
    for f in list(selected):
        if f not in factor_names:
            selected = [x for x in selected if x != f]
    if mode == "single" and len(selected) > 1:
        selected = selected[:1]
    if mode == "dual" and len(selected) != 2:
        if len(selected) < 2 and len(factor_names) >= 2:
            selected = factor_names[:2]
        else:
            selected = selected[:2]
    if mode == "multi" and len(selected) > 5:
        selected = selected[:5]
    # 后处理：剔除高相关因子，保证任意两两 |r| < 0.6
    selected = _drop_high_correlation_factors(selected, corr_df, max_abs_corr=0.6)
    # 过滤后若不足，按 mode 从候选里补足（尽量选与已选相关性低的）
    if mode == "dual" and len(selected) < 2 and len(factor_names) >= 2:
        for f in factor_names:
            if f in selected:
                continue
            if corr_df is not None and not corr_df.empty and f in corr_df.columns:
                if all(
                    not (np.isfinite(corr_df.loc[f, s]) and abs(float(corr_df.loc[f, s])) >= 0.6)
                    for s in selected if s in corr_df.columns
                ):
                    selected.append(f)
                    break
        if len(selected) < 2:
            selected = factor_names[:2]
    if mode == "multi" and len(selected) < 3 and len(factor_names) >= 3:
        for f in factor_names:
            if len(selected) >= 5 or f in selected:
                continue
            if corr_df is not None and not corr_df.empty and f in corr_df.columns:
                if all(
                    not (np.isfinite(corr_df.loc[f, s]) and abs(float(corr_df.loc[f, s])) >= 0.6)
                    for s in selected if s in corr_df.columns
                ):
                    selected.append(f)
        if len(selected) < 3:
            selected = factor_names[: min(5, len(factor_names))]
    return {"selected_factors": selected, "bell_transforms": bell_transforms or []}


def try_combinations_node(state: FactorMiningState) -> Dict[str, Any]:
    """枚举多组因子组合，对每组训练、回测、算 alpha/beta，按年化 alpha + val_rank_ic 选最佳。"""
    print("[factor_mining] try_combinations: 进入节点", flush=True)
    factor_df = state.get("factor_df")
    _fn = state.get("factor_names")
    factor_names = list(_fn) if isinstance(_fn, (list, tuple)) else (list(_fn) if hasattr(_fn, "__iter__") and not isinstance(_fn, (str, pd.DataFrame)) else [])
    corr_df = state.get("factor_corr_matrix")
    mode = (state.get("mode") or "single").lower()
    print(f"[factor_mining] try_combinations: 已取 state, mode={mode}, 因子数={len(factor_names)}", flush=True)
    benchmark_returns = state.get("benchmark_returns")
    label_horizon = max(1, state.get("label_horizon", 1))
    factor_empty = factor_df is None or (getattr(factor_df, "empty", True) if factor_df is not None else True)
    if factor_empty or not factor_names:
        print("[factor_mining] try_combinations: 无因子表或因子列表，跳过", flush=True)
        return {"error": "无因子表或因子列表"}
    bench_daily = benchmark_returns if isinstance(benchmark_returns, pd.Series) else None
    if bench_daily is None or (getattr(bench_daily, "empty", True) if bench_daily is not None else True):
        bench_daily = pd.Series(dtype=float)
    rebalance_freq = max(1, state.get("rebalance_freq", 1))
    max_combos = max(1, min(150, int(state.get("max_combos", 15) or 15)))
    print("[factor_mining] try_combinations: 枚举候选组合...", flush=True)
    candidates = _enumerate_factor_combos(factor_names, corr_df, mode, max_combos=max_combos, max_abs_corr=0.6)
    if not candidates:
        candidates = [[factor_names[0]]] if factor_names else []
    print(f"[factor_mining] try_combinations: mode={mode}, 共 {len(candidates)} 组候选，开始逐组训练与回测（每组约 1–2 分钟）...", flush=True)
    progress_cb, abort_check = get_factor_mining_progress()
    if progress_cb:
        progress_cb(0, max(1, len(candidates)), f"共 {len(candidates)} 组候选，开始逐组训练与回测…")
    best_score = -1e18
    best_out: Dict[str, Any] = {}
    for idx, combo in enumerate(candidates):
        if abort_check and abort_check():
            print("[factor_mining] try_combinations: 收到中止信号", flush=True)
            break
        missing = [f for f in combo if f not in factor_df.columns]
        if missing:
            continue
        if progress_cb:
            progress_cb(idx, len(candidates), f"第 {idx + 1}/{len(candidates)} 组训练中…")
        print(f"[factor_mining] try_combinations: 第 {idx + 1}/{len(candidates)} 组 开始训练 {combo}...", flush=True)
        config = TrainConfig(
            factor_set="hybrid",
            epochs=220,
            val_ratio=0.2,
            feature_names=combo,
        )
        try:
            train_result = train_linear_softmax_weights(factor_df, config)
        except Exception as e:
            print(f"[factor_mining] try_combinations: 第 {idx + 1}/{len(candidates)} 组 {combo} 训练失败: {e}", flush=True)
            continue
        feature_names_out = train_result.get("feature_names") or train_result.get("best_factor_combo") or combo
        learned = train_result.get("learned_weights") or {}
        flat = learned.get("flat") or {}
        weights = np.array([flat.get(f, 0.0) for f in feature_names_out], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(feature_names_out)) / len(feature_names_out)
        try:
            strat_ser = run_backtest(factor_df, feature_names_out, weights, top_n=10, rebalance_freq=rebalance_freq)
        except Exception as e:
            print(f"[factor_mining] try_combinations: 第 {idx + 1}/{len(candidates)} 组 回测失败: {e}", flush=True)
            continue
        if strat_ser.empty:
            continue
        try:
            bench_horizon = build_benchmark_horizon_returns(bench_daily, strat_ser.index, label_horizon)
        except Exception as e:
            print(f"[factor_mining] try_combinations: 第 {idx + 1}/{len(candidates)} 组 基准对齐失败: {e}", flush=True)
            continue
        common = strat_ser.index.intersection(bench_horizon.index)
        if len(common) < 3:
            bench_aligned = pd.Series(0.0, index=strat_ser.index)
            common = strat_ser.index
        else:
            bench_aligned = bench_horizon.loc[common].astype(float)
        strat_aligned = strat_ser.loc[common].astype(float)
        ab = compute_alpha_beta(strat_aligned, bench_aligned, risk_free=0.0, annualize=True)
        metrics = train_result.get("metrics") or {}
        ann_alpha = float(ab.get("annualized_alpha", 0) or 0)
        val_rank_ic = float(metrics.get("val_rank_ic", 0) or 0)
        score = ann_alpha * 0.01 + val_rank_ic
        if score > best_score:
            best_score = score
            best_out = {
                "selected_factors": feature_names_out,
                "train_result": train_result,
                "strategy_returns": strat_ser,
                "alpha_beta": ab,
                "alpha": ab.get("alpha", 0.0),
                "beta": ab.get("beta", 0.0),
                "backtest_stats": backtest_stats(
                    strat_ser, periods_per_year=max(1, int(252 / rebalance_freq))
                ),
            }
            print(f"[factor_mining] try_combinations: 第 {idx + 1}/{len(candidates)} 组 新最佳 score={score:.4f} combo={feature_names_out}", flush=True)
    if not best_out:
        print("[factor_mining] try_combinations: 无有效组合（训练/回测均未通过）", flush=True)
        return {"error": "无有效组合"}
    print(f"[factor_mining] try_combinations: 完成，最佳组合 {best_out.get('selected_factors', [])}", flush=True)
    return best_out


def apply_transforms_node(state: FactorMiningState) -> Dict[str, Any]:
    """对指定因子做钟形变换并生成最终用于训练的因子列名。"""
    factor_df = state.get("factor_df")
    _sel = state.get("selected_factors")
    selected = list(_sel) if isinstance(_sel, (list, tuple)) else (list(_sel) if hasattr(_sel, "__iter__") and not isinstance(_sel, (str, pd.DataFrame)) else [])
    _bt = state.get("bell_transforms")
    bell_transforms = list(_bt) if isinstance(_bt, (list, tuple)) else (list(_bt) if hasattr(_bt, "__iter__") and not isinstance(_bt, (str, pd.DataFrame)) else [])
    if factor_df is None or (getattr(factor_df, "empty", True) if factor_df is not None else True) or not selected:
        return {}
    df = apply_bell_transforms(factor_df, bell_transforms, date_col="date")
    final_names: List[str] = []
    for f in selected:
        if f in bell_transforms and f"{f}_bell" in df.columns:
            final_names.append(f"{f}_bell")
        elif f in df.columns:
            final_names.append(f)
    if not final_names:
        final_names = [f for f in selected if f in df.columns]
    return {"factor_df": df, "selected_factors": final_names}


def train_or_weight_node(state: FactorMiningState) -> Dict[str, Any]:
    """用选中因子训练线性 Softmax 权重。"""
    factor_df = state.get("factor_df")
    _sel = state.get("selected_factors")
    selected = list(_sel) if isinstance(_sel, (list, tuple)) else (list(_sel) if hasattr(_sel, "__iter__") and not isinstance(_sel, (str, pd.DataFrame)) else [])
    if factor_df is None or (getattr(factor_df, "empty", True) if factor_df is not None else True) or not selected:
        return {"error": "无因子表或未选因子"}
    missing = [f for f in selected if f not in factor_df.columns]
    if missing:
        return {"error": f"样本缺少列: {missing}"}
    config = TrainConfig(
        factor_set="hybrid",
        epochs=220,
        val_ratio=0.2,
        feature_names=selected,
    )
    try:
        train_result = train_linear_softmax_weights(factor_df, config)
    except Exception as e:
        return {"error": str(e)}
    return {"train_result": train_result}


def compute_alpha_beta_node(state: FactorMiningState) -> Dict[str, Any]:
    """回测：按日选 top-N 等权，收益为 label_horizon 期远期收益；与同 horizon 基准做 CAPM 回归。"""
    factor_df = state.get("factor_df")
    train_result = state.get("train_result")
    benchmark_returns = state.get("benchmark_returns")
    label_horizon = max(1, state.get("label_horizon", 1))
    if factor_df is None or train_result is None:
        return {}
    feature_names = train_result.get("feature_names") or train_result.get("best_factor_combo") or []
    learned = train_result.get("learned_weights") or {}
    flat = learned.get("flat") or {}
    if not feature_names or not flat:
        return {"alpha_beta": {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0}}
    weights = np.array([flat.get(f, 0.0) for f in feature_names], dtype=float)
    # 线性权重可为负，仅当非线性且 sum<=0 或全零时才用等权
    if train_result.get("weight_type") != "linear":
        if weights.sum() <= 0 or np.all(weights == 0):
            weights = np.ones(len(feature_names)) / len(feature_names)
    elif np.all(weights == 0):
        weights = np.ones(len(feature_names)) / len(feature_names)
    rebalance_freq = max(1, state.get("rebalance_freq", 1))
    strat_ser = run_backtest(factor_df, feature_names, weights, top_n=10, rebalance_freq=rebalance_freq)
    if strat_ser.empty:
        return {"alpha_beta": {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0}}
    bench_daily = benchmark_returns if isinstance(benchmark_returns, pd.Series) else pd.Series(dtype=float)
    if bench_daily is None or bench_daily.empty:
        bench_horizon = pd.Series(0.0, index=strat_ser.index)
    else:
        bench_horizon = build_benchmark_horizon_returns(bench_daily, strat_ser.index, label_horizon)
    common = strat_ser.index.intersection(bench_horizon.index)
    if len(common) < 3:
        return {
            "strategy_returns": strat_ser,
            "alpha_beta": {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0},
        }
    strat_aligned = strat_ser.loc[common].astype(float)
    bench_aligned = bench_horizon.loc[common].astype(float)
    ab = compute_alpha_beta(strat_aligned, bench_aligned, risk_free=0.0, annualize=True)
    periods_per_year = max(1, int(252 / rebalance_freq))
    stats = backtest_stats(strat_ser, periods_per_year=periods_per_year)
    return {
        "strategy_returns": strat_ser,
        "alpha_beta": ab,
        "alpha": ab["alpha"],
        "beta": ab["beta"],
        "backtest_stats": stats,
    }


def generate_report_node(state: FactorMiningState) -> Dict[str, Any]:
    """LLM 生成策略逻辑与轮仓逻辑；若上游报错且无训练结果，直接输出错误说明。"""
    progress_cb, _ = get_factor_mining_progress()
    if progress_cb:
        progress_cb(0, 1, "report: 正在生成策略与轮仓报告…")
    err = state.get("error")
    train_result = state.get("train_result") or {}
    # 避免对 DataFrame/Series 做布尔判断导致 "truth value is ambiguous"
    has_error = err is not None and (str(err).strip() if isinstance(err, str) else True)
    has_train_result = isinstance(train_result, dict) and len(train_result) > 0
    selected = state.get("selected_factors") or (train_result.get("best_factor_combo") if isinstance(train_result, dict) else []) or []
    print(f"[factor_mining] report: 生成策略/轮仓说明（error={has_error}, 已选因子数={len(selected)}）")
    if has_error and not has_train_result:
        return {
            "strategy_logic": f"[异常] {str(err) if err is not None else '未知'}",
            "rotation_logic": "请检查：股票池是否有效、数据是否拉取成功、单只行情是否≥100 条、样本量是否足够。",
        }
    learned = (train_result.get("learned_weights") or {}) if isinstance(train_result, dict) else {}
    flat = learned.get("flat") or {}
    alpha = state.get("alpha", 0.0)
    beta = state.get("beta", 0.0)
    ab = state.get("alpha_beta") or {}
    metrics = (train_result.get("metrics") or {}) if isinstance(train_result, dict) else {}
    rebalance_freq = max(1, state.get("rebalance_freq", 1))
    prompt = get_report_generation_prompt(selected, flat, alpha, beta, metrics, rebalance_freq=rebalance_freq)
    llm = _get_llm()
    strategy_logic = ""
    rotation_logic = ""
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(parts) >= 2:
            strategy_logic = parts[0]
            rotation_logic = parts[1]
        elif len(parts) == 1:
            strategy_logic = parts[0]
    except Exception:
        strategy_logic = f"选股依据：{selected}，权重 {flat}。Alpha={alpha:.4f}, Beta={beta:.4f}。"
        rotation_logic = "建议按综合得分排序取 TopN，调仓频率可设为周或双周，等权再平衡。"
    if has_error and err is not None:
        strategy_logic = f"[存在异常] {str(err)} | {strategy_logic}"
    return {"strategy_logic": strategy_logic, "rotation_logic": rotation_logic}


def should_apply_transforms(state: FactorMiningState) -> str:
    bell = state.get("bell_transforms")
    if isinstance(bell, (list, tuple)) and len(bell) > 0:
        return "apply"
    return "skip"


def after_compute_factors(state: FactorMiningState) -> str:
    """无因子表时直接报告（compute_factors 失败）；有因子表则按 mode 走 try_combinations 或 select_factors_llm。"""
    factor_df = state.get("factor_df")
    try:
        is_empty = factor_df is None or (getattr(factor_df, "empty", True) if factor_df is not None else True)
    except (ValueError, TypeError):
        is_empty = True
    if is_empty:
        return "report"
    mode = (state.get("mode") or "single").lower()
    if mode in ("dual", "multi"):
        return "try_combinations"
    return "select_factors_llm"


def build_graph() -> Any:
    workflow = StateGraph(FactorMiningState)
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("compute_factors", compute_factors_node)
    workflow.add_node("try_combinations", try_combinations_node)
    workflow.add_node("select_factors_llm", select_factors_llm_node)
    workflow.add_node("apply_transforms", apply_transforms_node)
    workflow.add_node("train", train_or_weight_node)
    workflow.add_node("alpha_beta", compute_alpha_beta_node)
    workflow.add_node("report", generate_report_node)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "compute_factors")
    workflow.add_conditional_edges(
        "compute_factors",
        after_compute_factors,
        {
            "try_combinations": "try_combinations",
            "select_factors_llm": "select_factors_llm",
            "report": "report",
        },
    )
    workflow.add_edge("try_combinations", "report")
    workflow.add_conditional_edges("select_factors_llm", should_apply_transforms, {"apply": "apply_transforms", "skip": "train"})
    workflow.add_edge("apply_transforms", "train")
    workflow.add_edge("train", "alpha_beta")
    workflow.add_edge("alpha_beta", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


factor_mining_graph = build_graph()
