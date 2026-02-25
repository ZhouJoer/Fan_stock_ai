"""
深度因子挖掘（仅 LLM Workflow 方式）。

使用 LangGraph 工作流：加载数据 → 计算因子 → LLM 选因子（单/双/多）→ 钟形变换 → 训练 → Alpha/Beta → 策略与轮仓报告。

示例：
python -m tools.deep_factor_search --universe-source index --universe-index 000300 --max-stocks 60
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.factor_mining import factor_mining_graph
from modules.factor_mining.prompts import (
    get_agent_next_step_prompt,
    get_orchestration_agent_prompt,
    get_orchestration_tasks_prompt,
    get_report_generation_prompt,
    get_reviewer_prompt,
    get_reviewer_prompt_phase1,
    get_reviewer_prompt_phase2,
    parse_orchestration_json,
    parse_reviewer_json,
    parse_reviewer_phase1_json,
    parse_reviewer_phase2_json,
    OrchestrationPlanOutput,
)
from modules.factor_mining.factor_eval import (
    backtest_stats,
    build_benchmark_horizon_returns,
    compute_alpha_beta,
    compute_factor_quantile_returns,
    generate_backtest_chart,
    run_backtest,
    run_backtest_detailed,
    suggest_bell_from_quantile_returns,
)
from modules.factor_mining.factor_trainer import (
    apply_bell_transforms,
    build_training_samples,
    train_linear_weights,
)
from modules.factor_mining.workflow import (
    load_data_node,
    compute_factors_node,
    set_factor_mining_progress,
)
from modules.stock_pool import (
    filter_stock_universe,
    get_index_constituents,
    select_industry_leaders,
)


def _is_kechuang(code: str) -> bool:
    """科创板：上交所 688 开头"""
    c = str(code).strip()
    return len(c) >= 3 and c.startswith("688")


def _resolve_universe(args) -> List[str]:
    cap_scope = getattr(args, "cap_scope", "none") or "none"
    if cap_scope not in ("none", "only_small_cap", "exclude_small_cap"):
        cap_scope = "exclude_small_cap" if getattr(args, "exclude_small_cap", False) else "none"
    small_cap_billion = float(getattr(args, "small_cap_max_billion", 30))
    universe_source = getattr(args, "universe_source", "manual") or "manual"
    universe_index = (getattr(args, "universe_index", "") or "").strip()
    # 指数模式且未填代码时按沪深300处理，避免空池
    if universe_source == "index" and not universe_index:
        universe_index = "000300"

    def _get_index_codes(idx: str) -> List[str]:
        c = get_index_constituents(idx, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_billion)
        return c if c else []

    if universe_source == "index":
        codes = _get_index_codes(universe_index)
        # 指数解析失败或不足 8 只时，仅用常见指数兜底（勿用 510300：其为基准 ETF 代码，非成分股指数）
        if len(codes) < 8:
            for fallback in ("000300", "000016"):
                if fallback != universe_index:
                    codes = _get_index_codes(fallback)
                    if len(codes) >= 8:
                        break
    elif universe_source == "industry":
        inds = [x.strip() for x in (args.industry_list or "").split(",") if x.strip()]
        codes, _ = select_industry_leaders(
            inds or None,
            leaders_per_industry=max(1, int(args.leaders_per_industry)),
            cap_scope=cap_scope,
            small_cap_threshold_billion=small_cap_billion,
        )
    else:
        stocks_raw = getattr(args, "stocks", "") or ""
        if isinstance(stocks_raw, list):
            raw = [str(x).strip() for x in stocks_raw if str(x).strip()]
        else:
            raw = [x.strip() for x in str(stocks_raw).split(",") if x.strip()]
        dedup = []
        seen = set()
        for c in raw:
            if c and c not in seen:
                dedup.append(c)
                seen.add(c)
        codes = filter_stock_universe(
            dedup, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_billion
        )
    if getattr(args, "exclude_kechuang", False):
        codes = [c for c in codes if not _is_kechuang(c)]
    if getattr(args, "max_stocks", 0) > 0:
        codes = codes[: args.max_stocks]
    return codes


def _parse_agent_next_step_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 回复中解析 Agent 下一步 JSON，不依赖 response_format。"""
    if not (text and isinstance(text, str)):
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if "```" in text:
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                if end == -1:
                    end = len(text)
                text = text[start:end].strip()
                break
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        action = obj.get("action")
        if not action or not isinstance(action, str):
            return None
        return {
            "action": str(action).strip(),
            "factor_name": obj.get("factor_name") if obj.get("factor_name") is not None else None,
            "use_bell": obj.get("use_bell") if "use_bell" in obj else None,
            "reason": obj.get("reason"),
        }
    except (json.JSONDecodeError, TypeError):
        return None


def _get_llm():
    """获取 LLM 实例（与 workflow 一致，用于编排 Agent）。"""
    try:
        from llm import tool_llm
        return tool_llm
    except Exception:
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1)


# 编排多任务时的默认任务列表（Agent 失败或未返回时回退）
_DEFAULT_ORCHESTRATION_TASKS = [
    ("multi", 5, 1),
    ("multi", 5, 5),
    ("dual", 5, 1),
    ("multi", 3, 5),
    ("dual", 3, 1),
]


def _get_orchestration_tasks_from_agent(
    args,
    universe_size: int = 0,
) -> List[tuple]:
    """由 AI Agent 生成编排任务列表，返回 [(mode, label_horizon, rebalance_freq), ...]。失败时回退到默认列表。"""
    context = {
        "universe_size": universe_size or getattr(args, "max_stocks", 60),
        "days": int(getattr(args, "days", 252)),
        "benchmark_code": getattr(args, "benchmark_code", "510300") or "510300",
        "user_preference": getattr(args, "orchestrate_user_preference", None) or "深度挖掘多组因子组合，覆盖不同模式与调仓频率",
    }
    prompt = get_orchestration_tasks_prompt(context)
    if OrchestrationPlanOutput is None:
        return _DEFAULT_ORCHESTRATION_TASKS
    try:
        llm = _get_llm()
        if not hasattr(llm, "with_structured_output"):
            return _DEFAULT_ORCHESTRATION_TASKS
        chain = llm.with_structured_output(OrchestrationPlanOutput)
        out = chain.invoke(prompt)
        if not out or not getattr(out, "tasks", None):
            return _DEFAULT_ORCHESTRATION_TASKS
        tasks = []
        for t in out.tasks:
            mode = (getattr(t, "mode", None) or "multi").strip().lower()
            if mode not in ("single", "dual", "multi"):
                mode = "multi"
            h = max(1, min(20, int(getattr(t, "label_horizon", 5) or 5)))
            r = max(1, min(20, int(getattr(t, "rebalance_freq", 1) or 1)))
            tasks.append((mode, h, r))
        if not tasks:
            return _DEFAULT_ORCHESTRATION_TASKS
        # 去重并限制数量，避免过多任务
        seen = set()
        unique = []
        for x in tasks:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        return unique[:15] if len(unique) > 15 else unique
    except Exception as e:
        print(f"[search] 编排 Agent 生成任务失败，使用默认任务列表: {e}", flush=True)
        return _DEFAULT_ORCHESTRATION_TASKS


def _compute_alpha_beta_robust(
    strat_ser: pd.Series,
    bench_daily: pd.Series,
    label_horizon: int,
) -> Dict[str, float]:
    """用二分法对齐日期字符串，计算 CAPM alpha/beta，避免 index 类型不匹配导致全零问题。"""
    _zero: Dict[str, float] = {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0}
    if bench_daily is None or bench_daily.empty or strat_ser is None or strat_ser.empty:
        return _zero
    import bisect as _bisect
    bench = bench_daily.copy()
    bench.index = pd.Index([str(x)[:10] for x in bench.index])
    bench = bench.sort_index()
    bench_dates = list(bench.index)
    bench_vals = bench.values.astype(float)
    n = len(bench_dates)
    horizon = max(1, int(label_horizon))
    result: Dict[str, float] = {}
    for d in strat_ser.index:
        d_str = str(d)[:10]
        pos = _bisect.bisect_left(bench_dates, d_str)
        if pos + horizon > n:
            continue
        chunk = bench_vals[pos : pos + horizon]
        if len(chunk) < horizon:
            continue
        result[d] = float((1.0 + chunk).prod() - 1.0)
    if len(result) < 3:
        return _zero
    bench_s = pd.Series(result)
    common = strat_ser.index.intersection(bench_s.index)
    if len(common) < 3:
        return _zero
    return compute_alpha_beta(
        strat_ser.loc[common].astype(float),
        bench_s.loc[common].astype(float),
        risk_free=0.0,
        annualize=True,
    )


def _run_one_agent_trial(
    factor_df: pd.DataFrame,
    factor_names: List[str],
    corr_text: str,
    quantile_returns: Dict[str, Any],
    suggest_bell_per: Dict[str, bool],
    quality_per_factor: Dict[str, Any],
    factor_mode: str,
    max_factors: int,
    rebalance_freq: int,
    label_horizon: int,
    benchmark_returns: pd.Series,
    tried_combos: Optional[List] = None,
    trial_idx: int = 0,
    forced_main_factor: Optional[str] = None,
    top_n: int = 10,
    max_steps: int = 12,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """执行单次 agent 探索：设主因子 → 加因子 → 计算权重。返回 {combo, bell_transforms, train_result}。"""
    main_factor: Optional[str] = forced_main_factor  # 若外部强制指定主因子则直接使用
    other_factors: List[str] = []
    bell_transforms: List[str] = []
    last_metrics: Optional[dict] = None
    train_result: Optional[dict] = None

    for step in range(max_steps):
        if abort_check and abort_check():
            break
        if progress_callback:
            progress_callback(step, max_steps, f"[探索{trial_idx + 1}] 第 {step + 1} 步…")
        selected_now = ([main_factor] if main_factor else []) + other_factors
        selected_now = [f for f in selected_now if f]
        remaining = max(0, max_factors - len(selected_now))
        if remaining == 0:
            if train_result is None and selected_now:
                combo_key_now = tuple(sorted(selected_now))
                already_tried = bool(tried_combos and any(tuple(sorted(c)) == combo_key_now for c in tried_combos))
                if already_tried:
                    print(f"[search] Trial {trial_idx + 1} 已达上限且组合 {selected_now} 已试过，放弃本次探索", flush=True)
                    main_factor = None
                    other_factors = []
                    bell_transforms = []
                    break
                print(f"[search] Trial {trial_idx + 1} 已达上限 {max_factors}，强制 compute_weights", flush=True)
                action: str = "compute_weights"
                factor_name: Optional[str] = None
                use_bell: Optional[bool] = None
            else:
                break
        else:
            state_summary: Dict[str, Any] = {
                "factor_names": factor_names,
                "corr_matrix_text": corr_text,
                "main_factor": main_factor,
                "other_factors": list(other_factors),
                "bell_transforms": list(bell_transforms),
                "quantile_returns_per_factor": quantile_returns,
                "suggest_bell_per_factor": suggest_bell_per,
                "quality_per_factor": quality_per_factor,
                "factor_mode": factor_mode,
                "max_factors": max_factors,
                "remaining_slots": remaining,
                "tried_combos": [list(c) for c in (tried_combos or [])],
                "trial_idx": trial_idx,
                "forced_main_factor": forced_main_factor,
                "step_count": step,
                "max_steps": max_steps,
                "last_metrics": last_metrics,
            }
            prompt = get_agent_next_step_prompt(state_summary)
            try:
                llm = _get_llm()
                resp = llm.invoke(prompt)
                text = getattr(resp, "content", None) or str(resp)
                out = _parse_agent_next_step_json(text)
                if out is None:
                    print(f"[search] Trial {trial_idx + 1} 无法解析 JSON，终止本次探索", flush=True)
                    break
            except Exception as e:
                print(f"[search] Trial {trial_idx + 1} 解析失败: {e}", flush=True)
                break
            action = (out.get("action") or "").strip().lower()
            factor_name = (out.get("factor_name") or "").strip() or None
            use_bell = out.get("use_bell")

        if action == "finish":
            break
        if action == "set_main" and factor_name and factor_name in factor_names:
            if forced_main_factor:
                # 主因子已由外部固定，忽略 LLM 的 set_main 请求
                print(f"[search] Trial {trial_idx + 1} 主因子已固定为 {forced_main_factor}，忽略 set_main({factor_name})", flush=True)
            else:
                main_factor = factor_name
        elif action == "add_factor" and factor_name and factor_name in factor_names:
            selected_check = ([main_factor] if main_factor else []) + other_factors
            selected_check = [f for f in selected_check if f]
            if len(selected_check) >= max_factors:
                print(f"[search] Trial {trial_idx + 1} 上限已满，忽略 add_factor({factor_name})", flush=True)
                continue
            q_info = quality_per_factor.get(factor_name, {})
            if q_info.get("spread", 0.0) == 0.0:
                print(f"[search] Trial {trial_idx + 1} 因子 {factor_name} spread=0，跳过", flush=True)
                continue
            if factor_name not in selected_check:
                other_factors.append(factor_name)
                if use_bell:
                    bell_transforms.append(factor_name)
        elif action == "remove_factor" and factor_name:
            if factor_name == main_factor:
                main_factor = None
            elif factor_name in other_factors:
                other_factors.remove(factor_name)
            if factor_name in bell_transforms:
                bell_transforms.remove(factor_name)
        elif action == "compute_weights":
            selected = ([main_factor] if main_factor else []) + other_factors
            selected = [f for f in selected if f]
            if not selected:
                continue
            # 已试过的组合不再重算
            combo_key = tuple(sorted(selected))
            if tried_combos and any(tuple(sorted(c)) == combo_key for c in tried_combos):
                print(f"[search] Trial {trial_idx + 1} 组合 {selected} 已试过，跳过", flush=True)
                continue
            df = apply_bell_transforms(factor_df, [f for f in bell_transforms if f in selected], date_col="date")
            final_names: List[str] = []
            for f in selected:
                if f in bell_transforms and f"{f}_bell" in df.columns:
                    final_names.append(f"{f}_bell")
                elif f in df.columns:
                    final_names.append(f)
            if not final_names:
                final_names = [f for f in selected if f in df.columns]
            if not final_names:
                continue
            try:
                train_result = train_linear_weights(df, final_names, val_ratio=0.2, lambda_grid=[1e-5, 1e-4, 1e-3])
            except Exception as e:
                print(f"[search] Trial {trial_idx + 1} 训练失败: {e}", flush=True)
                last_metrics = {"error": str(e)}
                train_result = None
                continue
            last_metrics = train_result.get("metrics") or {}
            flat = (train_result.get("learned_weights") or {}).get("flat") or {}
            weights_arr = np.array([flat.get(f, 0.0) for f in final_names], dtype=float)
            if np.all(weights_arr == 0):
                weights_arr = np.ones(len(final_names)) / len(final_names)
            strat_ser, rb_details = run_backtest_detailed(df, final_names, weights_arr, top_n=top_n, rebalance_freq=rebalance_freq)
            if strat_ser.empty:
                continue
            ab = _compute_alpha_beta_robust(strat_ser, benchmark_returns, label_horizon)
            ppy = max(1, int(252 / rebalance_freq))
            bt_stats = backtest_stats(strat_ser, periods_per_year=ppy)
            chart_b64 = generate_backtest_chart(
                strat_ser, benchmark_returns,
                title=f"Trial {trial_idx + 1}: {' + '.join(final_names)}",
            )
            train_result["alpha_beta"] = ab
            train_result["backtest_stats"] = {k: float(v) if hasattr(v, "item") else v for k, v in bt_stats.items()}
            train_result["rebalance_details"] = rb_details
            train_result["chart_base64"] = chart_b64
            last_metrics = {**last_metrics, "backtest_sharpe": bt_stats.get("sharpe_annual", 0.0)}

    combo = ([main_factor] if main_factor else []) + other_factors
    combo = [f for f in combo if f]

    # 若循环结束仍无权重（如 finish 提前退出），做一次兜底训练
    if not train_result and combo:
        df = apply_bell_transforms(factor_df, [f for f in bell_transforms if f in combo], date_col="date")
        fnames: List[str] = []
        for f in combo:
            if f in bell_transforms and f"{f}_bell" in df.columns:
                fnames.append(f"{f}_bell")
            elif f in df.columns:
                fnames.append(f)
        if not fnames:
            fnames = [f for f in combo if f in df.columns]
        if fnames:
            try:
                train_result = train_linear_weights(df, fnames, val_ratio=0.2, lambda_grid=[1e-5, 1e-4, 1e-3])
                flat2 = (train_result.get("learned_weights") or {}).get("flat") or {}
                w2 = np.array([flat2.get(f, 0.0) for f in fnames], dtype=float)
                if np.all(w2 == 0):
                    w2 = np.ones(len(fnames)) / len(fnames)
                ss, rb2 = run_backtest_detailed(df, fnames, w2, top_n=top_n, rebalance_freq=rebalance_freq)
                if not ss.empty:
                    ab2 = _compute_alpha_beta_robust(ss, benchmark_returns, label_horizon)
                    bt2 = backtest_stats(ss, periods_per_year=max(1, int(252 / rebalance_freq)))
                    chart2 = generate_backtest_chart(
                        ss, benchmark_returns,
                        title=f"Trial {trial_idx + 1} (fallback): {' + '.join(fnames)}",
                    )
                    train_result["alpha_beta"] = ab2
                    train_result["backtest_stats"] = {k: float(v) if hasattr(v, "item") else v for k, v in bt2.items()}
                    train_result["rebalance_details"] = rb2
                    train_result["chart_base64"] = chart2
            except Exception:
                train_result = None

    if train_result and combo:
        flat_log = (train_result.get("learned_weights") or {}).get("flat") or {}
        print(f"[search] Trial {trial_idx + 1} done. combo={combo} weights={flat_log}", flush=True)

    return {"combo": combo, "bell_transforms": list(bell_transforms), "train_result": train_result}


# ============================================================================
# 三 Agent 架构
# ============================================================================

def _run_evaluation_agent(
    factor_df: pd.DataFrame,
    combo: List[str],
    suggest_bell_per: Dict[str, bool],
    quality_per_factor: Dict[str, Any],
    benchmark_returns: pd.Series,
    rebalance_freq: int,
    label_horizon: int,
    top_n: int,
    trial_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    评价 Agent（代码驱动）：评估给定因子组合。

    钟形变换决策：纯基于分层收益数据（suggest_bell_from_quantile_returns），
    只有因子收益不单调时才应用钟形变换，不依赖 LLM 判断。
    """
    if not combo:
        return None

    # ── 钟形变换决策：仅当分层收益不单调时才变换 ──
    bell_transforms: List[str] = []
    for f in combo:
        direction = quality_per_factor.get(f, {}).get("direction", "?")
        if suggest_bell_per.get(f, False):
            print(f"[eval] Trial {trial_idx + 1}: {f} dir={direction} → 非单调，应用钟形变换", flush=True)
            bell_transforms.append(f)
        else:
            print(f"[eval] Trial {trial_idx + 1}: {f} dir={direction} → 单调，无需钟形变换", flush=True)

    # ── 应用变换 ──
    df = apply_bell_transforms(factor_df, bell_transforms, date_col="date")

    # ── 构建最终特征名 ──
    final_names: List[str] = []
    for f in combo:
        if f in bell_transforms and f"{f}_bell" in df.columns:
            final_names.append(f"{f}_bell")
        elif f in df.columns:
            final_names.append(f)
    if not final_names:
        final_names = [f for f in combo if f in df.columns]
    if not final_names:
        print(f"[eval] Trial {trial_idx + 1}: 无有效特征列，跳过", flush=True)
        return None

    # ── 训练线性权重 ──
    try:
        train_result = train_linear_weights(df, final_names, val_ratio=0.2, lambda_grid=[1e-5, 1e-4, 1e-3])
    except Exception as e:
        print(f"[eval] Trial {trial_idx + 1} 训练失败: {e}", flush=True)
        return None

    flat = (train_result.get("learned_weights") or {}).get("flat") or {}
    w = np.array([flat.get(f, 0.0) for f in final_names], dtype=float)
    if np.all(w == 0):
        w = np.ones(len(final_names)) / len(final_names)

    # ── 基础回测（轻量，仅用于 Agent 评分比较；详细回测与图表由用户点击"仅回测"时生成）──
    strat_ser = run_backtest(df, final_names, w, top_n=top_n, rebalance_freq=rebalance_freq)
    if strat_ser.empty:
        print(f"[eval] Trial {trial_idx + 1}: 回测序列为空，跳过", flush=True)
        return None

    ab = _compute_alpha_beta_robust(strat_ser, benchmark_returns, label_horizon)
    ppy = max(1, int(252 / rebalance_freq))
    bt_stats = backtest_stats(strat_ser, periods_per_year=ppy)

    train_result["alpha_beta"] = ab
    train_result["backtest_stats"] = {k: float(v) if hasattr(v, "item") else v for k, v in bt_stats.items()}
    train_result["feature_names"] = final_names

    print(
        f"[eval] Trial {trial_idx + 1} done. combo={combo} final={final_names} "
        f"bells={bell_transforms} sharpe={bt_stats.get('sharpe_annual', 0.0):.3f}",
        flush=True,
    )
    return {
        "combo": combo,
        "final_names": final_names,
        "bell_transforms": bell_transforms,
        "train_result": train_result,
    }


def _fallback_orchestration(
    factor_names: List[str],
    quality_per_factor: Dict[str, Any],
    corr_df: Optional[pd.DataFrame],
    max_factors: int,
    n_trials: int,
) -> List[Dict[str, Any]]:
    """回退规划：当编排 Agent 失败时，按质量排序系统性生成多样组合，保证至少 n_trials 个。"""

    def qscore(f: str) -> float:
        q = quality_per_factor.get(f, {})
        return float(q.get("spread", 0.0) or 0.0) * (1.0 + 5.0 * abs(float(q.get("ic", 0.0) or 0.0)))

    def low_corr_with(f: str, chosen: List[str]) -> bool:
        if corr_df is None or corr_df.empty:
            return True
        cols = set(corr_df.columns)
        for c in chosen:
            if f in cols and c in cols:
                try:
                    r = abs(float(corr_df.loc[f, c]))
                    if r >= 0.5:
                        return False
                except Exception:
                    pass
        return True

    # 用全部因子按质量排序，避免因 qscore>0 的因子少而只生成少量组合
    sorted_factors = sorted(factor_names, key=lambda f: qscore(f), reverse=True)
    if not sorted_factors:
        return []

    combos: List[Dict[str, Any]] = []
    seen: set = set()
    # 对每个“主因子”，尝试多种组合规模（1 到 max_factors），以凑足 n_trials
    for size in range(1, max_factors + 1):
        for main in sorted_factors:
            if len(combos) >= n_trials:
                break
            selected = [main]
            for f in sorted_factors:
                if len(selected) >= size:
                    break
                if f not in selected and low_corr_with(f, selected):
                    selected.append(f)
            if len(selected) < size:
                selected = (selected + [f for f in sorted_factors if f not in selected])[:size]
            if not selected:
                continue
            key = tuple(sorted(selected))
            if key not in seen:
                seen.add(key)
                combos.append({"factors": selected, "reason": f"主因子={main}，系统回退规划"})
        if len(combos) >= n_trials:
            break
    # 若仍不足，用“主因子+低相关补充”再补一批
    for main in sorted_factors:
        if len(combos) >= n_trials:
            break
        selected = [main]
        for f in sorted_factors:
            if len(selected) >= max_factors:
                break
            if f not in selected and low_corr_with(f, selected):
                selected.append(f)
        key = tuple(sorted(selected))
        if key not in seen:
            seen.add(key)
            combos.append({"factors": selected, "reason": f"主因子={main}，系统回退规划"})
    return combos[:n_trials]


def _run_orchestration_agent(
    factor_names: List[str],
    quality_per_factor: Dict[str, Any],
    corr_df: Optional[pd.DataFrame],
    factor_mode: str,
    max_factors: int,
    n_trials: int,
    event_sink: Optional[Callable[[str, str, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    编排 Agent（LLM 驱动）：综合因子质量与相关性，规划 n_trials 个因子组合方案。
    只负责确定「选哪些因子组合」，钟形变换由评价 Agent 根据数据自动决定。
    解析失败时回退到系统规划。
    """
    fn_set = set(factor_names)

    # 高相关因子对（|r| ≥ 0.4）
    high_corr_parts: List[str] = []
    if corr_df is not None and not corr_df.empty:
        cols = list(corr_df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                try:
                    r = float(corr_df.iloc[i, j])
                    if abs(r) >= 0.4:
                        high_corr_parts.append(f"  {cols[i]} ↔ {cols[j]}: r={r:.2f}")
                except Exception:
                    pass
    high_corr_text = "\n".join(high_corr_parts) if high_corr_parts else "无明显高相关因子对（|r|<0.4）"

    prompt = get_orchestration_agent_prompt(
        quality_per_factor=quality_per_factor,
        high_corr_text=high_corr_text,
        factor_mode=factor_mode,
        max_factors=max_factors,
        n_trials=n_trials,
    )
    if event_sink:
        event_sink("orchestration", "input", prompt[:4000] if len(prompt) > 4000 else prompt)

    try:
        llm = _get_llm()
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        if event_sink:
            event_sink("orchestration", "output", text[:4000] if len(text) > 4000 else text)
        raw_combos = parse_orchestration_json(text)
        if raw_combos:
            valid: List[Dict[str, Any]] = []
            seen: set = set()
            for c in raw_combos:
                factors = [f for f in c["factors"] if f in fn_set]
                if not factors:
                    continue
                factors = factors[:max_factors]
                key = tuple(sorted(factors))
                if key in seen:
                    continue
                seen.add(key)
                valid.append({"factors": factors, "reason": c.get("reason", "")})
            if valid:
                print(f"[orchestration] Agent 规划了 {len(valid)} 个组合", flush=True)
                # 若不足 n_trials，用回退补充
                if len(valid) < n_trials:
                    fallback = _fallback_orchestration(
                        factor_names, quality_per_factor, corr_df, max_factors, n_trials - len(valid)
                    )
                    exist_keys = {tuple(sorted(v["factors"])) for v in valid}
                    for fb in fallback:
                        k = tuple(sorted(fb["factors"]))
                        if k not in exist_keys:
                            valid.append(fb)
                            exist_keys.add(k)
                        if len(valid) >= n_trials:
                            break
                return valid[:n_trials]
    except Exception as e:
        print(f"[orchestration] Agent 调用失败: {e}，使用系统规划", flush=True)

    return _fallback_orchestration(factor_names, quality_per_factor, corr_df, max_factors, n_trials)


def _apply_reviewer_verdict_rules(
    reviewer_out: Dict[str, Any],
    train_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据选中 trial 的收益/风险指标，强制修正审查结论，避免差结果被标为「推荐」。
    """
    out = dict(reviewer_out)
    ab = train_result.get("alpha_beta") or {}
    bs = train_result.get("backtest_stats") or {}
    ann_alpha = float(ab.get("annualized_alpha") or 0)
    sharpe = float(bs.get("sharpe_annual") or 0)
    max_dd = float(bs.get("max_drawdown") or 0)

    risks = list(out.get("risks") or [])
    def add_risk(msg: str) -> None:
        if msg and msg not in risks:
            risks.append(msg)
    out["risks"] = risks

    # 年化 Alpha 很负 或 夏普过低 → 必须不推荐
    if ann_alpha < -0.3 or sharpe < 0.3:
        out["verdict"] = "不推荐"
        out["quality_score"] = min(float(out.get("quality_score", 5)), 4.0)
        add_risk("年化 Alpha 为负或夏普过低，风险调整后收益差，不建议实盘。")
    # 年化 Alpha ≤ 0 或 回撤过深 → 不得推荐
    elif ann_alpha <= 0 or max_dd < -0.20:
        if out.get("verdict") == "推荐":
            out["verdict"] = "谨慎推荐"
        out["quality_score"] = min(float(out.get("quality_score", 5)), 6.0)
        if ann_alpha <= 0:
            add_risk("年化 Alpha 非正，超额收益不足，请谨慎使用。")
        if max_dd < -0.20:
            add_risk("最大回撤超过 20%，波动较大。")
    return out


def _run_reviewer_agent(
    trial_results: List[Dict[str, Any]],
    quality_per_factor: Dict[str, Any],
    corr_df: Optional[pd.DataFrame],
    factor_mode: str,
    rebalance_freq: int,
    label_horizon: int,
    event_sink: Optional[Callable[[str, str, str], None]] = None,
) -> Dict[str, Any]:
    """
    审查 Agent：综合审查所有 trial 的因子组合质量，选出最优组合并输出
    质量评分、风险提示、策略/轮仓说明。
    编排 Agent 负责决策下一步；审查 Agent 负责质量判断与报告生成。
    """
    if not trial_results:
        return {
            "selected_trial_idx": 0,
            "verdict": "reject",
            "quality_score": 0.0,
            "reliability": "low",
            "comments": [],
            "risks": ["无可用 trial"],
            "strategy_logic": "",
            "rotation_logic": "",
        }
    trial_summaries = []
    for r in trial_results:
        tr = r.get("train_result") or {}
        flat = (tr.get("learned_weights") or {}).get("flat") or {}
        trial_summaries.append({
            "combo": r.get("combo") or [],
            "weights": {k: round(v, 4) for k, v in flat.items()},
            "backtest_stats": tr.get("backtest_stats") or {},
            "metrics": tr.get("metrics") or {},
            "alpha_beta": tr.get("alpha_beta") or {},
        })
    corr_text = ""
    if corr_df is not None and not corr_df.empty:
        try:
            corr_text = corr_df.round(2).to_string()
        except Exception:
            pass
    llm = _get_llm()
    # 两阶段审查：先选 trial（数字来源唯一），再仅用该 trial 指标生成评审/策略/轮仓
    try:
        prompt1 = get_reviewer_prompt_phase1(
            trial_summaries=trial_summaries,
            quality_per_factor=quality_per_factor,
            corr_text=corr_text,
            factor_mode=factor_mode,
            rebalance_freq=rebalance_freq,
            label_horizon=label_horizon,
        )
        if event_sink:
            event_sink("reviewer", "input", "[Phase1] " + (prompt1[:3800] + "..." if len(prompt1) > 3800 else prompt1))
        resp1 = llm.invoke(prompt1)
        text1 = getattr(resp1, "content", None) or str(resp1)
        if event_sink:
            event_sink("reviewer", "output", "[Phase1] " + (text1[:1500] + "..." if len(text1) > 1500 else text1))
        out1 = parse_reviewer_phase1_json(text1)
        if out1 is None:
            raise ValueError("Phase1 解析失败")
        idx = max(0, min(int(out1.get("selected_trial_idx", 0)), len(trial_results) - 1))
        selected_summary = trial_summaries[idx]
        prompt2 = get_reviewer_prompt_phase2(
            selected_trial_summary=selected_summary,
            quality_per_factor=quality_per_factor,
            rebalance_freq=rebalance_freq,
            factor_mode=factor_mode,
            label_horizon=label_horizon,
        )
        if event_sink:
            event_sink("reviewer", "input", "[Phase2] " + (prompt2[:3800] + "..." if len(prompt2) > 3800 else prompt2))
        resp2 = llm.invoke(prompt2)
        text2 = getattr(resp2, "content", None) or str(resp2)
        if event_sink:
            event_sink("reviewer", "output", "[Phase2] " + (text2[:1500] + "..." if len(text2) > 1500 else text2))
        out2 = parse_reviewer_phase2_json(text2)
        if out2 is None:
            out2 = {"strategy_logic": "", "rotation_logic": "", "comments": [], "risks": []}
        out = {**out1, **out2, "selected_trial_idx": idx}
        print(f"[reviewer] 两阶段审查完成 selected_trial_idx={idx}", flush=True)
        return out
    except Exception as e:
        print(f"[reviewer] 两阶段审查失败，回退单阶段: {e}", flush=True)
    # 回退：单阶段
    try:
        prompt = get_reviewer_prompt(
            trial_summaries=trial_summaries,
            quality_per_factor=quality_per_factor,
            corr_text=corr_text,
            factor_mode=factor_mode,
            rebalance_freq=rebalance_freq,
            label_horizon=label_horizon,
        )
        if event_sink:
            event_sink("reviewer", "input", prompt[:4000] if len(prompt) > 4000 else prompt)
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        if event_sink:
            event_sink("reviewer", "output", text[:4000] if len(text) > 4000 else text)
        out = parse_reviewer_json(text)
        if out is not None:
            idx = max(0, min(int(out.get("selected_trial_idx", 0)), len(trial_results) - 1))
            out["selected_trial_idx"] = idx
            return out
    except Exception as e2:
        print(f"[reviewer] 审查 Agent 调用失败: {e2}", flush=True)
    # 兜底：按夏普选最优
    best_idx = max(
        range(len(trial_results)),
        key=lambda i: float((trial_results[i]["train_result"] or {}).get("backtest_stats", {}).get("sharpe_annual", 0.0) or 0.0),
    )
    combo = trial_results[best_idx].get("combo") or []
    return {
        "selected_trial_idx": best_idx,
        "verdict": "谨慎推荐",
        "quality_score": 5.0,
        "reliability": "中",
        "cap_recommendation": "全市场",
        "comments": ["审查 Agent 调用失败，按夏普比率自动选出最优组合"],
        "risks": ["建议人工核验因子有效性"],
        "strategy_logic": f"基于因子 {combo} 的线性合成评分进行选股。",
        "rotation_logic": f"按综合得分排序，取 TopN，每 {rebalance_freq} 交易日调仓，等权再平衡。",
    }


def _run_agent_driven_search(
    args,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
    event_sink: Optional[Callable[[str, str, str], None]] = None,
) -> dict:
    """
    三 Agent 协作的深度因子挖掘：
    1. 编排 Agent（LLM）：综合因子质量与相关性，规划 N 套因子组合方案
    2. 评价 Agent（代码驱动）：对每套组合分析分层收益 → 仅在非单调时做钟形变换
       → 训练线性权重（允许负权重）→ 详细回测（含调仓明细与图表）
    3. 审查 Agent（LLM）：综合评估所有结果，选出最优组合，生成策略报告
    """
    codes = _resolve_universe(args)
    if len(codes) < 8:
        raise ValueError(
            f"有效股票池过少: {len(codes)}，请至少提供 8 只股票或选择指数/行业"
        )
    set_factor_mining_progress(progress_callback=progress_callback, abort_check=abort_check)
    if progress_callback:
        progress_callback(0, 1, "Agent 编排: 加载数据与计算因子…")
    initial = {
        "universe_codes": codes,
        "days": int(args.days),
        "label_horizon": max(1, int(getattr(args, "label_horizon", 5))),
        "rebalance_freq": max(1, int(getattr(args, "rebalance_freq", 1) or 1)),
        "benchmark_code": getattr(args, "benchmark_code", "510300") or "510300",
    }
    state = {**initial, **load_data_node(initial)}
    if state.get("error"):
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": False,
            "workflow": True,
            "orchestrated": True,
            "agent_driven": True,
            "error": state["error"],
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": 0,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    state = {**state, **compute_factors_node(state)}
    if state.get("error"):
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": False,
            "workflow": True,
            "orchestrated": True,
            "agent_driven": True,
            "error": state.get("error", "计算因子失败"),
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": 0,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    factor_df = state.get("factor_df")
    factor_names = list(state.get("factor_names") or [])
    corr_df = state.get("factor_corr_matrix")
    _br = state.get("benchmark_returns")
    benchmark_returns = _br if (_br is not None and not getattr(_br, "empty", True)) else pd.Series(dtype=float)
    if factor_df is None or factor_df.empty or not factor_names:
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": False,
            "workflow": True,
            "orchestrated": True,
            "agent_driven": True,
            "error": "无因子表或因子列表为空",
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": 0,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    corr_text = corr_df.round(3).to_string() if corr_df is not None and not getattr(corr_df, "empty", True) else ""
    quantile_returns: Dict[str, Any] = {}
    suggest_bell_per: Dict[str, bool] = {}
    quality_per_factor: Dict[str, Any] = {}
    for f in factor_names[:25]:
        try:
            qr = compute_factor_quantile_returns(factor_df, f, q=5)
            quantile_returns[f] = qr
            sb = suggest_bell_from_quantile_returns(qr)
            suggest_bell_per[f] = sb
            vals = [float(qr.get(str(i), 0.0)) for i in range(len(qr))] if qr else []
            spread = round(max(vals) - min(vals), 6) if vals else 0.0
            if len(vals) >= 2:
                inc = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
                dec = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
                direction = "up" if inc else ("down" if dec else ("bell" if sb else "mixed"))
            else:
                direction = "unknown"
            # 截面 IC（rank correlation between factor and y，均值衡量预测能力）
            ic_list: List[float] = []
            for _, grp in factor_df.groupby("date"):
                if f in grp.columns and "y" in grp.columns and len(grp) >= 5:
                    valid = grp[[f, "y"]].dropna()
                    if len(valid) >= 5 and float(valid[f].std()) > 1e-10 and float(valid["y"].std()) > 1e-10:
                        ic_val = float(valid[f].corr(valid["y"], method="spearman"))
                        if not np.isnan(ic_val):
                            ic_list.append(ic_val)
            ic_mean = round(float(np.mean(ic_list)) if ic_list else 0.0, 5)
            ic_ir = round(float(np.mean(ic_list) / np.std(ic_list)) if len(ic_list) > 1 and float(np.std(ic_list)) > 1e-10 else 0.0, 4)
            quality_per_factor[f] = {"spread": spread, "direction": direction, "ic": ic_mean, "ic_ir": ic_ir}
        except Exception:
            quantile_returns[f] = {}
            suggest_bell_per[f] = False
            quality_per_factor[f] = {"spread": 0.0, "direction": "unknown", "ic": 0.0, "ic_ir": 0.0}
    # ── 模式参数 ──
    factor_mode = (getattr(args, "factor_mode", "multi") or "multi").strip().lower()
    if factor_mode not in ("single", "dual", "multi"):
        factor_mode = "multi"
    max_factors = {"single": 1, "dual": 2, "multi": 5}.get(factor_mode, 5)
    rebalance_freq = max(1, int(getattr(args, "rebalance_freq", 1) or 1))
    label_horizon = max(1, int(getattr(args, "label_horizon", 5)))
    top_n = max(1, min(int(getattr(args, "top_n", 10) or 10), 50))
    n_trials = getattr(args, "n_trials", None)
    if n_trials is None or (isinstance(n_trials, (int, float)) and int(n_trials) <= 0):
        n_trials = {"single": 6, "dual": 10, "multi": 8}.get(factor_mode, 6)
    n_trials = max(1, min(int(n_trials), 150))

    # ── 1. 编排 Agent：规划因子组合方案 ──
    if progress_callback:
        progress_callback(0, n_trials + 2, "编排 Agent：规划因子组合方案…")
    planned_combos = _run_orchestration_agent(
        factor_names=factor_names,
        quality_per_factor=quality_per_factor,
        corr_df=corr_df,
        factor_mode=factor_mode,
        max_factors=max_factors,
        n_trials=n_trials,
        event_sink=event_sink,
    )
    print(f"[search] 编排 Agent 生成 {len(planned_combos)} 个组合方案", flush=True)

    # ── 2. 评价 Agent：逐组评估 ──
    trial_results: List[Dict[str, Any]] = []
    for trial_idx_outer, combo_info in enumerate(planned_combos):
        if abort_check and abort_check():
            break
        combo = combo_info["factors"]
        reason = combo_info.get("reason", "")
        if progress_callback:
            progress_callback(
                trial_idx_outer + 1, n_trials + 2,
                f"评价 Agent：第 {trial_idx_outer + 1}/{len(planned_combos)} 组 {combo}…",
            )
        print(f"[search] Trial {trial_idx_outer + 1}/{len(planned_combos)}: {combo} — {reason}", flush=True)
        trial_r = _run_evaluation_agent(
            factor_df=factor_df,
            combo=combo,
            suggest_bell_per=suggest_bell_per,
            quality_per_factor=quality_per_factor,
            benchmark_returns=benchmark_returns,
            rebalance_freq=rebalance_freq,
            label_horizon=label_horizon,
            top_n=top_n,
            trial_idx=trial_idx_outer,
        )
        if trial_r:
            trial_results.append(trial_r)
            if event_sink:
                bt = (trial_r.get("train_result") or {}).get("backtest_stats") or {}
                summary = {
                    "trial": trial_idx_outer + 1,
                    "combo": trial_r.get("combo", []),
                    "final_names": trial_r.get("final_names", []),
                    "sharpe_annual": bt.get("sharpe_annual"),
                    "val_rank_ic": (trial_r.get("train_result") or {}).get("val_rank_ic"),
                    "val_spread": (trial_r.get("train_result") or {}).get("val_spread"),
                }
                try:
                    event_sink("evaluation", "trial_result", json.dumps(summary, ensure_ascii=False))
                except Exception:
                    event_sink("evaluation", "trial_result", str(summary))
    if not trial_results:
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": bool(abort_check and abort_check()),
            "workflow": True,
            "orchestrated": True,
            "agent_driven": True,
            "error": "Agent 编排未得到有效权重结果",
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": 0,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    # ── 3. 审查 Agent：综合评估所有 trial，选出最优组合并生成报告 ──
    if progress_callback:
        progress_callback(len(planned_combos) + 1, n_trials + 2, "审查 Agent：综合评估所有结果…")
    reviewer_out = _run_reviewer_agent(
        trial_results=trial_results,
        quality_per_factor=quality_per_factor,
        corr_df=corr_df,
        factor_mode=factor_mode,
        rebalance_freq=rebalance_freq,
        label_horizon=label_horizon,
        event_sink=event_sink,
    )
    best_idx = reviewer_out.get("selected_trial_idx", 0)
    best_trial = trial_results[best_idx]
    train_result = best_trial["train_result"]
    if not train_result:
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": bool(abort_check and abort_check()),
            "workflow": True,
            "orchestrated": True,
            "agent_driven": True,
            "error": "审查 Agent 选出的 trial 无有效权重",
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": len(trial_results),
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
            "reviewer": reviewer_out,
        }
    # 按指标强制修正结论：负 Alpha/差夏普/大回撤 → 不推荐或谨慎推荐
    reviewer_out = _apply_reviewer_verdict_rules(reviewer_out, train_result)
    ab = train_result.get("alpha_beta") or {}
    backtest_stats_out = train_result.get("backtest_stats")
    # 注入选中 trial 的实际指标，供前端在审查报告中展示「与挖掘报告一致」的数值，避免 LLM 评审意见中引用错 trial 导致不一致
    _bs = train_result.get("backtest_stats") or {}
    _ab = train_result.get("alpha_beta") or {}
    reviewer_out["selected_trial_metrics"] = {
        "alpha": _ab.get("alpha"),
        "beta": _ab.get("beta"),
        "annualized_alpha": _ab.get("annualized_alpha"),
        "r_squared": _ab.get("r_squared"),
        "total_return": _bs.get("total_return"),
        "sharpe_annual": _bs.get("sharpe_annual"),
        "max_drawdown": _bs.get("max_drawdown"),
    }
    # feature_names 优先；若仍为空则从 best_trial combo 取
    selected = (
        train_result.get("feature_names")
        or train_result.get("best_factor_combo")
        or (best_trial.get("combo") if best_trial else [])
        or []
    )
    flat = (train_result.get("learned_weights") or {}).get("flat") or {}
    # 策略/轮仓逻辑由审查 Agent 生成（兜底：使用 get_report_generation_prompt）
    strategy_logic = reviewer_out.get("strategy_logic") or ""
    rotation_logic = reviewer_out.get("rotation_logic") or ""
    if not strategy_logic:
        try:
            report_prompt = get_report_generation_prompt(
                selected, flat,
                float(ab.get("alpha", 0) or 0), float(ab.get("beta", 0) or 0),
                train_result.get("metrics") or {}, rebalance_freq=rebalance_freq,
            )
            resp = _get_llm().invoke(report_prompt)
            text = getattr(resp, "content", None) or str(resp)
            parts = [p.strip() for p in text.split("\n") if p.strip()]
            if len(parts) >= 2:
                strategy_logic, rotation_logic = parts[0], parts[1]
            elif len(parts) == 1:
                strategy_logic = parts[0]
        except Exception:
            strategy_logic = f"选股依据：{selected}，线性权重（可负）{flat}。Alpha={ab.get('alpha')}, Beta={ab.get('beta')}。"
            rotation_logic = f"按综合得分排序取 TopN，每 {rebalance_freq} 日调仓，等权再平衡。"
    # 构建每个因子的详细信息：始终基于 factor_df 为选中因子计算质量指标，保证前端有 spread/ic/direction
    factor_quality_out: Dict[str, Any] = {}
    for fname in selected:
        base = fname[:-5] if fname.endswith("_bell") else fname
        q: Dict[str, Any] = {}
        if factor_df is not None and base in factor_df.columns:
            try:
                qr = compute_factor_quantile_returns(factor_df, base, q=5)
                vals = [float(qr.get(str(i), 0.0)) for i in range(len(qr))] if qr else []
                spread = round(max(vals) - min(vals), 6) if vals else 0.0
                if len(vals) >= 2:
                    inc = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
                    dec = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
                    direction = "up" if inc else ("down" if dec else ("bell" if suggest_bell_per.get(base, False) else "mixed"))
                else:
                    direction = "unknown"
                ic_list: List[float] = []
                for _, grp in factor_df.groupby("date"):
                    if base in grp.columns and "y" in grp.columns and len(grp) >= 5:
                        valid = grp[[base, "y"]].dropna()
                        if len(valid) >= 5 and float(valid[base].std()) > 1e-10 and float(valid["y"].std()) > 1e-10:
                            ic_val = float(valid[base].corr(valid["y"], method="spearman"))
                            if not np.isnan(ic_val):
                                ic_list.append(ic_val)
                ic_mean = round(float(np.mean(ic_list)) if ic_list else 0.0, 5)
                ic_ir = round(float(np.mean(ic_list) / np.std(ic_list)) if len(ic_list) > 1 and float(np.std(ic_list)) > 1e-10 else 0.0, 4)
                q = {"spread": spread, "direction": direction, "ic": ic_mean, "ic_ir": ic_ir}
            except Exception:
                q = quality_per_factor.get(base) or quality_per_factor.get(fname) or {}
        else:
            q = quality_per_factor.get(base) or quality_per_factor.get(fname) or {}
        factor_quality_out[fname] = {
            "spread": q.get("spread", 0.0),
            "ic": q.get("ic", 0.0),
            "ic_ir": q.get("ic_ir", 0.0),
            "direction": q.get("direction", "unknown"),
            "is_bell": fname.endswith("_bell"),
            "bell_formula": "(x − 截面均值)²" if fname.endswith("_bell") else None,
        }
    best = {
        "factor_set": "workflow",
        "label_horizon": label_horizon,
        "rebalance_freq": rebalance_freq,
        "top_n": top_n,
        "feature_select_count": len(selected),
        "val_ratio": 0.2,
        "objective": float((train_result.get("metrics") or {}).get("val_rank_ic", 0.0)),
        "metrics": train_result.get("metrics", {}),
        "best_factor_combo": selected,
        "backtest_stats": backtest_stats_out,
        "learned_weights": flat,
        "weight_type": "linear",
        "factor_quality": factor_quality_out,
        "bell_transforms": best_trial.get("bell_transforms", []),
    }
    # 所有试验结果列表（按 sharpe 降序）
    all_tops = []
    for r in sorted(trial_results, key=lambda x: float((x["train_result"].get("backtest_stats") or {}).get("sharpe_annual", 0.0) or 0.0), reverse=True):
        tr = r["train_result"]
        all_tops.append({
            "factor_set": "workflow",
            "best_factor_combo": tr.get("feature_names") or r["combo"],
            "learned_weights": (tr.get("learned_weights") or {}).get("flat"),
            "backtest_stats": tr.get("backtest_stats"),
            "metrics": tr.get("metrics", {}),
        })
    output = {
        "created_at": datetime.now().isoformat(),
        "stopped": False,
        "workflow": True,
        "orchestrated": True,
        "agent_driven": True,
        "universe_source": args.universe_source,
        "universe_size": len(codes),
        "search_space": {"days": initial["days"], "label_horizon": label_horizon, "rebalance_freq": rebalance_freq, "top_n": top_n},
        "best": best,
        "top": all_tops,
        "all_results_count": len(trial_results),
        "alpha": ab.get("alpha"),
        "beta": ab.get("beta"),
        "annualized_alpha": ab.get("annualized_alpha"),
        "r_squared": ab.get("r_squared"),
        "backtest_stats": backtest_stats_out,
        "strategy_logic": strategy_logic,
        "rotation_logic": rotation_logic,
        "reviewer": reviewer_out,
    }
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join(
        "outputs", f"factor_search_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"[search] agent-driven done. combo={best.get('best_factor_combo')} weights={best.get('learned_weights')}", flush=True)
    return output


def run_factor_backtest_only(
    params: dict,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, Any]:
    """仅回测：根据给定股票池、因子组合与权重，运行回测并返回 alpha/beta/夏普等。progress_callback(phase, pct, message)。"""
    # 规范化列表参数（API 可能传 list 或 str）
    industry_list_raw = params.get("industry_list")
    if isinstance(industry_list_raw, list):
        industry_list_str = ",".join(str(x).strip() for x in industry_list_raw if x)
    else:
        industry_list_str = str(industry_list_raw or "")
    stocks_raw = params.get("stocks")
    if isinstance(stocks_raw, list):
        stocks_str = ",".join(str(x).strip() for x in stocks_raw if x)
    else:
        stocks_str = str(stocks_raw or "")

    universe_source = str(params.get("universe_source", "manual")).strip() or "manual"
    universe_index = str(params.get("universe_index", "")).strip()
    if universe_source == "index" and not universe_index:
        universe_index = "000300"

    args = SimpleNamespace(
        universe_source=universe_source,
        universe_index=universe_index,
        industry_list=industry_list_str,
        leaders_per_industry=max(1, int(params.get("leaders_per_industry", 1))),
        stocks=stocks_str,
        max_stocks=int(params.get("max_stocks", 60)),
        days=int(params.get("days", 252)),
        exclude_kechuang=bool(params.get("exclude_kechuang", False)),
        exclude_small_cap=bool(params.get("exclude_small_cap", False)),
        small_cap_max_billion=float(
            params.get("small_cap_max_billion", params.get("small_cap_threshold_billion", 30))
        ),
        cap_scope=str(params.get("cap_scope", "none") or "none"),
        benchmark_code=str(params.get("benchmark_code", "510300") or "510300"),
    )
    def _progress(phase: str, pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(phase, pct, msg)

    codes = _resolve_universe(args)
    # 指数模式下若成分股获取失败，尝试用 000300 / 000016 再解析
    if len(codes) < 8 and universe_source == "index":
        print(f"[backtest] 指数 {universe_index} 成分股获取结果: {len(codes)} 只", flush=True)
        if universe_index != "000300":
            args.universe_index = "000300"
            codes = _resolve_universe(args)
            if len(codes) < 8:
                print(f"[backtest] 兜底 000300 成分股: {len(codes)} 只", flush=True)
        if len(codes) < 8 and universe_index != "000016":
            args.universe_index = "000016"
            codes = _resolve_universe(args)
            if len(codes) < 8:
                print(f"[backtest] 兜底 000016 成分股: {len(codes)} 只", flush=True)
    _progress("load", 5.0, f"股票池 {len(codes)} 只")
    if len(codes) < 8:
        print(f"[backtest] 有效股票池过少: universe_source={universe_source} universe_index={universe_index} codes={len(codes)}", flush=True)
        hint = ""
        if universe_source == "index":
            hint = "（指数池：成分股获取失败，请确认 1) 指数代码正确如 000300/000016 2) 网络可访问 akshare）"
        elif universe_source == "industry":
            hint = "（请至少选择一个行业）"
        else:
            hint = "（手动池请至少输入 8 只股票代码，用逗号分隔）"
        return {
            "error": f"有效股票池过少: {len(codes)}，请至少提供 8 只股票{hint}",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    factor_combo = params.get("factor_combo") or params.get("best_factor_combo") or []
    weights_dict = params.get("weights") or {}
    if not factor_combo:
        return {
            "error": "缺少 factor_combo（因子名列表）",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    days = int(params.get("days", 252))
    label_horizon = max(1, int(params.get("label_horizon", 5)))
    rebalance_freq = max(1, int(params.get("rebalance_freq", 1) or 1))
    benchmark_code = str(params.get("benchmark_code", "510300") or "510300")
    top_n = max(1, min(int(params.get("top_n", 10)), 50))
    try:
        from tools.stock_data import get_stock_data
    except ImportError:
        return {
            "error": "tools.stock_data 不可用",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    data_dict = {}
    # 回测 3 年(约 756 交易日)时需足够历史；预留 150 日给因子计算窗口与标签
    load_days = max(400, days + 150)
    n_codes = len(codes)
    for idx, code in enumerate(codes):
        try:
            raw = get_stock_data(code, load_days, use_cache=True)
            if raw and len(raw) >= 60:
                df = pd.DataFrame(raw)
                if {"date", "close", "high", "low", "volume"}.issubset(df.columns):
                    data_dict[code] = df.dropna().reset_index(drop=True)
        except Exception:
            continue
        if n_codes > 0 and (idx + 1) % max(1, n_codes // 10) == 0:
            _progress("load", 5.0 + 30.0 * (idx + 1) / n_codes, f"加载行情 {idx + 1}/{n_codes}")
    _progress("load", 35.0, f"已加载 {len(data_dict)} 只行情")
    if len(data_dict) < 8:
        return {
            "error": f"加载行情不足: {len(data_dict)} 只",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    extra_by_code_date = {}
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
    except Exception as _e:
        pass
    _progress("load", 38.0, "构建因子样本…")
    factor_df_raw = None
    if dates:
        start_d = min(dates)
        end_d = max(dates)
        try:
            from db.factor_storage import load_factor_df
            from modules.factor_mining.factor_registry import get_all_factor_ids
            base_names = [f[:-5] if f.endswith("_bell") else f for f in factor_combo]
            requested = list(dict.fromkeys(base_names))
            if not requested:
                requested = list(get_all_factor_ids())[:20]
            loaded = load_factor_df(
                symbols=list(data_dict.keys()),
                start_date=start_d,
                end_date=end_d,
                factor_names=requested,
                label_horizon=label_horizon,
            )
            if loaded is not None and len(loaded) >= 80 and all(f in loaded.columns for f in base_names):
                factor_df_raw = loaded
        except Exception as _e:
            pass
    if factor_df_raw is None or factor_df_raw.empty:
        factor_df_raw = build_training_samples(
            data_dict=data_dict,
            days=days,
            label_horizon=label_horizon,
            max_window=120,
            extra_factors_by_code_date=extra_by_code_date if extra_by_code_date else None,
        )
        if factor_df_raw is not None and not factor_df_raw.empty:
            try:
                from db.factor_storage import save_factor_df
                cols = [c for c in factor_df_raw.columns if c not in ("date", "stock_code", "y")]
                if cols:
                    save_factor_df(factor_df_raw, factor_columns=cols, label_horizon=label_horizon)
            except Exception:
                pass
    _progress("load", 40.0, "因子样本就绪")
    if factor_df_raw is None or factor_df_raw.empty:
        return {
            "error": "构建训练样本为空",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    # 对含 _bell 后缀的因子，先对其基础因子做钟形变换
    bell_base_factors = [f[:-5] for f in factor_combo if f.endswith("_bell") and f[:-5] in factor_df_raw.columns]
    factor_df = apply_bell_transforms(factor_df_raw, bell_base_factors, date_col="date") if bell_base_factors else factor_df_raw
    missing = [f for f in factor_combo if f not in factor_df.columns]
    if missing:
        return {
            "error": f"样本缺少因子列: {missing}（bell 变换后仍找不到，请检查因子名）",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
        }
    # 仅使用最近 days 个交易日，使回测与图表与用户选择的回测天数一致（如 3 年=756 日）
    unique_dates = factor_df["date"].drop_duplicates().sort_values()
    if len(unique_dates) > days:
        last_dates = set(unique_dates.iloc[-int(days) :].tolist())
        factor_df = factor_df[factor_df["date"].isin(last_dates)].copy()
    weights_arr = np.array(
        [float(weights_dict.get(f, 0.0)) for f in factor_combo],
        dtype=float,
    )
    if weights_arr.sum() <= 0:
        weights_arr = np.ones(len(factor_combo)) / len(factor_combo)
    position_weight_method = (params.get("position_weight_method") or "equal").strip().lower()
    if position_weight_method not in ("equal", "score_weighted", "kelly"):
        position_weight_method = "equal"

    def _bt_progress(cur: int, total: int, msg: str) -> None:
        if total > 0 and progress_callback:
            pct = 40.0 + 50.0 * cur / total
            progress_callback("backtest", pct, msg)

    strat_ser, rb_details = run_backtest_detailed(
        factor_df, factor_combo, weights_arr,
        top_n=top_n, rebalance_freq=rebalance_freq,
        position_weight_method=position_weight_method,
        progress_callback=_bt_progress,
    )
    if strat_ser.empty:
        return {
            "error": "回测收益序列为空",
            "alpha": None,
            "beta": None,
            "annualized_alpha": None,
            "r_squared": None,
            "backtest_stats": None,
            "rebalance_details": [],
            "chart_base64": None,
        }
    bench_daily = pd.Series(dtype=float)
    try:
        raw_b = get_stock_data(benchmark_code, load_days, use_cache=True)
        if raw_b and len(raw_b) >= 10:
            bdf = pd.DataFrame(raw_b)
            if "date" in bdf.columns and "close" in bdf.columns:
                bdf = bdf.sort_values("date")
                bdf["ret"] = bdf["close"].astype(float).pct_change()
                bench_daily = bdf.set_index("date")["ret"].dropna()
    except Exception:
        pass
    _progress("post", 90.0, "计算 Alpha/Beta…")
    ab = _compute_alpha_beta_robust(strat_ser, bench_daily, label_horizon)
    periods_per_year = max(1, int(252 / rebalance_freq))
    stats = backtest_stats(strat_ser, periods_per_year=periods_per_year)
    stats_serializable = {k: float(v) if hasattr(v, "item") else v for k, v in stats.items()}
    _progress("post", 95.0, "生成图表…")
    chart_b64 = generate_backtest_chart(
        strat_ser, bench_daily,
        title=f"回测：{' + '.join(factor_combo)}",
    )
    weights_out = {f: float(weights_dict.get(f, 0.0)) for f in factor_combo}
    out = {
        "error": None,
        "factor_combo": factor_combo,
        "weights": weights_out,
        "alpha": ab.get("alpha"),
        "beta": ab.get("beta"),
        "annualized_alpha": ab.get("annualized_alpha"),
        "r_squared": ab.get("r_squared"),
        "backtest_stats": stats_serializable,
        "rebalance_details": rb_details,
        "chart_base64": chart_b64,
    }
    # 稳健性检查：同一数据下多组 (TopN, 调仓周期) 回测，评估参数敏感性
    if params.get("robustness_check"):
        param_sets = [(top_n, rebalance_freq), (5, 1), (5, 8), (10, 5), (15, 1)]
        seen = set()
        unique_sets = []
        for tn, rf in param_sets:
            if (tn, rf) in seen:
                continue
            seen.add((tn, rf))
            unique_sets.append((tn, rf))
        robustness_results = []
        for idx, (tn, rf) in enumerate(unique_sets):
            if progress_callback and len(unique_sets) > 1:
                progress_callback("post", 95.0 + 4.0 * (idx + 1) / len(unique_sets), f"稳健性检查 {idx + 1}/{len(unique_sets)}: TopN={tn} 调仓={rf}日")
            try:
                rs, _ = run_backtest_detailed(
                    factor_df, factor_combo, weights_arr,
                    top_n=tn, rebalance_freq=rf,
                    position_weight_method=position_weight_method,
                )
                if rs.empty:
                    robustness_results.append({"top_n": tn, "rebalance_freq": rf, "total_return": None, "sharpe_annual": None, "max_drawdown": None})
                    continue
                ppy = max(1, int(252 / rf))
                bt = backtest_stats(rs, periods_per_year=ppy)
                robustness_results.append({
                    "top_n": tn,
                    "rebalance_freq": rf,
                    "total_return": float(bt.get("total_return", 0)),
                    "sharpe_annual": float(bt.get("sharpe_annual", 0)),
                    "max_drawdown": float(bt.get("max_drawdown", 0)),
                })
            except Exception as _e:
                robustness_results.append({"top_n": tn, "rebalance_freq": rf, "total_return": None, "sharpe_annual": None, "max_drawdown": None})
        out["robustness_results"] = robustness_results
    _progress("post", 100.0, "完成")
    return out


def _run_orchestrated_search(
    args,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """编排多任务：由 AI Agent 生成任务列表，多组 mode/label_horizon/rebalance_freq 依次执行，汇总后按得分取 Top-K。"""
    codes = _resolve_universe(args)
    if len(codes) < 8:
        raise ValueError(
            f"有效股票池过少: {len(codes)}，请至少提供 8 只股票或选择指数/行业"
        )
    # 由 AI Agent 生成任务列表，失败时回退到默认
    if progress_callback:
        progress_callback(0, 1, "编排 Agent: 生成任务列表…")
    tasks = _get_orchestration_tasks_from_agent(args, universe_size=len(codes))
    if progress_callback:
        progress_callback(0, len(tasks), f"已生成 {len(tasks)} 个编排任务，开始执行…")
    max_combos = max(1, min(150, int(getattr(args, "max_combos", 15) or 15)))
    benchmark_code = getattr(args, "benchmark_code", "510300") or "510300"
    days = int(args.days)
    set_factor_mining_progress(progress_callback=progress_callback, abort_check=abort_check)
    results: List[tuple] = []  # (score, best_item, ab, backtest_stats, strategy_logic, rotation_logic, err)
    for idx, (mode, label_horizon, rebalance_freq) in enumerate(tasks):
        if abort_check and abort_check():
            break
        if progress_callback:
            progress_callback(idx, len(tasks), f"编排任务 {idx + 1}/{len(tasks)}: mode={mode} horizon={label_horizon} rebal={rebalance_freq}")
        initial = {
            "universe_codes": codes,
            "days": days,
            "label_horizon": label_horizon,
            "rebalance_freq": rebalance_freq,
            "max_combos": max_combos,
            "mode": mode,
            "benchmark_code": benchmark_code,
        }
        try:
            final = factor_mining_graph.invoke(initial)
        except Exception as e:
            if progress_callback:
                progress_callback(idx + 1, len(tasks), f"任务 {idx + 1} 异常: {e}")
            continue
        err = final.get("error")
        train_result = final.get("train_result") or {}
        if err and not train_result:
            continue
        ab = final.get("alpha_beta") or {}
        backtest_stats = final.get("backtest_stats")
        if isinstance(backtest_stats, dict):
            backtest_stats = {k: float(v) if hasattr(v, "item") else v for k, v in backtest_stats.items()}
        ann_alpha = float(ab.get("annualized_alpha", 0) or 0)
        val_rank_ic = float((train_result.get("metrics") or {}).get("val_rank_ic", 0) or 0)
        score = ann_alpha * 0.01 + val_rank_ic
        best_item = {
            "factor_set": "workflow",
            "label_horizon": label_horizon,
            "rebalance_freq": rebalance_freq,
            "mode": mode,
            "feature_select_count": len(final.get("selected_factors") or []),
            "val_ratio": 0.2,
            "objective": score,
            "metrics": train_result.get("metrics", {}),
            "oos_stability": train_result.get("oos_stability", {}),
            "best_factor_combo": final.get("selected_factors") or train_result.get("best_factor_combo", []),
            "backtest_stats": backtest_stats,
        }
        results.append((score, best_item, ab, backtest_stats, final.get("strategy_logic"), final.get("rotation_logic"), err))
    if not results:
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": bool(abort_check and abort_check()),
            "workflow": True,
            "orchestrated": True,
            "error": "编排多任务均未得到有效结果",
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "top": [],
            "all_results_count": 0,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    results.sort(key=lambda x: x[0], reverse=True)
    top_k = results[:10]
    best_item = top_k[0][1]
    ab = top_k[0][2]
    backtest_stats = top_k[0][3]
    strategy_logic = top_k[0][4]
    rotation_logic = top_k[0][5]
    top_list = [r[1] for r in top_k]
    output = {
        "created_at": datetime.now().isoformat(),
        "stopped": False,
        "workflow": True,
        "orchestrated": True,
        "universe_source": args.universe_source,
        "universe_size": len(codes),
        "search_space": {
            "tasks": len(tasks),
            "days": days,
            "max_combos": max_combos,
        },
        "best": best_item,
        "top": top_list,
        "all_results_count": len(results),
        "alpha": ab.get("alpha"),
        "beta": ab.get("beta"),
        "annualized_alpha": ab.get("annualized_alpha"),
        "r_squared": ab.get("r_squared"),
        "backtest_stats": backtest_stats,
        "strategy_logic": strategy_logic,
        "rotation_logic": rotation_logic,
    }
    if top_k[0][6]:
        output["error"] = top_k[0][6] if isinstance(top_k[0][6], str) else str(top_k[0][6])
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join(
        "outputs", f"factor_search_orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"[search] orchestrated done. best combo={best_item.get('best_factor_combo')} from {len(results)} tasks")
    return output


def run_search_workflow(
    args,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """使用 LangGraph 工作流：LLM 选因子（单/双/多）、钟形变换、训练、Alpha/Beta、策略与轮仓报告。"""
    codes = _resolve_universe(args)
    if len(codes) < 8:
        raise ValueError(
            f"有效股票池过少: {len(codes)}，请至少提供 8 只股票或选择指数/行业"
        )
    mode = getattr(args, "factor_mode", "multi") or "multi"
    if mode not in ("single", "dual", "multi"):
        mode = "multi"
    set_factor_mining_progress(progress_callback=progress_callback, abort_check=abort_check)
    if progress_callback:
        progress_callback(0, 1, "workflow: 启动因子挖掘")
    max_combos = max(1, min(150, int(getattr(args, "max_combos", 15) or 15)))
    initial = {
        "universe_codes": codes,
        "days": int(args.days),
        "label_horizon": max(1, int(getattr(args, "label_horizon", 5))),
        "rebalance_freq": max(1, int(getattr(args, "rebalance_freq", 1) or 1)),
        "max_combos": max_combos,
        "mode": mode,
        "benchmark_code": getattr(args, "benchmark_code", "510300") or "510300",
    }
    try:
        final = factor_mining_graph.invoke(initial)
    except Exception as e:
        return {
            "created_at": datetime.now().isoformat(),
            "stopped": False,
            "workflow": True,
            "error": str(e),
            "universe_source": args.universe_source,
            "universe_size": len(codes),
            "best": None,
            "alpha": None,
            "beta": None,
            "strategy_logic": None,
            "rotation_logic": None,
        }
    err = final.get("error")
    train_result = final.get("train_result") or {}
    backtest_stats = final.get("backtest_stats")
    if isinstance(backtest_stats, dict):
        backtest_stats = {k: float(v) if hasattr(v, "item") else v for k, v in backtest_stats.items()}
    learned = (train_result.get("learned_weights") or {}).get("flat") or {}
    best = {
        "factor_set": "workflow",
        "label_horizon": initial["label_horizon"],
        "feature_select_count": len(final.get("selected_factors") or []),
        "val_ratio": 0.2,
        "objective": float(
            (train_result.get("metrics") or {}).get("val_rank_ic", 0.0)
        ),
        "metrics": train_result.get("metrics", {}),
        "oos_stability": train_result.get("oos_stability", {}),
        "best_factor_combo": final.get("selected_factors")
        or train_result.get("best_factor_combo", []),
        "backtest_stats": backtest_stats,
        "learned_weights": {k: float(v) for k, v in learned.items()},
    }
    ab = final.get("alpha_beta") or {}
    output = {
        "created_at": datetime.now().isoformat(),
        "stopped": False,
        "workflow": True,
        "universe_source": args.universe_source,
        "universe_size": len(codes),
        "loaded_size": len(final.get("data_dict") or {}),
        "search_space": {
            "mode": mode,
            "days": initial["days"],
            "label_horizon": initial["label_horizon"],
            "rebalance_freq": initial.get("rebalance_freq", 1),
        },
        "best": best,
        "top": [best],
        "all_results_count": 1,
        "alpha": ab.get("alpha"),
        "beta": ab.get("beta"),
        "annualized_alpha": ab.get("annualized_alpha"),
        "r_squared": ab.get("r_squared"),
        "backtest_stats": backtest_stats,
        "strategy_logic": final.get("strategy_logic"),
        "rotation_logic": final.get("rotation_logic"),
    }
    if err is not None:
        output["error"] = err if isinstance(err, str) else str(err)
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join(
        "outputs", f"factor_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"[search] workflow done. best combo={best.get('best_factor_combo')}")
    print(f"[search] alpha={output.get('alpha')} beta={output.get('beta')}")
    print(f"[search] output={out_file}")
    return output


def run_search_from_params(
    params: dict,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
    event_sink: Optional[Callable[[str, str, str], None]] = None,
) -> dict:
    """供 API 调用：仅支持 LLM workflow，传入参数字典，返回统一 output 结构。event_sink(role, phase, content) 用于推送 Agent 输入/输出供前端监视。"""
    factor_mode = str(params.get("factor_mode", "multi")).lower() or "multi"
    stocks_raw = params.get("stocks", "")
    if isinstance(stocks_raw, list):
        stocks_str = ",".join(str(x).strip() for x in stocks_raw if str(x).strip())
    else:
        stocks_str = str(stocks_raw or "")
    args = SimpleNamespace(
        universe_source=str(params.get("universe_source", "manual") or "manual"),
        universe_index=(params.get("universe_index") or "").strip() or "",
        industry_list=str(params.get("industry_list", "")),
        leaders_per_industry=max(1, int(params.get("leaders_per_industry", 1))),
        stocks=stocks_str,
        max_stocks=int(params.get("max_stocks", 60)),
        days=int(params.get("days", 252)),
        exclude_kechuang=bool(params.get("exclude_kechuang", False)),
        exclude_small_cap=bool(params.get("exclude_small_cap", False)),
        small_cap_max_billion=float(
            params.get("small_cap_max_billion", params.get("small_cap_threshold_billion", 30))
        ),
        cap_scope=str(params.get("cap_scope", "none") or "none"),
        factor_mode=factor_mode,
        label_horizon=int(params.get("label_horizon", 5)),
        rebalance_freq=max(1, int(params.get("rebalance_freq", 1) or 1)),
        max_combos=max(1, int(params.get("max_combos", 15) or 15)),
        benchmark_code=str(params.get("benchmark_code", "510300") or "510300"),
        orchestrate_user_preference=str(params.get("orchestrate_user_preference", "") or "").strip(),
        n_trials=params.get("n_trials"),
    )
    orchestrate_tasks = bool(params.get("orchestrate_tasks", True))
    if orchestrate_tasks:
        return _run_agent_driven_search(
            args, progress_callback=progress_callback, abort_check=abort_check, event_sink=event_sink
        )
    return run_search_workflow(
        args,
        progress_callback=progress_callback,
        abort_check=abort_check,
    )


def main():
    parser = argparse.ArgumentParser(
        description="深度因子挖掘（LLM Workflow）"
    )
    parser.add_argument(
        "--universe-source",
        choices=["index", "industry", "manual"],
        default="index",
    )
    parser.add_argument("--universe-index", default="000300")
    parser.add_argument("--industry-list", default="")
    parser.add_argument("--leaders-per-industry", type=int, default=1)
    parser.add_argument("--stocks", default="")
    parser.add_argument("--max-stocks", type=int, default=60)
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--exclude-kechuang", action="store_true", help="排除科创板(688)")
    parser.add_argument(
        "--cap-scope",
        choices=["none", "only_small_cap", "exclude_small_cap"],
        default="none",
        help="市值范围: none | only_small_cap | exclude_small_cap",
    )
    parser.add_argument("--small-cap-max-billion", type=float, default=30)
    parser.add_argument(
        "--factor-mode",
        choices=["single", "dual", "multi"],
        default="multi",
        help="单/双/多因子模式",
    )
    parser.add_argument("--benchmark-code", default="510300", help="Alpha/Beta 基准")
    parser.add_argument("--label-horizon", type=int, default=5, help="预测步长")
    args = parser.parse_args()
    run_search_workflow(args)


if __name__ == "__main__":
    main()
