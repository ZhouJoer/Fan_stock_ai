"""
因子评估器：IC / 分层收益 / 稳定性。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    if len(scores) < 3 or len(returns) < 3:
        return 0.0
    s = scores.rank(pct=True)
    r = returns.rank(pct=True)
    corr = s.corr(r)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _longest_non_positive_streak(values: pd.Series) -> int:
    longest = 0
    current = 0
    for v in values.fillna(0.0).tolist():
        if float(v) <= 0:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def compute_factor_quantile_returns(
    factor_df: pd.DataFrame,
    factor_name: str,
    q: int = 5,
) -> Dict[str, float]:
    """
    按因子值分层（q 个分位），计算各层平均远期收益（y），再在时间上取平均。
    返回 { "0": mean_ret_q0, "1": mean_ret_q1, ... }，其中 0 为因子最低分位。
    用于判断因子与收益是否单调；若中间层收益高于两端，则适合钟形变换。
    """
    if factor_df is None or factor_df.empty or factor_name not in factor_df.columns or "y" not in factor_df.columns:
        return {}
    q = max(2, min(int(q), 10))
    out: Dict[str, float] = {}
    dates = factor_df["date"].drop_duplicates().tolist()
    if not dates:
        return {}
    bucket_returns: List[Dict[int, float]] = []
    for date in dates:
        grp = factor_df[factor_df["date"] == date][[factor_name, "y"]].dropna()
        if len(grp) < q * 2:
            continue
        try:
            grp = grp.copy()
            grp["_bucket"] = pd.qcut(grp[factor_name].rank(method="first"), q=q, labels=False, duplicates="drop")
            by_bucket = grp.groupby("_bucket")["y"].mean()
            bucket_returns.append(by_bucket.to_dict())
        except Exception:
            continue
    if not bucket_returns:
        return {}
    for b in range(q):
        vals = [r.get(b) for r in bucket_returns if b in r and pd.notna(r.get(b))]
        out[str(b)] = float(np.mean(vals)) if vals else 0.0
    return out


def suggest_bell_from_quantile_returns(quantile_returns: Dict[str, float]) -> bool:
    """
    根据分层收益判断是否建议钟形变换：若中间分位收益明显高于两端（倒 U 形），返回 True。
    """
    if not quantile_returns or len(quantile_returns) < 3:
        return False
    keys = sorted([int(k) for k in quantile_returns.keys()])
    vals = [float(quantile_returns[str(k)]) for k in keys]
    n = len(vals)
    mid = n // 2
    mid_ret = float(np.mean(vals[mid - 1 : mid + 2])) if n >= 3 else vals[mid]
    end_ret = (vals[0] + vals[-1]) / 2.0
    # 中间高于两端一定阈值则建议钟形
    return mid_ret > end_ret + 1e-6


@dataclass
class FactorEvaluator:
    daily_records: List[Dict[str, object]] = field(default_factory=list)
    selection_turnover: List[float] = field(default_factory=list)
    _last_selected: set[str] = field(default_factory=set)

    def add_record(
        self,
        date: str,
        scores: Dict[str, float],
        forward_returns: Dict[str, float],
        selected_codes: List[str],
    ) -> None:
        score_s = pd.Series(scores, dtype=float)
        ret_s = pd.Series(forward_returns, dtype=float)
        common = score_s.index.intersection(ret_s.index)
        if len(common) == 0:
            return

        score_s = score_s.loc[common]
        ret_s = ret_s.loc[common]
        ic = float(score_s.corr(ret_s)) if len(common) >= 3 else 0.0
        rank_ic = _rank_ic(score_s, ret_s)

        tmp = pd.DataFrame({"score": score_s, "ret": ret_s}).sort_values("score")
        q = min(5, max(2, len(tmp)))
        try:
            tmp["bucket"] = pd.qcut(tmp["score"], q=q, labels=False, duplicates="drop")
            bucket_ret = tmp.groupby("bucket")["ret"].mean().to_dict()
            top = bucket_ret.get(max(bucket_ret.keys()), 0.0) if bucket_ret else 0.0
            bottom = bucket_ret.get(min(bucket_ret.keys()), 0.0) if bucket_ret else 0.0
            spread = float(top - bottom)
        except Exception:
            bucket_ret = {}
            spread = 0.0

        selected = set(selected_codes or [])
        if self._last_selected:
            inter = len(self._last_selected.intersection(selected))
            turnover = 1.0 - inter / max(1, len(selected))
            self.selection_turnover.append(turnover)
        self._last_selected = selected

        self.daily_records.append(
            {
                "date": date,
                "ic": ic if not np.isnan(ic) else 0.0,
                "rank_ic": rank_ic,
                "top_bottom_spread": spread,
                "selected_count": len(selected),
                "bucket_return": {str(k): float(v) for k, v in bucket_ret.items()},
            }
        )

    def summary(self) -> Dict[str, object]:
        if not self.daily_records:
            return {
                "records": 0,
                "ic_mean": 0.0,
                "ic_ir": 0.0,
                "rank_ic_mean": 0.0,
                "spread_mean": 0.0,
                "selection_turnover_mean": 0.0,
                "ic_positive_ratio": 0.0,
                "rank_ic_positive_ratio": 0.0,
                "max_consecutive_fail_days": 0,
                "rolling_ic_ir_mean": 0.0,
                "rolling_ic_ir_latest": 0.0,
                "stability": "insufficient_data",
            }

        ic_series = pd.Series([r["ic"] for r in self.daily_records], dtype=float)
        rank_ic_series = pd.Series([r["rank_ic"] for r in self.daily_records], dtype=float)
        spread_series = pd.Series([r["top_bottom_spread"] for r in self.daily_records], dtype=float)
        turnover_series = pd.Series(self.selection_turnover, dtype=float) if self.selection_turnover else pd.Series([], dtype=float)

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std(ddof=0)) if len(ic_series) else 0.0
        ic_ir = float(ic_mean / ic_std) if ic_std > 1e-9 else 0.0
        rank_ic_mean = float(rank_ic_series.mean())
        spread_mean = float(spread_series.mean())
        turnover_mean = float(turnover_series.mean()) if len(turnover_series) else 0.0
        ic_positive_ratio = float((ic_series > 0).mean()) if len(ic_series) else 0.0
        rank_ic_positive_ratio = float((rank_ic_series > 0).mean()) if len(rank_ic_series) else 0.0
        max_consecutive_fail_days = _longest_non_positive_streak(ic_series)

        rolling_win = min(20, max(5, len(ic_series) // 3))
        rolling_ic_ir = pd.Series([], dtype=float)
        if len(ic_series) >= rolling_win:
            rolling_mean = ic_series.rolling(rolling_win).mean()
            rolling_std = ic_series.rolling(rolling_win).std(ddof=0).replace(0, np.nan)
            rolling_ic_ir = (rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan).dropna()
        rolling_ic_ir_mean = float(rolling_ic_ir.mean()) if len(rolling_ic_ir) else 0.0
        rolling_ic_ir_latest = float(rolling_ic_ir.iloc[-1]) if len(rolling_ic_ir) else 0.0

        stability = "weak"
        if abs(ic_ir) >= 0.5 and spread_mean > 0 and ic_positive_ratio >= 0.55:
            stability = "strong"
        elif abs(ic_ir) >= 0.2 and ic_positive_ratio >= 0.5:
            stability = "medium"

        return {
            "records": len(self.daily_records),
            "ic_mean": round(ic_mean, 6),
            "ic_ir": round(ic_ir, 6),
            "rank_ic_mean": round(rank_ic_mean, 6),
            "spread_mean": round(spread_mean, 6),
            "selection_turnover_mean": round(turnover_mean, 6),
            "ic_positive_ratio": round(ic_positive_ratio, 6),
            "rank_ic_positive_ratio": round(rank_ic_positive_ratio, 6),
            "max_consecutive_fail_days": int(max_consecutive_fail_days),
            "rolling_ic_ir_mean": round(rolling_ic_ir_mean, 6),
            "rolling_ic_ir_latest": round(rolling_ic_ir_latest, 6),
            "stability_basis": {
                "thresholds": {
                    "ic_ir_pass": 0.2,
                    "ic_positive_ratio_pass": 0.55,
                    "spread_mean_pass": 0.0,
                },
                "actual": {
                    "ic_ir": round(ic_ir, 6),
                    "ic_positive_ratio": round(ic_positive_ratio, 6),
                    "spread_mean": round(spread_mean, 6),
                    "selection_turnover_mean": round(turnover_mean, 6),
                },
            },
            "stability": stability,
            "daily": self.daily_records[-60:],
        }


def build_benchmark_horizon_returns(
    daily_bench: pd.Series,
    strategy_dates: pd.Index,
    horizon: int,
) -> pd.Series:
    """
    由日频基准收益构建与策略对齐的 horizon 期收益。
    对 strategy_dates 中每个日期 d，计算从 d 起连续 horizon 日的基准复利收益 (1+r1)*...*(1+rH)-1。
    仅当有足够后续数据时才计入，返回的 index 为 strategy_dates 的子集。
    """
    if daily_bench is None or daily_bench.empty or horizon < 1:
        return pd.Series(dtype=float)
    bench = daily_bench.sort_index()
    # 统一为字符串日期比较，避免 index 的 datetime/str 混用导致 TypeError 或匹配不到
    def _norm(x):
        if pd.isna(x):
            return ""
        if hasattr(x, "strftime"):
            return str(x)[:10]
        return str(x)[:10]
    dates = [_norm(x) for x in bench.index.tolist()]
    n = len(dates)
    out = {}
    for d in strategy_dates:
        d_str = _norm(d)
        if not d_str:
            continue
        try:
            pos = dates.index(d_str)
        except (ValueError, TypeError):
            try:
                pos = next(i for i, x in enumerate(dates) if x >= d_str)
            except StopIteration:
                continue
        if pos + horizon > n:
            continue
        chunk = bench.iloc[pos : pos + horizon]
        if len(chunk) < horizon:
            continue
        compound = (1.0 + chunk.astype(float)).prod() - 1.0
        out[d] = float(compound)
    return pd.Series(out).sort_index()


def _period_return_by_weight_method(
    top: pd.DataFrame,
    weight_method: str,
) -> float:
    """按选定的个股权重方式计算当期组合收益。weight_method: equal | score_weighted | kelly"""
    if top is None or top.empty or "y" not in top.columns:
        return 0.0
    y = top["y"].values.astype(float)
    n = len(y)
    if n == 0:
        return 0.0
    if weight_method == "equal":
        return float(np.mean(y))
    if weight_method == "score_weighted" and "_score" in top.columns:
        s = top["_score"].values.astype(float)
        s_min = float(np.min(s))
        s = s - s_min + 1e-8
        w = s / float(np.sum(s))
        return float(np.dot(w, y))
    if weight_method == "kelly":
        # 简化凯利：按预期收益（y）的正部分做权重并归一化，负收益权重为 0
        y_pos = np.maximum(y, 0.0)
        s = y_pos.sum()
        if s < 1e-12:
            return float(np.mean(y))
        w = y_pos / s
        return float(np.dot(w, y))
    return float(np.mean(y))


def run_backtest(
    factor_df: pd.DataFrame,
    feature_names: List[str],
    weights: np.ndarray,
    top_n: int = 10,
    rebalance_freq: int = 1,
    position_weight_method: str = "equal",
) -> pd.Series:
    """
    按策略调仓的回测。position_weight_method: equal=等权, score_weighted=按得分加权, kelly=凯利式权重。
    """
    if not feature_names or len(weights) != len(feature_names) or factor_df is None or factor_df.empty:
        return pd.Series(dtype=float)
    weight_method = (position_weight_method or "equal").strip().lower()
    if weight_method not in ("equal", "score_weighted", "kelly"):
        weight_method = "equal"
    freq = max(1, int(rebalance_freq))
    dates_sorted = factor_df["date"].drop_duplicates().sort_values().tolist()
    if not dates_sorted:
        return pd.Series(dtype=float)
    rebalance_dates = set(dates_sorted[i] for i in range(0, len(dates_sorted), freq))
    strategy_returns_list: List[tuple] = []
    for date, grp in factor_df.groupby("date"):
        if date not in rebalance_dates:
            continue
        g = grp.copy()
        for col in feature_names:
            if col not in g.columns:
                continue
            mean = g[col].mean()
            std_val = g[col].std()
            std_scalar = float(std_val) if np.isscalar(std_val) or getattr(std_val, "ndim", -1) == 0 else 0.0
            if std_scalar > 1e-12:
                g[col] = (g[col] - mean) / std_scalar
            else:
                g[col] = 0.0
        g["_score"] = g[feature_names].values @ weights
        top = g.nlargest(min(top_n, len(g)), "_score")
        if "y" in top.columns and len(top) > 0:
            period_ret = _period_return_by_weight_method(top, weight_method)
            strategy_returns_list.append((date, period_ret))
    if not strategy_returns_list:
        return pd.Series(dtype=float)
    return pd.Series({d: r for d, r in strategy_returns_list}).sort_index()


def backtest_stats(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """从收益序列计算总收益、年化夏普、最大回撤（序列为按期收益，非累计）。"""
    if returns is None or len(returns) < 2:
        return {"total_return": 0.0, "sharpe_annual": 0.0, "max_drawdown": 0.0}
    r = returns.astype(float).dropna()
    if len(r) < 2:
        return {"total_return": 0.0, "sharpe_annual": 0.0, "max_drawdown": 0.0}
    total_return = float((1.0 + r).prod() - 1.0)
    mean_r = float(r.mean())
    std_r = float(r.std())
    sharpe_annual = (mean_r / std_r * (periods_per_year ** 0.5)) if std_r > 1e-12 else 0.0
    cum = (1.0 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": round(total_return, 6),
        "sharpe_annual": round(sharpe_annual, 6),
        "max_drawdown": round(max_drawdown, 6),
    }


def compute_alpha_beta(
    strategy_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    risk_free: float = 0.0,
    annualize: bool = True,
) -> Dict[str, float]:
    """
    CAPM 回归（带截距 OLS）：R_strategy - Rf = alpha + beta * (R_bench - Rf)。
    beta = Cov(R_bench, R_strategy) / Var(R_bench)，alpha = E(R_strategy) - beta * E(R_bench)。
    返回 alpha（截距）、beta（斜率）、r_squared；若 annualize 则 alpha 年化（×252）。
    """
    if hasattr(strategy_returns, "values"):
        strategy_returns = strategy_returns.values
    if hasattr(benchmark_returns, "values"):
        benchmark_returns = benchmark_returns.values
    strategy_returns = np.asarray(strategy_returns, dtype=float).ravel()
    benchmark_returns = np.asarray(benchmark_returns, dtype=float).ravel()
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 3:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0}
    strategy_returns = strategy_returns[:n]
    benchmark_returns = benchmark_returns[:n]
    excess_strategy = strategy_returns - risk_free
    excess_bench = benchmark_returns - risk_free
    x_mean = float(np.mean(excess_bench))
    y_mean = float(np.mean(excess_strategy))
    cov_xy = float(np.mean((excess_bench - x_mean) * (excess_strategy - y_mean)))
    var_x = float(np.mean((excess_bench - x_mean) ** 2))
    if var_x < 1e-20:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "annualized_alpha": 0.0}
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * excess_bench
    ss_res = np.sum((excess_strategy - y_hat) ** 2)
    ss_tot = np.sum((excess_strategy - y_mean) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else 0.0
    annualized_alpha = float(alpha * 252) if annualize else alpha
    return {
        "alpha": round(alpha, 8),
        "beta": round(beta, 6),
        "r_squared": round(r_squared, 6),
        "annualized_alpha": round(annualized_alpha, 6),
    }


def run_backtest_detailed(
    factor_df: pd.DataFrame,
    feature_names: List[str],
    weights: np.ndarray,
    top_n: int = 10,
    rebalance_freq: int = 1,
    position_weight_method: str = "equal",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    与 run_backtest 逻辑相同，但额外返回每次调仓的明细列表。
    progress_callback(current, total, message) 可选，按调仓日进度回调。
    """
    if not feature_names or len(weights) != len(feature_names) or factor_df is None or factor_df.empty:
        return pd.Series(dtype=float), []
    weight_method = (position_weight_method or "equal").strip().lower()
    if weight_method not in ("equal", "score_weighted", "kelly"):
        weight_method = "equal"
    has_code = "stock_code" in factor_df.columns
    freq = max(1, int(rebalance_freq))
    dates_sorted = factor_df["date"].drop_duplicates().sort_values().tolist()
    if not dates_sorted:
        return pd.Series(dtype=float), []
    rebalance_dates = set(dates_sorted[i] for i in range(0, len(dates_sorted), freq))
    total_dates = len(rebalance_dates)
    returns_list: List[tuple] = []
    details: List[Dict[str, Any]] = []
    done = 0
    for date, grp in factor_df.groupby("date"):
        if date not in rebalance_dates:
            continue
        g = grp.copy()
        for col in feature_names:
            if col not in g.columns:
                continue
            mean = g[col].mean()
            std_val = float(g[col].std()) if np.isscalar(g[col].std()) else 0.0
            if std_val > 1e-12:
                g[col] = (g[col] - mean) / std_val
            else:
                g[col] = 0.0
        g["_score"] = g[feature_names].values @ weights
        top = g.nlargest(min(top_n, len(g)), "_score")
        if "y" not in top.columns or len(top) == 0:
            continue
        period_return = _period_return_by_weight_method(top, weight_method)
        returns_list.append((date, period_return))
        stocks = top["stock_code"].tolist() if has_code else []
        scores_raw = top["_score"].round(4).tolist()
        y_vals = top["y"].round(6).tolist()
        details.append({
            "date": str(date),
            "stocks": stocks,
            "scores": dict(zip(stocks, scores_raw)) if stocks else {},
            "period_return": round(period_return, 6),
            "stock_returns": dict(zip(stocks, y_vals)) if stocks else {},
        })
        done += 1
        if progress_callback and total_dates > 0:
            progress_callback(done, total_dates, f"回测 {done}/{total_dates} 日")
    if not returns_list:
        return pd.Series(dtype=float), []
    return pd.Series({d: r for d, r in returns_list}).sort_index(), details


def run_backtest_detailed_multi(
    factor_df: pd.DataFrame,
    strategies: List[Dict[str, Any]],
    top_n: int = 10,
    rebalance_freq: int = 1,
    position_weight_method: str = "equal",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    多策略批量回测：仅遍历一遍日期日历，同时计算多组因子组合。
    strategies: [{"id": "...", "feature_names": [...], "weights": np.ndarray}, ...]
    返回: {id: {"returns": pd.Series, "details": [...]}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if factor_df is None or factor_df.empty or not strategies:
        return out

    weight_method = (position_weight_method or "equal").strip().lower()
    if weight_method not in ("equal", "score_weighted", "kelly"):
        weight_method = "equal"
    has_code = "stock_code" in factor_df.columns
    freq = max(1, int(rebalance_freq))
    dates_sorted = factor_df["date"].drop_duplicates().sort_values().tolist()
    if not dates_sorted:
        return out
    rebalance_dates = set(dates_sorted[i] for i in range(0, len(dates_sorted), freq))
    total_dates = len(rebalance_dates)

    valid_strategies: List[Dict[str, Any]] = []
    all_cols: set[str] = set()
    for s in strategies:
        sid = str(s.get("id") or "").strip()
        fns = list(s.get("feature_names") or [])
        w = s.get("weights")
        if not sid or not fns or w is None or len(fns) != len(w):
            continue
        out[sid] = {"returns": pd.Series(dtype=float), "details": []}
        valid_strategies.append({"id": sid, "feature_names": fns, "weights": np.asarray(w, dtype=float)})
        all_cols.update(fns)
    if not valid_strategies:
        return out

    returns_map: Dict[str, List[tuple]] = {s["id"]: [] for s in valid_strategies}
    details_map: Dict[str, List[Dict[str, Any]]] = {s["id"]: [] for s in valid_strategies}

    done = 0
    for date, grp in factor_df.groupby("date"):
        if date not in rebalance_dates:
            continue
        g = grp.copy()
        # 同一日期上，公共因子列标准化仅做一次
        for col in all_cols:
            if col not in g.columns:
                continue
            mean = g[col].mean()
            std_val = g[col].std()
            std_scalar = float(std_val) if np.isscalar(std_val) or getattr(std_val, "ndim", -1) == 0 else 0.0
            if std_scalar > 1e-12:
                g[col] = (g[col] - mean) / std_scalar
            else:
                g[col] = 0.0

        for s in valid_strategies:
            sid = s["id"]
            fns = s["feature_names"]
            missing = [c for c in fns if c not in g.columns]
            if missing:
                continue
            scores = g[fns].values @ s["weights"]
            tmp = g.copy()
            tmp["_score"] = scores
            top = tmp.nlargest(min(top_n, len(tmp)), "_score")
            if "y" not in top.columns or len(top) == 0:
                continue
            period_return = _period_return_by_weight_method(top, weight_method)
            returns_map[sid].append((date, period_return))
            stocks = top["stock_code"].tolist() if has_code else []
            scores_raw = top["_score"].round(4).tolist()
            y_vals = top["y"].round(6).tolist()
            details_map[sid].append({
                "date": str(date),
                "stocks": stocks,
                "scores": dict(zip(stocks, scores_raw)) if stocks else {},
                "period_return": round(period_return, 6),
                "stock_returns": dict(zip(stocks, y_vals)) if stocks else {},
            })

        done += 1
        if progress_callback and total_dates > 0:
            progress_callback(done, total_dates, f"批量回测 {done}/{total_dates} 日")

    for s in valid_strategies:
        sid = s["id"]
        rs = returns_map.get(sid) or []
        if rs:
            out[sid] = {
                "returns": pd.Series({d: r for d, r in rs}).sort_index(),
                "details": details_map.get(sid) or [],
            }
        else:
            out[sid] = {"returns": pd.Series(dtype=float), "details": []}
    return out


def generate_backtest_chart(
    strat_returns: pd.Series,
    bench_daily: Optional[pd.Series] = None,
    title: str = "策略累计收益",
) -> Optional[str]:
    """
    生成累计收益 + 回撤图，返回 base64 PNG 字符串；失败返回 None。
    bench_daily 为日频基准收益序列（可选）。图表使用中文标签。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import bisect
        try:
            from utils.matplotlib_chinese import setup_chinese_font
            setup_chinese_font()
        except Exception:
            for font in ("Microsoft YaHei", "SimHei", "STHeiti", "Arial Unicode MS"):
                try:
                    matplotlib.rcParams["font.sans-serif"] = [font] + matplotlib.rcParams.get("font.sans-serif", [])
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    break
                except Exception:
                    pass

        r = strat_returns.dropna().astype(float)
        if len(r) < 2:
            return None

        cum = (1.0 + r).cumprod()
        xs = list(range(len(cum)))
        cum_pct = (cum.values - 1.0) * 100.0
        drawdown_pct = ((cum - cum.cummax()) / cum.cummax() * 100.0).values

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#1a1a2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.spines["bottom"].set_color("#444")
            ax.spines["top"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.spines["right"].set_color("#444")

        ax1.plot(xs, cum_pct, color="#4fc3f7", linewidth=1.5, label="策略")
        ax1.axhline(0, color="#666", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("累计收益率 (%)", color="#cccccc", fontsize=9)
        ax1.grid(True, alpha=0.2, color="#444")
        ax1.set_title(title, color="#eeeeee", fontsize=11, pad=8)

        # 基准叠加：在每个调仓日，从回测起始日累计所有日频收益，得到该日真实累计收益
        if bench_daily is not None and not bench_daily.empty:
            import bisect as _bisect
            bench = bench_daily.copy()
            bench.index = pd.Index([str(x)[:10] for x in bench.index])
            bench = bench.sort_index().astype(float)
            bdates = list(bench.index)
            bvals = bench.values
            start_str = str(r.index[0])[:10]
            start_pos = _bisect.bisect_left(bdates, start_str)
            bench_pct_list: List[float] = []
            for d in r.index:
                d_str = str(d)[:10]
                end_pos = _bisect.bisect_right(bdates, d_str)
                # 从回测第一个调仓日到当前调仓日，累乘所有日频基准收益
                chunk = bvals[start_pos:end_pos]
                if len(chunk) > 0:
                    cum_val = float((1.0 + chunk).prod() - 1.0) * 100.0
                else:
                    cum_val = 0.0
                bench_pct_list.append(cum_val)
            ax1.plot(xs, bench_pct_list, color="#ef9a9a", linewidth=1.0,
                     linestyle="--", alpha=0.8, label="基准")

        ax1.legend(fontsize=8, facecolor="#16213e", labelcolor="#cccccc",
                   loc="upper left", framealpha=0.6)

        ax2.fill_between(xs, drawdown_pct, 0, color="#ef5350", alpha=0.45)
        ax2.set_ylabel("回撤 (%)", color="#cccccc", fontsize=9)
        ax2.set_xlabel("调仓期", color="#cccccc", fontsize=9)
        ax2.grid(True, alpha=0.2, color="#444")

        # X 轴日期标签
        if len(r.index) > 0:
            step = max(1, len(r.index) // 8)
            ticks = list(range(0, len(r.index), step))
            labels = [str(r.index[i])[:10] for i in ticks]
            for ax in (ax1, ax2):
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, rotation=30, fontsize=7, color="#aaaaaa")

        plt.tight_layout(pad=1.5)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[backtest] 图表生成失败: {e}", flush=True)
        return None
