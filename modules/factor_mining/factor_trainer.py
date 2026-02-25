"""
线性 + Softmax 因子权重训练器。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .factor_definitions import DEFAULT_FACTOR_WEIGHTS, build_factor_values
from .factor_registry import get_factor_ids_by_set, get_all_factor_ids


def apply_bell_transforms(
    df: pd.DataFrame,
    factor_names: List[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """钟形变换：对指定因子按截面（同 date）求均值 x̄，新增列 {name}_bell = (x - x̄)²，极值在中间。"""
    if not factor_names or date_col not in df.columns:
        return df.copy()
    out = df.copy()
    for name in factor_names:
        if name not in out.columns:
            continue
        mean_by_date = out.groupby(date_col)[name].transform("mean")
        out[f"{name}_bell"] = (out[name].astype(float) - mean_by_date) ** 2
    return out


# 因子列表统一从 factor_registry 读取（style=风格与估值，trading=情绪与交易）
STYLE_FACTOR_KEYS = get_factor_ids_by_set("style")
TRADING_FACTOR_KEYS = get_factor_ids_by_set("trading")
ALL_FACTOR_KEYS = get_all_factor_ids()
MOMENTUM_FACTOR_KEYS = get_factor_ids_by_set("momentum")
VOLATILITY_FACTOR_KEYS = get_factor_ids_by_set("volatility")
VOLUME_FACTOR_KEYS = get_factor_ids_by_set("volume")
REVERSAL_FACTOR_KEYS = get_factor_ids_by_set("reversal")


@dataclass
class TrainConfig:
    factor_set: str = "hybrid"
    epochs: int = 300
    lr: float = 0.05
    l2_lambda: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    early_stop_positive: bool = True
    feature_select_count: int = 5
    feature_names: Optional[List[str]] = None  # 显式指定因子列时使用（含 _bell 等），覆盖 factor_set 选取


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    s = np.sum(exp)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return exp / s


def _select_features(factor_set: str) -> List[str]:
    # factor_set: style | trading | momentum | volatility | volume | reversal | hybrid
    return get_factor_ids_by_set(factor_set)


def _pick_top_features(
    x_df: pd.DataFrame,
    y: np.ndarray,
    candidate_features: List[str],
    k: int,
) -> List[str]:
    """
    从候选因子中挑选最有解释力的前 k 个，降低过拟合风险。
    评分使用 |corr| 与 |rank_corr| 的加权和。
    """
    scores = []
    for f in candidate_features:
        arr = x_df[f].to_numpy(dtype=float)
        c = abs(_corr(arr, y))
        rc = abs(_rank_corr(arr, y))
        score = 0.45 * c + 0.55 * rc
        scores.append((f, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = [f for f, _ in scores[:k]]
    return chosen


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ra = pd.Series(a).rank(pct=True).to_numpy()
    rb = pd.Series(b).rank(pct=True).to_numpy()
    c = np.corrcoef(ra, rb)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _top_bottom_spread(scores: np.ndarray, returns: np.ndarray, q: int = 5) -> float:
    if len(scores) < 10 or len(returns) < 10:
        return 0.0
    df = pd.DataFrame({"score": scores, "ret": returns}).dropna()
    if len(df) < 10:
        return 0.0
    try:
        df["bucket"] = pd.qcut(df["score"], q=min(q, len(df)), labels=False, duplicates="drop")
        grp = df.groupby("bucket")["ret"].mean()
        if len(grp) < 2:
            return 0.0
        return float(grp.iloc[-1] - grp.iloc[0])
    except Exception:
        return 0.0


def _build_learned_weights(feature_names: List[str], w: np.ndarray) -> Dict[str, Dict[str, float]]:
    style_weights = {k: 0.0 for k in STYLE_FACTOR_KEYS}
    trading_weights = {k: 0.0 for k in TRADING_FACTOR_KEYS}
    for name, value in zip(feature_names, w):
        if name in STYLE_FACTOR_KEYS:
            style_weights[name] = float(value)
        elif name in TRADING_FACTOR_KEYS:
            trading_weights[name] = float(value)

    style_sum = sum(style_weights.values())
    trading_sum = sum(trading_weights.values())
    groups_total = style_sum + trading_sum
    if groups_total <= 0:
        groups = {"style": 0.6, "trading": 0.4}
    else:
        groups = {
            "style": float(style_sum / groups_total),
            "trading": float(trading_sum / groups_total),
        }

    def _normalize(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        if s <= 0:
            return d
        return {k: float(v / s) for k, v in d.items()}

    return {
        "style": _normalize(style_weights),
        "trading": _normalize(trading_weights),
        "groups": groups,
        "flat": {name: float(v) for name, v in zip(feature_names, w)},
        "flat_all": {
            **{k: float(style_weights.get(k, 0.0)) for k in STYLE_FACTOR_KEYS},
            **{k: float(trading_weights.get(k, 0.0)) for k in TRADING_FACTOR_KEYS},
        },
    }


def _selection_objective(metrics: Dict[str, float]) -> float:
    # 强调排序效果，其次线性相关，最后看分层收益
    return (
        float(metrics.get("val_rank_ic", 0.0))
        + 0.7 * float(metrics.get("val_corr", 0.0))
        + 15.0 * float(metrics.get("val_spread", 0.0))
    )


def _calc_oos_stability(val_pred: np.ndarray, y_val: np.ndarray, slices: int = 3) -> Dict[str, object]:
    n = len(val_pred)
    if n < 30:
        return {
            "slice_metrics": [],
            "distribution": {"rank_ic_mean": 0.0, "rank_ic_std": 0.0, "corr_mean": 0.0, "corr_std": 0.0, "spread_mean": 0.0, "spread_std": 0.0},
            "oos_stability_score": 0.0,
        }
    k = max(2, min(int(slices), 5))
    idx_splits = np.array_split(np.arange(n), k)
    slice_metrics = []
    for i, idx_arr in enumerate(idx_splits):
        if len(idx_arr) < 8:
            continue
        p = val_pred[idx_arr]
        y = y_val[idx_arr]
        slice_metrics.append(
            {
                "slice": i + 1,
                "count": int(len(idx_arr)),
                "corr": round(_corr(p, y), 6),
                "rank_ic": round(_rank_corr(p, y), 6),
                "spread": round(_top_bottom_spread(p, y), 6),
            }
        )
    if not slice_metrics:
        return {
            "slice_metrics": [],
            "distribution": {"rank_ic_mean": 0.0, "rank_ic_std": 0.0, "corr_mean": 0.0, "corr_std": 0.0, "spread_mean": 0.0, "spread_std": 0.0},
            "oos_stability_score": 0.0,
        }
    rank_ic_arr = np.array([m["rank_ic"] for m in slice_metrics], dtype=float)
    corr_arr = np.array([m["corr"] for m in slice_metrics], dtype=float)
    spread_arr = np.array([m["spread"] for m in slice_metrics], dtype=float)
    rank_mean, rank_std = float(rank_ic_arr.mean()), float(rank_ic_arr.std(ddof=0))
    corr_mean, corr_std = float(corr_arr.mean()), float(corr_arr.std(ddof=0))
    spread_mean, spread_std = float(spread_arr.mean()), float(spread_arr.std(ddof=0))
    # 简单稳定度打分：均值奖励 - 波动惩罚
    score = rank_mean + 0.6 * corr_mean + 8.0 * spread_mean - 0.5 * rank_std - 0.3 * corr_std - 2.0 * spread_std
    return {
        "slice_metrics": slice_metrics,
        "distribution": {
            "rank_ic_mean": round(rank_mean, 6),
            "rank_ic_std": round(rank_std, 6),
            "corr_mean": round(corr_mean, 6),
            "corr_std": round(corr_std, 6),
            "spread_mean": round(spread_mean, 6),
            "spread_std": round(spread_std, 6),
        },
        "oos_stability_score": round(float(score), 6),
    }


def _train_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2_lambda: float,
    early_stop_positive: bool,
    seed: int,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    w_raw = rng.normal(0, 0.01, size=x_train.shape[1])
    history_train = []
    history_val = []
    best = None
    best_val = float("inf")

    stop_meta = {"stopped_early": False, "stopped_epoch": None, "stop_reason": ""}
    total_epochs = max(50, int(epochs))
    min_epochs = max(30, int(total_epochs * 0.2))

    for epoch_idx in range(total_epochs):
        w = _softmax(w_raw)
        y_hat = x_train @ w
        err = y_hat - y_train
        train_loss = float(np.mean(err ** 2) + l2_lambda * np.sum(w ** 2))
        history_train.append(train_loss)

        grad_w = (2.0 / len(x_train)) * (x_train.T @ err) + 2.0 * l2_lambda * w
        dot = float(np.dot(grad_w, w))
        grad_raw = w * (grad_w - dot)
        w_raw = w_raw - float(lr) * grad_raw

        w_val = _softmax(w_raw)
        val_err = (x_val @ w_val) - y_val
        val_loss = float(np.mean(val_err ** 2) + l2_lambda * np.sum(w_val ** 2))
        history_val.append(val_loss)
        val_pred = x_val @ w_val
        val_corr = _corr(val_pred, y_val)
        val_rank_ic = _rank_corr(val_pred, y_val)
        val_spread = _top_bottom_spread(val_pred, y_val)
        if progress_callback:
            try:
                progress_callback(epoch_idx + 1, total_epochs, train_loss, val_loss)
            except Exception:
                pass
        if val_loss < best_val:
            best_val = val_loss
            best = w_val.copy()

        if (
            early_stop_positive
            and (epoch_idx + 1) >= min_epochs
            and val_corr > 0
            and val_rank_ic > 0
            and val_spread > 0
        ):
            best = w_val.copy()
            best_val = val_loss
            stop_meta = {
                "stopped_early": True,
                "stopped_epoch": epoch_idx + 1,
                "stop_reason": "ic_rankic_spread_positive",
            }
            break

    w_final = best if best is not None else _softmax(w_raw)
    train_pred = x_train @ w_final
    val_pred = x_val @ w_final
    metrics = {
        "train_loss": round(float(np.mean((train_pred - y_train) ** 2)), 8),
        "val_loss": round(float(np.mean((val_pred - y_val) ** 2)), 8),
        "train_corr": round(_corr(train_pred, y_train), 6),
        "val_corr": round(_corr(val_pred, y_val), 6),
        "train_rank_ic": round(_rank_corr(train_pred, y_train), 6),
        "val_rank_ic": round(_rank_corr(val_pred, y_val), 6),
        "train_spread": round(_top_bottom_spread(train_pred, y_train), 6),
        "val_spread": round(_top_bottom_spread(val_pred, y_val), 6),
    }
    return {
        "weights": w_final,
        "metrics": metrics,
        "history_train": history_train,
        "history_val": history_val,
        "stop_meta": stop_meta,
    }


def build_training_samples(
    data_dict: Dict[str, pd.DataFrame],
    days: int = 252,
    label_horizon: int = 1,
    max_window: int = 120,
    extra_factors_by_code: Optional[Dict[str, Dict[str, float]]] = None,
    extra_factors_by_code_date: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    """构建训练样本。截面因子 PE/换手率 通过 extra_factors_by_code_date 传入：{(code, date_str): {"pe_ratio", "turnover_ratio"}}；兼容旧接口 extra_factors_by_code（按代码一份，不按日）。"""
    rows: List[Dict[str, float]] = []
    horizon = max(1, int(label_horizon))
    lookback_floor = 60
    extra_by_code = extra_factors_by_code or {}
    extra_by_code_date = extra_factors_by_code_date or {}

    def _date_str(x) -> str:
        s = str(x).strip()[:10]
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    for code, df in data_dict.items():
        if df is None or len(df) < lookback_floor + horizon + 1:
            continue
        df = df.copy().dropna().reset_index(drop=True)
        start_i = max(lookback_floor, len(df) - int(days) - horizon)
        end_i = len(df) - horizon - 1
        for i in range(start_i, end_i + 1):
            window = df.iloc[max(0, i - max_window + 1): i + 1]
            if len(window) < lookback_floor:
                continue
            row_date = _date_str(df.iloc[i]["date"])
            extra = extra_by_code_date.get((code, row_date)) or extra_by_code.get(code) or {}
            try:
                factors = build_factor_values(window, extra_factors=extra if extra else None)
                p0 = float(df.iloc[i]["close"])
                p1 = float(df.iloc[i + horizon]["close"])
                if p0 <= 0:
                    continue
                y = p1 / p0 - 1.0
                row = {"date": str(df.iloc[i]["date"]), "stock_code": code, "y": y}
                for k in ALL_FACTOR_KEYS:
                    row[k] = float(factors.get(k, 0.0))
                rows.append(row)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values("date").reset_index(drop=True)


def train_linear_softmax_weights(
    sample_df: pd.DataFrame,
    config: TrainConfig,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    if sample_df is None or sample_df.empty:
        raise ValueError("训练样本为空")

    if config.feature_names:
        base_feature_names = [f for f in config.feature_names if f in sample_df.columns]
        if len(base_feature_names) == 0:
            raise ValueError("显式 feature_names 在样本中无匹配列")
    else:
        base_feature_names = _select_features(config.factor_set)
    if len(base_feature_names) == 0:
        raise ValueError("无可训练因子")
    missing = [f for f in base_feature_names if f not in sample_df.columns]
    if missing:
        raise ValueError(f"训练样本缺少因子列: {missing}")

    df = sample_df[base_feature_names + ["y", "date"]].copy().dropna()
    if len(df) < 80:
        raise ValueError(f"训练样本不足（{len(df)}），至少需要80条")

    # 与线上打分口径对齐：按日期做横截面标准化（z-score）
    for f in base_feature_names:
        grp = df.groupby("date")[f]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        z = (df[f] - mean) / std
        df[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    x_all_df = df[base_feature_names].copy()
    y = df["y"].to_numpy(dtype=float)

    split = int(len(df) * (1.0 - min(max(config.val_ratio, 0.05), 0.4)))
    split = max(40, min(split, len(df) - 20))
    x_train_all_df = x_all_df.iloc[:split].copy()
    y_train = y[:split]
    y_val = y[split:]

    if config.feature_names:
        feature_names = list(base_feature_names)
        subset_metrics = [{"features": feature_names, "metrics": {}, "objective": 0.0}]
    else:
        k = int(config.feature_select_count or 5)
        k = max(3, min(5, k))
        k = min(k, len(base_feature_names))
        feature_names = _pick_top_features(x_train_all_df, y_train, base_feature_names, k)
        subset_candidates: List[List[str]] = []
        candidate_pool = list(base_feature_names)
        max_k = min(5, len(candidate_pool))
        min_k = min(3, max_k)
        for kk in range(min_k, max_k + 1):
            subset_candidates.extend([list(c) for c in combinations(candidate_pool, kk)])
        if not subset_candidates:
            subset_candidates = [feature_names]
        scout_epochs = max(60, min(160, int(config.epochs * 0.4)))
        best_subset = subset_candidates[0]
        best_subset_score = -1e18
        subset_metrics = []
        for idx, subset in enumerate(subset_candidates):
            x_sub = x_all_df[subset].to_numpy(dtype=float)
            x_train_sub = x_sub[:split]
            x_val_sub = x_sub[split:]
            scout = _train_subset(
                x_train_sub,
                y_train,
                x_val_sub,
                y_val,
                epochs=scout_epochs,
                lr=config.lr,
                l2_lambda=config.l2_lambda,
                early_stop_positive=False,
                seed=config.seed + idx,
                progress_callback=None,
            )
            score = _selection_objective(scout["metrics"])
            subset_metrics.append({"features": subset, "metrics": scout["metrics"], "objective": round(float(score), 6)})
            if score > best_subset_score:
                best_subset_score = score
                best_subset = subset
        feature_names = list(best_subset)

    x = x_all_df[feature_names].to_numpy(dtype=float)
    x_train = x[:split]
    x_val = x[split:]

    trained = _train_subset(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=config.epochs,
        lr=config.lr,
        l2_lambda=config.l2_lambda,
        early_stop_positive=config.early_stop_positive,
        seed=config.seed,
        progress_callback=progress_callback,
    )
    w_final = trained["weights"]
    history_train = trained["history_train"]
    history_val = trained["history_val"]
    stop_meta = trained["stop_meta"]
    metrics = trained["metrics"]
    learned = _build_learned_weights(feature_names, w_final)
    oos_stability = _calc_oos_stability(val_pred=(x_val @ w_final), y_val=y_val, slices=3)

    return {
        "factor_set": config.factor_set,
        "base_feature_names": base_feature_names,
        "feature_names": feature_names,
        "selected_feature_count": len(feature_names),
        "best_factor_combo": feature_names,
        "best_factor_weights": {k: float(learned["flat_all"].get(k, 0.0)) for k in ALL_FACTOR_KEYS},
        "subset_search": subset_metrics,
        "learned_weights": learned,
        "metrics": metrics,
        "sample_count": int(len(df)),
        "train_count": int(len(x_train)),
        "val_count": int(len(x_val)),
        "sample_start": str(df.iloc[0]["date"]),
        "sample_end": str(df.iloc[-1]["date"]),
        "normalization": {
            "method": "cross_section_zscore_by_date",
            "mean": {k: 0.0 for k in feature_names},
            "std": {k: 1.0 for k in feature_names},
        },
        "history": {
            "train_loss_tail": [round(float(v), 8) for v in history_train[-10:]],
            "val_loss_tail": [round(float(v), 8) for v in history_val[-10:]],
        },
        "stop_meta": stop_meta,
        "oos_stability": oos_stability,
    }


def train_linear_weights(
    sample_df: pd.DataFrame,
    feature_names: List[str],
    val_ratio: float = 0.2,
    l2_lambda: Optional[float] = None,
    lambda_grid: Optional[List[float]] = None,
) -> Dict[str, object]:
    """
    用 Ridge 回归拟合 y = X @ w，权重可为负数。若提供 lambda_grid 则在验证集上选 val_rank_ic 最大的 lambda。
    """
    if sample_df is None or sample_df.empty:
        raise ValueError("训练样本为空")
    missing = [f for f in feature_names if f not in sample_df.columns]
    if missing:
        raise ValueError(f"训练样本缺少因子列: {missing}")
    if "y" not in sample_df.columns or "date" not in sample_df.columns:
        raise ValueError("训练样本需含 date 与 y 列")
    df = sample_df[feature_names + ["y", "date"]].copy().dropna()
    if len(df) < 80:
        raise ValueError(f"训练样本不足（{len(df)}），至少需要80条")
    for f in feature_names:
        grp = df.groupby("date")[f]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        z = (df[f] - mean) / std
        df[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_all = df[feature_names].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    split = max(40, min(int(len(df) * (1.0 - val_ratio)), len(df) - 20))
    x_train, x_val = x_all[:split], x_all[split:]
    y_train, y_val = y[:split], y[split:]
    lambdas = lambda_grid if lambda_grid else [float(l2_lambda) if l2_lambda is not None else 1e-4]
    best_val_ic = -1e9
    best_w = None
    best_lam = lambdas[0]
    for lam in lambdas:
        reg = lam * np.eye(x_train.shape[1])
        w_cand = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
        w_cand = np.asarray(w_cand).ravel()
        v_ic = _rank_corr(x_val @ w_cand, y_val)
        if v_ic > best_val_ic:
            best_val_ic = v_ic
            best_w = w_cand
            best_lam = lam
    w = best_w if best_w is not None else np.linalg.solve(
        x_train.T @ x_train + best_lam * np.eye(x_train.shape[1]), x_train.T @ y_train
    ).ravel()
    train_pred = x_train @ w
    val_pred = x_val @ w
    metrics = {
        "train_corr": round(_corr(train_pred, y_train), 6),
        "val_corr": round(_corr(val_pred, y_val), 6),
        "train_rank_ic": round(_rank_corr(train_pred, y_train), 6),
        "val_rank_ic": round(_rank_corr(val_pred, y_val), 6),
        "train_spread": round(_top_bottom_spread(train_pred, y_train), 6),
        "val_spread": round(_top_bottom_spread(val_pred, y_val), 6),
        "best_l2_lambda": round(best_lam, 8),
    }
    flat = {f: float(w[i]) for i, f in enumerate(feature_names)}
    learned = {
        "flat": flat,
        "style": {k: flat.get(k, 0.0) for k in STYLE_FACTOR_KEYS},
        "trading": {k: flat.get(k, 0.0) for k in TRADING_FACTOR_KEYS},
        "groups": {"style": 0.5, "trading": 0.5},
        "flat_all": {**{k: flat.get(k, 0.0) for k in STYLE_FACTOR_KEYS}, **{k: flat.get(k, 0.0) for k in TRADING_FACTOR_KEYS}},
    }
    return {
        "feature_names": feature_names,
        "best_factor_combo": feature_names,
        "learned_weights": learned,
        "metrics": metrics,
        "weight_type": "linear",
    }
