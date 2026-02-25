"""
横截面因子打分引擎。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .factor_definitions import DEFAULT_FACTOR_GROUP_WEIGHTS, DEFAULT_FACTOR_WEIGHTS, build_factor_values
from .factor_trainer import (
    STYLE_FACTOR_KEYS,
    TRADING_FACTOR_KEYS,
    MOMENTUM_FACTOR_KEYS,
    VOLATILITY_FACTOR_KEYS,
    VOLUME_FACTOR_KEYS,
    REVERSAL_FACTOR_KEYS,
)

# 因子集 -> (风格与估值侧用到的键, 情绪与交易侧用到的键)，空则该侧权重置 0
FACTOR_SET_SUBSET = {
    "momentum": (MOMENTUM_FACTOR_KEYS, []),
    "volatility": (VOLATILITY_FACTOR_KEYS, []),
    "volume": ([], VOLUME_FACTOR_KEYS),
    "reversal": ([], REVERSAL_FACTOR_KEYS),
}


def _zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std is None or std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: float(v) for k, v in (weights or {}).items() if v is not None and float(v) >= 0}
    total = sum(clean.values())
    if total <= 0:
        return clean
    return {k: v / total for k, v in clean.items()}


def _merge_factor_weights(
    factor_set: str,
    score_weights: Dict[str, float] | None,
    learned_weights: Dict[str, Dict[str, float]] | None = None,
    weight_source: str = "manual",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], str]:
    source_used = "manual"
    group_weights = dict(DEFAULT_FACTOR_GROUP_WEIGHTS)
    if factor_set == "style":
        group_weights = {"style": 1.0, "trading": 0.0}
    elif factor_set == "trading":
        group_weights = {"style": 0.0, "trading": 1.0}
    elif factor_set in FACTOR_SET_SUBSET:
        style_keys, trading_keys = FACTOR_SET_SUBSET[factor_set]
        group_weights = {"style": 1.0 if style_keys else 0.0, "trading": 1.0 if trading_keys else 0.0}
    group_weights = _normalize_weights(group_weights)

    style_w = dict(DEFAULT_FACTOR_WEIGHTS["style"])
    trading_w = dict(DEFAULT_FACTOR_WEIGHTS["trading"])
    if weight_source == "learned" and learned_weights:
        style_w = dict(learned_weights.get("style", style_w))
        trading_w = dict(learned_weights.get("trading", trading_w))
        group_weights = dict(learned_weights.get("groups", group_weights))
        source_used = "learned"
    # learned 模式下默认不再叠加手工权重，避免覆盖训练结果
    if score_weights and source_used != "learned":
        for key, value in score_weights.items():
            if key in style_w:
                style_w[key] = float(value)
            elif key in trading_w:
                trading_w[key] = float(value)
            elif key in ("style", "trading"):
                group_weights[key] = float(value)
    style_w = _normalize_weights(style_w)
    trading_w = _normalize_weights(trading_w)
    # 子集因子集：只保留对应键的权重，其余置 0
    if factor_set in FACTOR_SET_SUBSET:
        style_keys, trading_keys = FACTOR_SET_SUBSET[factor_set]
        style_w = {k: (style_w.get(k, 0) if k in style_keys else 0) for k in STYLE_FACTOR_KEYS}
        trading_w = {k: (trading_w.get(k, 0) if k in trading_keys else 0) for k in TRADING_FACTOR_KEYS}
        if style_keys:
            style_w = _normalize_weights(style_w)
        if trading_keys:
            trading_w = _normalize_weights(trading_w)
    group_weights = _normalize_weights(group_weights)
    return style_w, trading_w, group_weights, source_used


def score_cross_section(
    daily_df_dict: Dict[str, pd.DataFrame],
    factor_set: str = "hybrid",
    score_weights: Dict[str, float] | None = None,
    top_n: int = 10,
    learned_weights: Dict[str, Dict[str, float]] | None = None,
    weight_source: str = "manual",
    model_name: str = "",
) -> Dict[str, object]:
    if not daily_df_dict:
        return {
            "selected_codes": [],
            "scores": {},
            "ranked": [],
            "detail": {},
            "weights_used": {},
        }

    style_w, trading_w, group_w, source_used = _merge_factor_weights(
        factor_set,
        score_weights,
        learned_weights=learned_weights,
        weight_source=weight_source,
    )
    factor_map: Dict[str, Dict[str, float]] = {}
    for code, df in daily_df_dict.items():
        if df is None or len(df) < 60:
            continue
        try:
            factor_map[code] = build_factor_values(df)
        except Exception:
            continue

    if not factor_map:
        return {
            "selected_codes": [],
            "scores": {},
            "ranked": [],
            "detail": {},
            "weights_used": {
                "factor_set": factor_set,
                "style": style_w,
                "trading": trading_w,
                "groups": group_w,
                "source": source_used,
                "model_name": model_name,
            },
        }

    frame = pd.DataFrame.from_dict(factor_map, orient="index").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in frame.columns:
        frame[col] = _zscore_series(frame[col].astype(float))

    style_score = pd.Series([0.0] * len(frame), index=frame.index)
    trading_score = pd.Series([0.0] * len(frame), index=frame.index)
    for fac, w in style_w.items():
        if fac in frame.columns:
            style_score += frame[fac] * w
    for fac, w in trading_w.items():
        if fac in frame.columns:
            trading_score += frame[fac] * w

    final_score = style_score * group_w.get("style", 0.0) + trading_score * group_w.get("trading", 0.0)
    ranking = final_score.sort_values(ascending=False)
    n = max(1, min(int(top_n or 10), len(ranking)))
    selected_codes: List[str] = ranking.head(n).index.tolist()

    detail = {}
    for code in ranking.index:
        contrib: Dict[str, float] = {}
        for fac, w in style_w.items():
            if fac in frame.columns:
                contrib[fac] = float(frame.at[code, fac] * w * group_w.get("style", 0.0))
        for fac, w in trading_w.items():
            if fac in frame.columns:
                contrib[fac] = float(frame.at[code, fac] * w * group_w.get("trading", 0.0))
        sorted_contrib = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
        top_positive = [{"factor": k, "contrib": round(float(v), 6)} for k, v in sorted_contrib[:3] if v > 0]
        top_negative = next(({"factor": k, "contrib": round(float(v), 6)} for k, v in reversed(sorted_contrib) if v < 0), None)
        detail[code] = {
            "style_score": round(float(style_score.get(code, 0.0)), 6),
            "trading_score": round(float(trading_score.get(code, 0.0)), 6),
            "total_score": round(float(final_score.get(code, 0.0)), 6),
            "factor_contrib": {k: round(float(v), 6) for k, v in sorted_contrib},
            "top_positive_factors": top_positive,
            "top_negative_factor": top_negative,
        }

    return {
        "selected_codes": selected_codes,
        "scores": {code: float(score) for code, score in ranking.to_dict().items()},
        "ranked": ranking.index.tolist(),
        "detail": detail,
        "weights_used": {
            "factor_set": factor_set,
            "style": style_w,
            "trading": trading_w,
            "groups": group_w,
            "source": source_used,
            "model_name": model_name,
        },
    }
