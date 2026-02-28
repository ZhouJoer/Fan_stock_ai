"""
因子定义与默认权重。

因子大类（与 factor_registry 一致）：style=风格与估值（动量、波动、PE/换手等），
trading=情绪与交易（成交量、反转、RSI/量价背离等）。PE、换手率由 cross_section_data 在样本构建时按 (code, date) join。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# 建议做钟形变换的因子（收益呈中间极值）
FACTORS_SUGGEST_BELL: tuple = ("rsi", "intraday_amplitude")

# 大类权重：style=风格与估值，trading=情绪与交易
DEFAULT_FACTOR_GROUP_WEIGHTS: Dict[str, float] = {
    "style": 0.6,
    "trading": 0.4,
}

DEFAULT_FACTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "style": {  # 风格与估值
        "momentum_20": 0.22,
        "momentum_60": 0.18,
        "trend_strength": 0.15,
        "low_volatility": 0.18,
        "pe_ratio": 0.12,
        "turnover_ratio": 0.15,
    },
    "trading": {  # 情绪与交易
        "volume_surge": 0.20,
        "turnover_proxy": 0.18,
        "short_reversal": 0.12,
        "intraday_amplitude": 0.12,
        "rsi": 0.12,
        "price_volume_divergence": 0.12,
        "high_position_volume": 0.12,
        "volume_oscillation": 0.14,
    },
}


def _safe_get_last(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return float(default)
    value = series.iloc[-1]
    if pd.isna(value):
        return float(default)
    return float(value)


def _calc_momentum(close: pd.Series, period: int) -> float:
    if close is None or len(close) <= period:
        return 0.0
    base = float(close.iloc[-period - 1])
    if base <= 0:
        return 0.0
    return float(close.iloc[-1] / base - 1.0)


def _calc_trend_strength(close: pd.Series, period: int = 20) -> float:
    if close is None or len(close) < period:
        return 0.0
    y = close.tail(period).astype(float).values
    x = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    mean_price = np.mean(y)
    if mean_price == 0:
        return 0.0
    return float(slope / mean_price)


def _calc_low_volatility(close: pd.Series, period: int = 20) -> float:
    if close is None or len(close) < period + 1:
        return 0.0
    ret = close.pct_change().dropna().tail(period)
    vol = float(ret.std()) if len(ret) else 0.0
    return -vol


def _calc_volume_surge(volume: pd.Series, period: int = 20) -> float:
    if volume is None or len(volume) < period + 1:
        return 0.0
    ma = float(volume.tail(period).mean())
    last = float(volume.iloc[-1])
    if ma <= 0:
        return 0.0
    return last / ma - 1.0


def _calc_turnover_proxy(volume: pd.Series, period: int = 5) -> float:
    if volume is None or len(volume) < period:
        return 0.0
    short = float(volume.tail(period).mean())
    long = float(volume.tail(max(20, period)).mean())
    if long <= 0:
        return 0.0
    return short / long - 1.0


def _calc_short_reversal(close: pd.Series, period: int = 5) -> float:
    if close is None or len(close) <= period:
        return 0.0
    # 短期涨幅取负，偏向“过热回落/超跌反弹”的反转逻辑
    return -_calc_momentum(close, period)


def _calc_intraday_amplitude(df: pd.DataFrame, period: int = 10) -> float:
    if df is None or len(df) < period:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float).replace(0, np.nan)
    amp = ((high - low) / close).replace([np.inf, -np.inf], np.nan).dropna().tail(period)
    if len(amp) == 0:
        return 0.0
    return float(amp.mean())


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI(14)。返回值 0~100，缺数据时返回 50（中性）。"""
    if close is None or len(close) < period + 1:
        return 50.0
    delta = close.astype(float).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0 or np.isnan(avg_loss):
        return 100.0 if (avg_gain and not np.isnan(avg_gain)) else 50.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi) if not np.isnan(rsi) else 50.0


def _calc_price_volume_divergence(df: pd.DataFrame, period: int = 10) -> float:
    """量价背离：价变化与量变化的相关系数，负或弱正表示背离。返回 -corr 使背离大时因子值大。"""
    if df is None or len(df) < period + 1:
        return 0.0
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    if len(volume) < period + 1:
        return 0.0
    price_ret = close.pct_change().dropna().tail(period)
    vol_ret = volume.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(period)
    common = price_ret.index.intersection(vol_ret.index)
    if len(common) < 5:
        return 0.0
    p = price_ret.loc[common].values
    v = vol_ret.loc[common].values
    c = np.corrcoef(p, v)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(-c)


def _calc_high_position_volume(df: pd.DataFrame, period: int = 10) -> float:
    """高位成交：近期收盘在 N 日高点的分位 × 成交量加权的度量。高表示成交更多集中在高位。"""
    if df is None or len(df) < period:
        return 0.0
    high = df["high"].astype(float).tail(period)
    low = df["low"].astype(float).tail(period)
    close = df["close"].astype(float).tail(period)
    volume = df["volume"].astype(float).tail(period) if "volume" in df.columns else pd.Series(1.0, index=close.index)
    h, l = high.max(), low.min()
    if h <= l:
        return 0.0
    pos = (close - l) / (h - l)
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    if volume.sum() <= 0:
        return float(pos.mean())
    weight = volume / volume.sum()
    return float((pos * weight).sum())


def _calc_volume_oscillation(volume: pd.Series, period: int = 20) -> float:
    """成交量震荡：成交量相对均线的变异系数（std/mean），无量纲波动。"""
    if volume is None or len(volume) < period:
        return 0.0
    v = volume.astype(float).tail(period)
    mean_v = v.mean()
    if mean_v <= 0:
        return 0.0
    std_v = v.std()
    if std_v is None or np.isnan(std_v) or std_v == 0:
        return 0.0
    return float(std_v / mean_v)


def _calc_bias(close: pd.Series, period: int) -> float:
    if close is None or len(close) < period:
        return 0.0
    ma = float(close.tail(period).mean())
    if ma <= 0:
        return 0.0
    return float(close.iloc[-1] / ma - 1.0)


def _calc_roc(close: pd.Series, period: int) -> float:
    return _calc_momentum(close, period)


def _calc_price_ratio(close: pd.Series, period: int) -> float:
    if close is None or len(close) < period:
        return 0.0
    ma = float(close.tail(period).mean())
    if ma <= 0:
        return 0.0
    return float(close.iloc[-1] / ma - 1.0)


def _calc_ema_ratio(close: pd.Series, period: int) -> float:
    if close is None or len(close) < period:
        return 0.0
    ema = close.astype(float).ewm(span=period, adjust=False).mean().iloc[-1]
    last = float(close.iloc[-1])
    if ema <= 0:
        return 0.0
    return float(last / ema - 1.0)


def _calc_trix(close: pd.Series, period: int) -> float:
    if close is None or len(close) < period * 3:
        return 0.0
    s = close.astype(float)
    ema1 = s.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change()
    return float(trix.iloc[-1]) if len(trix) and pd.notna(trix.iloc[-1]) else 0.0


def _calc_cci(df: pd.DataFrame, period: int = 20) -> float:
    if df is None or len(df) < period:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
    last = cci.iloc[-1] if len(cci) else np.nan
    return float(last) if pd.notna(last) else 0.0


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    last_atr = atr.iloc[-1] if len(atr) else np.nan
    last_close = close.iloc[-1] if len(close) else np.nan
    if pd.isna(last_atr) or pd.isna(last_close) or last_close == 0:
        return 0.0
    # 归一化，避免价格量纲干扰
    return float(last_atr / last_close)


def _calc_boll(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[float, float]:
    if close is None or len(close) < period:
        return 0.0, 0.0
    s = close.astype(float).tail(period)
    mid = float(s.mean())
    std = float(s.std()) if len(s) else 0.0
    if mid <= 0:
        return 0.0, 0.0
    upper = mid + k * std
    lower = max(mid - k * std, 1e-8)
    last = float(close.iloc[-1])
    # 使用相对位置，便于横截面比较
    return float(upper / last - 1.0), float(last / lower - 1.0)


def _calc_ret_series(close: pd.Series) -> pd.Series:
    if close is None or len(close) < 3:
        return pd.Series(dtype=float)
    return close.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _calc_variance(ret: pd.Series, period: int) -> float:
    if ret is None or len(ret) < period:
        return 0.0
    r = ret.tail(period)
    return float(r.var() * 252.0) if len(r) else 0.0


def _calc_skewness(ret: pd.Series, period: int) -> float:
    if ret is None or len(ret) < period:
        return 0.0
    v = ret.tail(period).skew()
    return float(v) if pd.notna(v) else 0.0


def _calc_kurtosis(ret: pd.Series, period: int) -> float:
    if ret is None or len(ret) < period:
        return 0.0
    v = ret.tail(period).kurt()
    return float(v) if pd.notna(v) else 0.0


def _calc_sharpe(ret: pd.Series, period: int) -> float:
    if ret is None or len(ret) < period:
        return 0.0
    r = ret.tail(period)
    std = float(r.std()) if len(r) else 0.0
    if std <= 0:
        return 0.0
    return float(np.sqrt(252.0) * r.mean() / std)


def _calc_vol_ma(volume: pd.Series, period: int) -> float:
    if volume is None or len(volume) < period:
        return 0.0
    return float(volume.astype(float).tail(period).mean())


def build_factor_values(
    df: pd.DataFrame,
    extra_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """从 OHLCV 计算因子。extra_factors 可传入截面数据（如 pe_ratio、turnover_ratio）由调用方 join。"""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    ret = _calc_ret_series(close)
    boll_up, boll_down = _calc_boll(close, 20, 2.0)

    out = {
        "momentum_20": _calc_momentum(close, 20),
        "momentum_60": _calc_momentum(close, 60),
        "ema5": _calc_ema_ratio(close, 5),
        "ema10": _calc_ema_ratio(close, 10),
        "ema12": _calc_ema_ratio(close, 12),
        "ema20": _calc_ema_ratio(close, 20),
        "ema26": _calc_ema_ratio(close, 26),
        "ema120": _calc_ema_ratio(close, 120),
        "bias5": _calc_bias(close, 5),
        "bias10": _calc_bias(close, 10),
        "bias20": _calc_bias(close, 20),
        "bias60": _calc_bias(close, 60),
        "roc6": _calc_roc(close, 6),
        "roc12": _calc_roc(close, 12),
        "roc60": _calc_roc(close, 60),
        "roc120": _calc_roc(close, 120),
        "price1m": _calc_price_ratio(close, 20),
        "price3m": _calc_price_ratio(close, 60),
        "price1y": _calc_price_ratio(close, 240),
        "trix5": _calc_trix(close, 5),
        "trix10": _calc_trix(close, 10),
        "cci20": _calc_cci(df, 20),
        "trend_strength": _calc_trend_strength(close, 20),
        "low_volatility": _calc_low_volatility(close, 20),
        "variance20": _calc_variance(ret, 20),
        "variance60": _calc_variance(ret, 60),
        "variance120": _calc_variance(ret, 120),
        "skewness_20": _calc_skewness(ret, 20),
        "skewness_60": _calc_skewness(ret, 60),
        "skewness_120": _calc_skewness(ret, 120),
        "kurtosis_20": _calc_kurtosis(ret, 20),
        "kurtosis_60": _calc_kurtosis(ret, 60),
        "kurtosis_120": _calc_kurtosis(ret, 120),
        "sharpe_ratio_20": _calc_sharpe(ret, 20),
        "sharpe_ratio_60": _calc_sharpe(ret, 60),
        "sharpe_ratio_120": _calc_sharpe(ret, 120),
        "boll_up": boll_up,
        "boll_down": boll_down,
        "atr6": _calc_atr(df, 6),
        "atr14": _calc_atr(df, 14),
        "volume_surge": _calc_volume_surge(volume, 20),
        "turnover_proxy": _calc_turnover_proxy(volume, 5),
        "vol5": _calc_vol_ma(volume, 5),
        "vol10": _calc_vol_ma(volume, 10),
        "vol20": _calc_vol_ma(volume, 20),
        "vol60": _calc_vol_ma(volume, 60),
        "vol120": _calc_vol_ma(volume, 120),
        "vol240": _calc_vol_ma(volume, 240),
        "vroc6": _calc_roc(volume, 6),
        "vroc12": _calc_roc(volume, 12),
        "short_reversal": _calc_short_reversal(close, 5),
        "intraday_amplitude": _calc_intraday_amplitude(df, 10),
        "rsi": _calc_rsi(close, 14),
        "price_volume_divergence": _calc_price_volume_divergence(df, 10),
        "high_position_volume": _calc_high_position_volume(df, 10),
        "volume_oscillation": _calc_volume_oscillation(volume, 20),
        "close": _safe_get_last(close),
    }
    if extra_factors:
        for k, v in extra_factors.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[k] = float(v)
    return out
