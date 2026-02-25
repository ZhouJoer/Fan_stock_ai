"""
缠论（缠中说禅）技术分析模块

以缠论结构（分型、笔、线段、中枢）为核心，辅以多指标验证，提升信号可靠性：
1. K线包含关系处理、分型识别、笔的划分、中枢识别
2. 背驰判断（价格与 MACD 动能背离）
3. 三类买卖点（一买背驰、二买回踩、三买突破回踩）
4. MACD：背驰、零轴作用、防狼术（黄白线长期零轴下回避做多）
5. 布林带：二买回踩中轨企稳、三买布林收口辅助
6. 均线：MA20 支撑、均线多头排列辅助二买
7. 成交量：二买缩量企稳、三买放量突破
8. 大周期：周线/月线定方向，顺势做多

输入：DataFrame，需含 open/high/low/close；建议含 MACD/Signal/Histogram、MA5/MA10/MA20、BB_Middle/BB_Upper/BB_Lower
输出：结构数据与交易信号，可与选股池/每日决策对接
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


# ---------- 数据结构 ----------

@dataclass
class KLine:
    """单根K线（可含包含处理后的高低）"""
    idx: int           # 原始索引
    high: float
    low: float
    open: float
    close: float
    date: Any = None
    merged: bool = False  # 是否由包含合并得到


@dataclass
class Fenxing:
    """分型：顶分型或底分型"""
    idx: int           # 中间K线索引
    fx_type: str       # 'top' | 'bottom'
    price: float       # 顶分型为高点，底分型为低点
    date: Any = None


@dataclass
class Bi:
    """笔：连接相邻顶底分型"""
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: int     # 1=上涨笔, -1=下跌笔
    start_date: Any = None
    end_date: Any = None


@dataclass
class Zhongshu:
    """中枢：连续三笔重叠区间，ZG=中枢高点，ZD=中枢低点（多空平衡区）"""
    high: float        # ZG 中枢上沿 = min(三笔高点)
    low: float         # ZD 中枢下沿 = max(三笔低点)
    bi_indices: List[int]  # 构成中枢的笔在笔列表中的索引
    start_idx: int
    end_idx: int


@dataclass
class ChanlunResult:
    """缠论分析结果"""
    fenxings: List[Fenxing] = field(default_factory=list)
    bis: List[Bi] = field(default_factory=list)
    zhongshus: List[Zhongshu] = field(default_factory=list)
    divergence: Optional[str] = None   # 'top' 顶背驰 | 'bottom' 底背驰 | None
    buy_sell_point: Optional[str] = None  # '1buy'|'2buy'|'3buy'|'1sell'|'2sell'|'3sell'|None
    last_bi_direction: Optional[int] = None  # 最后一笔方向 1/-1
    current_zhongshu: Optional[Zhongshu] = None
    reason: str = ""
    volume_ok_2buy: bool = True   # 二买是否满足缩量企稳（量能配合）
    volume_ok_3buy: bool = True   # 三买是否满足放量突破（量能配合）
    # 波段专用：中枢转移 + 中枢幅度（适配 3–10 日波段）
    pivot_transfer: Optional[str] = None   # 'up' 上升波段 | 'down' 下降波段 | None 盘整
    pivot_range_pct: float = 0.0           # 当前中枢幅度 (ZG-ZD)/mid*100，<2% 视为盘整
    volume_breakout_ratio: float = 1.0     # 近期量/均量，三买要求 >=1.3
    # 多指标辅助（缠论结构为核心，指标验证）
    macd_above_zero: bool = True           # 防狼术：MACD 黄白线在零轴上方为 True，长期在零轴下则回避
    boll_mid_ok: bool = False             # 二买时价格回踩布林中轨企稳
    ma_support_ok: bool = False           # 均线支撑/多头：价格 near MA20 或 MA5>MA10>MA20
    boll_squeeze: bool = False            # 布林收口，中阴将结束，可辅助三买


# ---------- K线包含关系处理 ----------

def _is_contained(high1: float, low1: float, high2: float, low2: float) -> bool:
    """判断两根K线是否存在包含关系（一根的 high/low 在另一根范围内）"""
    if high1 <= high2 and low1 >= low2:
        return True
    if high2 <= high1 and low2 >= low1:
        return True
    return False


def _merge_contained_kline(h1: float, l1: float, h2: float, l2: float, up_trend: bool) -> Tuple[float, float]:
    """合并包含关系的两根K线：up_trend 时取高高、低高；down_trend 时取高低、低低"""
    if up_trend:
        return max(h1, h2), max(l1, l2)
    else:
        return min(h1, h2), min(l1, l2)


def process_containment(df: pd.DataFrame) -> List[KLine]:
    """
    处理K线包含关系，得到合并后的K线序列。
    要求 df 有 high, low, open, close，可选 date。
    """
    if df.empty or len(df) < 2:
        return []
    rows = df.reset_index(drop=True)
    result: List[KLine] = []
    i = 0
    while i < len(rows):
        r = rows.iloc[i]
        h, l = float(r['high']), float(r['low'])
        o, c = float(r['open']), float(r['close'])
        date = r.get('date', i)
        merged = False
        # 向后合并所有与当前K线存在包含关系的K线
        j = i + 1
        up_trend = c >= o if i == 0 else (result[-1].close >= result[-1].open)
        while j < len(rows):
            rj = rows.iloc[j]
            hj, lj = float(rj['high']), float(rj['low'])
            if not _is_contained(h, l, hj, lj):
                break
            h, l = _merge_contained_kline(h, l, hj, lj, up_trend)
            merged = True
            date = rj.get('date', j)
            j += 1
        result.append(KLine(idx=i, high=h, low=l, open=o, close=c, date=date, merged=merged))
        i = j if j > i + 1 else i + 1
    return result


# ---------- 分型识别 ----------

def find_fenxings(klines: List[KLine]) -> List[Fenxing]:
    """在合并后的K线序列上识别顶分型与底分型（连续三根K线，中间一根为极值）"""
    fenxings: List[Fenxing] = []
    for i in range(1, len(klines) - 1):
        mid = klines[i]
        left = klines[i - 1]
        right = klines[i + 1]
        # 顶分型：中间高点严格最大
        if mid.high > left.high and mid.high > right.high:
            fenxings.append(Fenxing(idx=mid.idx, fx_type='top', price=mid.high, date=mid.date))
        # 底分型：中间低点严格最小
        elif mid.low < left.low and mid.low < right.low:
            fenxings.append(Fenxing(idx=mid.idx, fx_type='bottom', price=mid.low, date=mid.date))
    return fenxings


# ---------- 笔的划分 ----------

def _fenxing_to_bi_price(f: Fenxing, klines: List[KLine]) -> float:
    """分型对应在K线序列中的实际价格"""
    for k in klines:
        if k.idx == f.idx:
            return k.high if f.fx_type == 'top' else k.low
    return f.price


def build_bis(fenxings: List[Fenxing], klines: List[KLine]) -> List[Bi]:
    """
    由分型序列构建笔：相邻顶-底、底-顶连接成笔。
    笔至少包含 1 个顶 1 个底（缠论要求至少5根K线，此处用分型间隔简化）。
    """
    if len(fenxings) < 2:
        return []
    bis: List[Bi] = []
    i = 0
    while i < len(fenxings) - 1:
        a, b = fenxings[i], fenxings[i + 1]
        if a.fx_type == b.fx_type:
            i += 1
            continue
        p1 = _fenxing_to_bi_price(a, klines)
        p2 = _fenxing_to_bi_price(b, klines)
        direction = 1 if p2 > p1 else -1
        bis.append(Bi(
            start_idx=a.idx, end_idx=b.idx,
            start_price=p1, end_price=p2,
            direction=direction,
            start_date=a.date, end_date=b.date
        ))
        i += 1
    return bis


# ---------- 中枢识别 ----------

def _bi_interval(bi: Bi) -> Tuple[float, float]:
    """笔的价格区间 [low, high]"""
    return (min(bi.start_price, bi.end_price), max(bi.start_price, bi.end_price))


def find_zhongshus(bis: List[Bi]) -> List[Zhongshu]:
    """
    从笔序列中识别中枢：连续三笔重叠区间。
    定义：ZG（中枢高点）= min(三笔高点)，ZD（中枢低点）= max(三笔低点)；有效条件 ZG > ZD。
    """
    if len(bis) < 3:
        return []
    zhongshus: List[Zhongshu] = []
    for i in range(len(bis) - 2):
        b1, b2, b3 = bis[i], bis[i + 1], bis[i + 2]
        low1, high1 = _bi_interval(b1)
        low2, high2 = _bi_interval(b2)
        low3, high3 = _bi_interval(b3)
        zd = max(low1, low2, low3)   # 中枢低点
        zg = min(high1, high2, high3)  # 中枢高点
        if zd < zg:
            zhongshus.append(Zhongshu(
                high=zg, low=zd,
                bi_indices=[i, i + 1, i + 2],
                start_idx=b1.start_idx, end_idx=b3.end_idx
            ))
    return zhongshus


# ---------- 背驰判断 ----------

def check_divergence(
    df: pd.DataFrame,
    bis: List[Bi],
    lookback_bis: int = 2,
    macd_col: str = 'MACD_Hist'
) -> Optional[str]:
    """
    背驰判断：比较最近两段同向走势的价格与 MACD。
    - 顶背驰：价格创新高但 MACD_Hist 未创新高 -> 'top'
    - 底背驰：价格创新低但 MACD_Hist 未创新低 -> 'bottom'
    """
    if len(bis) < 2 or macd_col not in df.columns or len(df) < 10:
        return None
    df = df.reset_index(drop=True)
    # 取最近两笔
    b1, b2 = bis[-2], bis[-1]
    if b1.direction != b2.direction:
        return None
    # 每笔对应区间的价格极值与 MACD 极值
    def segment_extremes(start_idx: int, end_idx: int):
        seg = df.iloc[max(0, start_idx): min(end_idx + 1, len(df))]
        if seg.empty:
            return None, None, None, None
        price_high = float(seg['high'].max())
        price_low = float(seg['low'].min())
        macd_vals = seg[macd_col].dropna()
        macd_max = float(macd_vals.max()) if len(macd_vals) else None
        macd_min = float(macd_vals.min()) if len(macd_vals) else None
        return price_high, price_low, macd_max, macd_min

    ph1, pl1, mh1, ml1 = segment_extremes(b1.start_idx, b1.end_idx)
    ph2, pl2, mh2, ml2 = segment_extremes(b2.start_idx, b2.end_idx)
    if ph1 is None or ph2 is None:
        return None
    # 上涨两笔：顶背驰 = 价格新高但 MACD 未新高
    if b1.direction == 1 and mh1 is not None and mh2 is not None:
        if ph2 > ph1 and mh2 < mh1:
            return 'top'
    # 下跌两笔：底背驰 = 价格新低但 MACD 未新低
    if b1.direction == -1 and ml1 is not None and ml2 is not None:
        if pl2 < pl1 and ml2 > ml1:
            return 'bottom'
    return None


# ---------- 大周期（周线/月线）判断 ----------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 有 DatetimeIndex，便于重采样。不修改原 df，返回带索引的副本。"""
    if df is None or df.empty:
        return df
    d = df.copy()
    if 'date' in d.columns:
        d = d.set_index(pd.to_datetime(d['date']))
    elif not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    return d.dropna(how='all', subset=['open', 'high', 'low', 'close'])


def _resample_to_weekly(df_daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """日线重采样为周线 OHLC。需至少约 25 根日线。"""
    if df_daily is None or len(df_daily) < 20:
        return None
    d = _ensure_datetime_index(df_daily)
    if d is None or len(d) < 20:
        return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in d.columns:
        agg['volume'] = 'sum'
    try:
        w = d.resample('W').agg(agg).dropna(subset=['close'])
    except Exception:
        return None
    return w if len(w) >= 3 else None


def _resample_to_monthly(df_daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """日线重采样为月线 OHLC。需至少约 60 根日线。"""
    if df_daily is None or len(df_daily) < 40:
        return None
    d = _ensure_datetime_index(df_daily)
    if d is None or len(d) < 40:
        return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in d.columns:
        agg['volume'] = 'sum'
    try:
        m = d.resample('ME').agg(agg).dropna(subset=['close'])
    except Exception:
        return None
    return m if len(m) >= 2 else None


def get_big_trend(df_daily: pd.DataFrame) -> str:
    """
    基于周线、月线判断大周期方向，用于顺势交易、提升利润。
    - 月线：收盘在短期均线之上且趋势向上 -> 偏多
    - 周线：同上
    返回 'up' | 'down' | 'neutral'
    """
    if df_daily is None or len(df_daily) < 60:
        return 'neutral'
    weekly = _resample_to_weekly(df_daily)
    monthly = _resample_to_monthly(df_daily)
    if weekly is None or len(weekly) < 3:
        weekly_ok = False
    else:
        weekly_ok = True
        w_close = float(weekly['close'].iloc[-1])
        w_ma2 = float(weekly['close'].iloc[-3:].mean())
        weekly_up = w_close > w_ma2 and weekly['close'].iloc[-1] >= weekly['close'].iloc[-2]
    if monthly is None or len(monthly) < 2:
        monthly_ok = False
    else:
        monthly_ok = True
        m_close = float(monthly['close'].iloc[-1])
        m_ma2 = float(monthly['close'].iloc[-2:].mean())
        monthly_up = m_close > m_ma2 and (len(monthly) < 2 or monthly['close'].iloc[-1] >= monthly['close'].iloc[-2])
    if weekly_ok and monthly_ok:
        if weekly_up and monthly_up:
            return 'up'
        if not weekly_up and not monthly_up:
            return 'down'
    if monthly_ok:
        return 'up' if monthly_up else 'down'
    if weekly_ok:
        return 'up' if weekly_up else 'down'
    return 'neutral'


# ---------- 成交量配合（买卖点须结合量） ----------

def _volume_context(df: pd.DataFrame, recent_days: int = 5, avg_days: int = 20) -> Tuple[bool, bool, float]:
    """
    二买需缩量企稳，三买需放量突破（买卖点须结合量）。
    波段纪律：突破中枢时成交量需放大 30% 以上，缩量突破视为假信号。
    返回 (volume_ok_2buy, volume_ok_3buy, volume_ratio)。
    若 df 无成交量列则返回 (True, True, 1.0) 不因量能过滤。
    """
    vol_col = 'volume' if 'volume' in df.columns else ('成交量' if '成交量' in df.columns else None)
    if df is None or len(df) < avg_days or vol_col is None:
        return True, True, 1.0
    vol = df[vol_col].replace(0, np.nan).dropna()
    if len(vol) < avg_days:
        return True, True, 1.0
    vol = vol.astype(float)
    recent_vol = vol.iloc[-recent_days:].mean()
    avg_vol = vol.iloc[-avg_days:].mean()
    if avg_vol <= 0:
        return True, True, 1.0
    ratio = float(recent_vol / avg_vol)
    # 二买：缩量企稳，近期量不宜明显放大（<=1.35 倍均量）
    volume_ok_2buy = ratio <= 1.35
    # 三买：放量突破，突破时量能需放大 30% 以上（>=1.3），缩量突破大概率假信号
    volume_ok_3buy = ratio >= 1.3
    return volume_ok_2buy, volume_ok_3buy, ratio


def _indicator_context(df: pd.DataFrame) -> Tuple[bool, bool, bool, bool]:
    """
    缠论与 MACD/布林带/均线 的综合辅助（先以缠论结构为核心，再辅以指标验证）。
    返回 (macd_above_zero, boll_mid_ok, ma_support_ok, boll_squeeze)。
    """
    macd_above_zero = True
    boll_mid_ok = False
    ma_support_ok = False
    boll_squeeze = False
    if df is None or len(df) < 20:
        return macd_above_zero, boll_mid_ok, ma_support_ok, boll_squeeze

    # 1）MACD 零轴与防狼术：黄白线（DIF/DEA）长期在零轴下方则回避做多
    macd_col = 'MACD' if 'MACD' in df.columns else None
    sig_col = 'Signal' if 'Signal' in df.columns else None
    if macd_col and sig_col:
        recent = df[[macd_col, sig_col]].iloc[-20:].dropna()
        if len(recent) >= 10:
            # 近期多数时间在零轴下则防狼
            below_zero = (recent[macd_col] < 0) & (recent[sig_col] < 0)
            macd_above_zero = below_zero.sum() < len(recent) * 0.6  # 至少 40% 时间在零轴上方

    # 2）布林中轨：二买时价格回踩中轨企稳（当前 close 在中轨附近）
    bb_mid = 'BB_Middle' if 'BB_Middle' in df.columns else None
    if bb_mid:
        close = float(df['close'].iloc[-1])
        mid = float(df[bb_mid].iloc[-1])
        if mid > 0:
            ratio = close / mid
            boll_mid_ok = 0.98 <= ratio <= 1.03  # 回踩中轨附近企稳

    # 3）均线支撑/多头：MA20 支撑或 MA5>MA10>MA20
    if 'MA20' in df.columns and 'MA5' in df.columns and 'MA10' in df.columns:
        close = float(df['close'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma5 = float(df['MA5'].iloc[-1])
        ma10 = float(df['MA10'].iloc[-1])
        if ma20 > 0:
            ma_support_ok = (0.98 <= close / ma20 <= 1.03) or (ma5 > ma10 > ma20)

    # 4）布林收口：带宽近期缩小，中阴将结束，可辅助三买
    if bb_mid and 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        width = (df['BB_Upper'] - df['BB_Lower']) / df[bb_mid].replace(0, np.nan)
        width = width.dropna()
        if len(width) >= 10:
            cur_w = float(width.iloc[-1])
            avg_w = float(width.iloc[-10:].mean())
            boll_squeeze = avg_w > 0 and cur_w < avg_w * 0.92

    return macd_above_zero, boll_mid_ok, ma_support_ok, boll_squeeze


# ---------- 买卖点（三买三卖） ----------

def _bi_low(bi: Bi) -> float:
    """笔的低点"""
    return min(bi.start_price, bi.end_price)


def _bi_high(bi: Bi) -> float:
    """笔的高点"""
    return max(bi.start_price, bi.end_price)


def _classify_buy_sell(
    divergence: Optional[str],
    last_bi_direction: Optional[int],
    current_price: float,
    zhongshus: List[Zhongshu],
    bis: List[Bi]
) -> Tuple[Optional[str], str]:
    """
    按三买三卖理论识别买卖点（波段优先 2买/3买、2卖/3卖）。
    一买：底背驰引发，形成一笔上攻后确认（底背驰+当前为上涨笔）。
    二买：上升波段中，回踩不破中枢下沿 + 底背驰（回调确认，最安全）。
    三买：突破中枢后回踩不破上沿，无背驰（趋势延续，性价比高）。
    卖点对称：2卖=反弹不破+顶背驰，3卖=跌破后反抽不破。
    返回 (buy_sell_point, reason)。
    """
    if not bis:
        return None, "无笔"
    last_bi = bis[-1]
    z = zhongshus[-1] if zhongshus else None

    # 一买：底背驰 + 已形成一笔上攻（当前为上涨笔，确认转势）
    if divergence == 'bottom':
        if last_bi_direction == 1:
            return '1buy', "缠论一买确认：底背驰后一笔上攻"
        return '1buy', "缠论一买区域：底背驰（待回踩确认）"
    # 一卖：顶背驰
    if divergence == 'top':
        return '1sell', "缠论一卖区域：顶背驰（价格新高MACD未新高）"

    if z is None or len(bis) < 3:
        reason = "价格在中枢上方，偏多" if last_bi_direction == 1 and z else "无中枢或笔不足"
        return None, reason or "无明确买卖点"

    # 二买：回踩不破中枢下沿 + 底背驰（波段最安全买点，拒绝模糊信号）
    if last_bi_direction == -1:
        pullback_low = _bi_low(last_bi)
        if z.low <= pullback_low <= z.high and divergence == 'bottom':
            return '2buy', "缠论二买：回踩不破中枢下沿且底背驰"
    # 三买：上一笔向上离开中枢，当前笔向下回踩且不重新跌入中枢（回踩低点>中枢上沿）
    if len(bis) >= 2 and last_bi_direction == -1:
        prev_bi = bis[-2]
        if prev_bi.direction == 1:
            pullback_low = _bi_low(last_bi)
            if pullback_low > z.high:
                return '3buy', "缠论三买：突破中枢后回踩不破中枢"
    # 二卖：下降波段中，反弹不突破前中枢上沿 + 顶背驰（止盈核心）
    if last_bi_direction == 1:
        pullback_high = _bi_high(last_bi)
        if pullback_high < z.low and divergence == 'top':
            return '2sell', "缠论二卖：跌破中枢反抽不破且顶背驰"
    # 三卖：上一笔向下离开中枢，当前笔向上反抽且不突破中枢上沿（破位止损）
    if len(bis) >= 2 and last_bi_direction == 1:
        prev_bi = bis[-2]
        if prev_bi.direction == -1:
            pullback_high = _bi_high(last_bi)
            if pullback_high < z.high:
                return '3sell', "缠论三卖：跌破中枢反抽不破中枢上沿"

    if last_bi_direction == 1:
        if current_price > z.high:
            return None, "价格在中枢上方，偏多"
        if z.low <= current_price <= z.high:
            return None, "价格在中枢内"
    if last_bi_direction == -1 and current_price < z.low:
        return None, "价格在中枢下方，偏空"
    return None, "无明确买卖点"


# ---------- 主分析入口 ----------

def analyze_chanlun(
    df: pd.DataFrame,
    macd_col: str = 'MACD_Hist'
) -> ChanlunResult:
    """
    对行情 DataFrame 做缠论分析。
    df 需含 high, low, open, close；建议先 add_technical_indicators_to_df 以含 MACD_Hist。
    """
    if df is None or len(df) < 5:
        return ChanlunResult(reason="数据不足")
    df = df.copy()
    if 'date' not in df.columns and df.index.name is None:
        df['date'] = df.index
    klines = process_containment(df)
    if len(klines) < 3:
        return ChanlunResult(reason="合并K线不足")
    fenxings = find_fenxings(klines)
    bis = build_bis(fenxings, klines)
    zhongshus = find_zhongshus(bis)
    divergence = check_divergence(df, bis, macd_col=macd_col)
    last_bi_direction = bis[-1].direction if bis else None
    current_zhongshu = zhongshus[-1] if zhongshus else None
    current_price = float(df['close'].iloc[-1])
    buy_sell_point, reason = _classify_buy_sell(
        divergence, last_bi_direction, current_price, zhongshus, bis
    )
    volume_ok_2buy, volume_ok_3buy, volume_breakout_ratio = _volume_context(df)
    macd_above_zero, boll_mid_ok, ma_support_ok, boll_squeeze = _indicator_context(df)

    # 中枢转移：新中枢比前中枢上移=上升波段，下移=下降波段，否则盘整（波段不做）
    pivot_transfer = None
    if len(zhongshus) >= 2:
        z_prev, z_cur = zhongshus[-2], zhongshus[-1]
        if z_cur.low > z_prev.high:
            pivot_transfer = 'up'
        elif z_cur.high < z_prev.low:
            pivot_transfer = 'down'

    # 中枢幅度：(ZG-ZD)/mid*100，<2% 视为盘整，买卖点频繁失效则观望
    pivot_range_pct = 0.0
    if current_zhongshu:
        mid = (current_zhongshu.high + current_zhongshu.low) / 2
        if mid > 0:
            pivot_range_pct = (current_zhongshu.high - current_zhongshu.low) / mid * 100

    return ChanlunResult(
        fenxings=fenxings,
        bis=bis,
        zhongshus=zhongshus,
        divergence=divergence,
        buy_sell_point=buy_sell_point,
        last_bi_direction=last_bi_direction,
        current_zhongshu=current_zhongshu,
        reason=reason,
        volume_ok_2buy=volume_ok_2buy,
        volume_ok_3buy=volume_ok_3buy,
        pivot_transfer=pivot_transfer,
        pivot_range_pct=pivot_range_pct,
        volume_breakout_ratio=volume_breakout_ratio,
        macd_above_zero=macd_above_zero,
        boll_mid_ok=boll_mid_ok,
        ma_support_ok=ma_support_ok,
        boll_squeeze=boll_squeeze,
    )


# ---------- 与选股池/每日决策兼容的信号接口 ----------

def chanlun_signal(
    df: pd.DataFrame,
    has_position: bool = False,
    entry_price: Optional[float] = None,
    holding_days: int = 0,
    highest_price: float = 0,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.12,
    trailing_stop_pct: float = 0.05,
    trailing_activate_pct: float = 0.06,
) -> Tuple[str, float, str]:
    """
    基于缠论分析输出与 ai_daily_decision 相同格式的 (action, confidence, reason)。
    - action: 'BUY' | 'SELL' | 'REDUCE' | 'HOLD'
    - confidence: 0~1
    - reason: 文字说明
    可用于选股池在 strategy_type='chanlun' 时调用。
    注意：仅高质量信号触发买卖，避免频繁交易（见 MIN_CONFIDENCE_*）。
    """
    # 信号质量门槛：低于门槛不触发买卖，避免频繁交易
    MIN_CONFIDENCE_BUY = 0.72
    MIN_CONFIDENCE_SELL = 0.70
    MIN_CONFIDENCE_REDUCE = 0.68

    if df is None or len(df) < 20:
        return 'HOLD', 0.5, "缠论：数据不足"
    # 大周期：周线/月线方向，顺势做多、逆势不做多，以提升利润
    big_trend = get_big_trend(df)
    res = analyze_chanlun(df)
    price = float(df['close'].iloc[-1])
    # 持仓时先检查止损止盈（波段纪律：单波段亏损不超过总资金 3% 由上层仓位控制）
    if has_position and entry_price and entry_price > 0:
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct <= -stop_loss_pct * 100:
            return 'SELL', 0.9, f"缠论：止损，亏损{profit_pct:.1f}%"
        if highest_price > 0 and highest_price > entry_price * (1 + trailing_activate_pct):
            drawdown = (highest_price - price) / highest_price * 100
            if drawdown >= trailing_stop_pct * 100:
                return 'SELL', 0.85, f"缠论：移动止盈，自高点回撤{drawdown:.1f}%"
        if take_profit_pct < 10 and profit_pct >= take_profit_pct * 100:
            return 'SELL', 0.85, f"缠论：固定止盈，盈利{profit_pct:.1f}%"

    # 波段纪律：中枢过窄（<2%）时买卖点频繁失效，暂停操作
    if res.current_zhongshu and res.pivot_range_pct > 0 and res.pivot_range_pct < 2.0:
        return 'HOLD', 0.5, "缠论：中枢过窄盘整，观望"
    # 无中枢转移则为盘整，暂不做波段（不开新仓）
    if not has_position and res.buy_sell_point in ('2buy', '3buy') and res.pivot_transfer != 'up':
        return 'HOLD', 0.5, "缠论：无中枢转移盘整，不做波段"
    if not has_position and res.pivot_transfer == 'down':
        return 'HOLD', 0.5, "缠论：下降波段，不做多"

    # 大周期过滤：周线/月线向下时不做多，避免逆势亏损
    if not has_position and big_trend == 'down':
        return 'HOLD', 0.5, "缠论：大周期(周/月)向下，不做多"

    # 防狼术：MACD 黄白线长期在零轴下方则回避做多，减少被大幅下跌侵犯
    if not has_position and not res.macd_above_zero:
        return 'HOLD', 0.5, "缠论：MACD零轴下防狼术，不做多"

    # 买卖点优先级：高质量信号为主，适度放宽以便有交易（如 600875 等个股）
    # 大周期向上时顺势做多，买点置信度略升；布林/均线辅助验证可再加分
    boost = 0.02 if big_trend == 'up' else 0.0
    indicator_boost = 0.02 if (res.boll_mid_ok or res.ma_support_ok) else 0.0  # 二买回踩中轨/均线支撑
    if res.buy_sell_point == '1buy' and not has_position:
        conf = 0.78 + boost if res.last_bi_direction == 1 else 0.70 + boost
        if res.ma_support_ok:
            conf = min(0.92, conf + 0.02)
        reason_extra = ("（大周期向上）" if big_trend == 'up' else "") + ("（均线支撑）" if res.ma_support_ok else "")
        if res.last_bi_direction == 1:
            return 'BUY', min(0.92, conf), res.reason + reason_extra
        return 'BUY', min(0.88, conf), res.reason + "（一买待确认，轻仓）" + reason_extra
    if res.buy_sell_point == '2buy' and not has_position:
        conf = 0.78 + boost + indicator_boost if res.volume_ok_2buy else 0.68 + boost + indicator_boost
        extra = "（缩量企稳）" if res.volume_ok_2buy else "（量能未缩量，降级）"
        if res.boll_mid_ok:
            extra += "（回踩布林中轨）"
        if res.ma_support_ok:
            extra += "（均线支撑）"
        return 'BUY', min(0.92, conf), res.reason + extra
    if res.buy_sell_point == '3buy' and not has_position:
        squeeze_boost = 0.02 if res.boll_squeeze else 0.0  # 布林收口辅助三买
        conf = 0.78 + boost + squeeze_boost if res.volume_ok_3buy else 0.68 + boost + squeeze_boost
        extra = "（放量突破）" if res.volume_ok_3buy else "（量能未放量，降级）"
        if res.boll_squeeze:
            extra += "（布林收口）"
        return 'BUY', min(0.92, conf), res.reason + extra
    if res.buy_sell_point == '1sell' and has_position:
        return 'SELL', 0.76, res.reason
    if res.buy_sell_point in ('2sell', '3sell') and has_position:
        return 'SELL', 0.74, res.reason
    if res.divergence == 'bottom' and not has_position:
        return 'BUY', min(0.88, 0.70 + boost), "缠论：底背驰，潜在一买"
    if res.divergence == 'top' and has_position:
        return 'SELL', 0.72, res.reason
    n_bis = len(res.bis) if res.bis else 0
    if n_bis >= 2 and res.last_bi_direction == -1 and has_position:
        # 大周期向上时少减仓，持仓吃利润
        if big_trend == 'up':
            return 'HOLD', 0.55, "缠论：大周期向上，下跌笔持仓观望"
        return 'REDUCE', 0.65, "缠论：下跌笔中，建议减仓"
    if res.last_bi_direction == 1 and not has_position:
        return 'BUY', min(0.88, 0.65 + boost), "缠论：上涨笔中，轻仓试多"
    if res.last_bi_direction == -1 and has_position:
        if big_trend == 'up':
            return 'HOLD', 0.55, "缠论：大周期向上，下跌笔持仓观望"
        return 'REDUCE', 0.62, "缠论：下跌笔，减仓"
    return 'HOLD', 0.5, res.reason or "缠论：无明确买卖点"
