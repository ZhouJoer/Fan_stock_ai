"""
选股池管理模块

提供选股池配置、AI信号自动交易、持仓管理和回测功能

功能特性：
1. 选股池配置 - 用户可自定义股票池、初始资金、风险偏好
2. AI信号生成 - 对池内每只股票生成买入/卖出信号
3. 持仓比例分配 - 根据信号强度智能分配仓位
4. 组合回测 - 支持多股票组合的历史回测
5. 风险控制 - 单股最大仓位、组合止损等
"""
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import akshare as ak
from tools.stock_data import get_stock_data, get_stock_code_from_name
from tools.technical_indicators import add_technical_indicators_to_df
from tools.risk_control import audit_risk, RiskAuditInput, RiskAuditResult, get_recent_daily_returns_from_equity
from modules.strategy_config import (
    get_strategy_config, 
    analyze_market_state, 
    get_adaptive_strategy_config,
    generate_ai_audit_signal,
    SignalStrengthEvaluator,
    DynamicRiskController,
    AI_AUDIT_STRATEGY_PARAMS,
    SIGNAL_STRENGTH_THRESHOLDS,
    RISK_PREFERENCES,
    SignalType,
    MarketState,
    VOLATILITY_WEIGHT_MAP,
    LIQUIDITY_THRESHOLDS,
    calculate_liquidity_score,
    calculate_volatility_level,
)
# ==================== 枚举类型 ====================

class PositionAction(Enum):
    """持仓操作类型"""
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"
    INCREASE = "加仓"
    DECREASE = "减仓"


class AllocationMethod(Enum):
    """仓位分配方法"""
    EQUAL_WEIGHT = "等权重"           # 等权重分配
    SIGNAL_STRENGTH = "信号强度"      # 按信号强度分配
    RISK_PARITY = "风险平价"          # 按风险平价分配
    KELLY = "凯利公式"                # 按凯利公式分配（期望收益/方差）
    MARKET_CAP = "市值加权"           # 按市值加权
    CUSTOM = "自定义"                 # 自定义权重


# 凯利公式：波动率等级 -> 日收益率标准差近似（用于计算 Kelly 权重）
KELLY_VOLATILITY_SIGMA = {"低": 0.01, "中": 0.02, "高": 0.03}


# ==================== 数据类定义 ====================

@dataclass
class StockPosition:
    """单只股票持仓信息"""
    stock_code: str
    stock_name: str = ""
    shares: int = 0                    # 持仓股数
    avg_cost: float = 0.0              # 平均成本
    current_price: float = 0.0         # 当前价格
    market_value: float = 0.0          # 市值
    profit_loss: float = 0.0           # 盈亏金额
    profit_loss_pct: float = 0.0       # 盈亏比例
    weight: float = 0.0                # 持仓权重
    target_weight: float = 0.0         # 目标权重
    signal_strength: float = 0.0       # 信号强度
    holding_days: int = 0              # 持仓天数
    entry_date: str = ""               # 建仓日期
    last_signal: str = ""              # 最新信号
    
    def update_market_value(self, price: float):
        """更新市值和盈亏"""
        self.current_price = price
        self.market_value = self.shares * price
        if self.shares > 0 and self.avg_cost > 0:
            self.profit_loss = self.market_value - self.shares * self.avg_cost
            self.profit_loss_pct = (price - self.avg_cost) / self.avg_cost * 100


@dataclass
class PoolConfig:
    """
    选股池配置
    
    止损/止盈参数默认从 strategy_config.RISK_PREFERENCES 获取，
    确保与策略配置一致性
    """
    name: str = "默认选股池"
    stocks: List[str] = field(default_factory=list)       # 股票代码列表
    initial_capital: float = 1000000.0                     # 初始资金（默认100万）
    strategy_type: str = "adaptive"                        # 策略类型
    risk_preference: str = "balanced"                      # 风险偏好
    allocation_method: str = "signal_strength"             # 仓位分配方法
    max_position_per_stock: float = 0.3                    # 单只股票最大仓位
    min_position_per_stock: float = 0.05                   # 单只股票最小仓位
    max_total_position: float = 0.9                        # 最大总仓位
    min_cash_ratio: float = 0.1                            # 最小现金比例
    rebalance_threshold: float = 0.05                      # 再平衡阈值
    signal_threshold: float = None                         # 信号阈值（None时从SIGNAL_STRENGTH_THRESHOLDS获取）
    stop_loss_pct: float = None                            # 组合止损比例（None时从RISK_PREFERENCES获取）
    take_profit_pct: float = None                          # 组合止盈比例（None时从RISK_PREFERENCES获取）
    single_stop_loss_pct: float = None                     # 单股止损比例
    single_take_profit_pct: float = None                   # 单股止盈比例
    custom_weights: Dict[str, float] = field(default_factory=dict)  # 自定义权重
    # 选股与动态配置（计划：选股与动态配置集成）
    universe_source: str = "manual"                        # 股票来源: manual | index | industry
    universe_index: str = ""                               # 指数代码（如 000300），universe_source=index 时使用
    industry_list: Optional[List[str]] = None               # 行业名称列表，universe_source=industry 时使用；空为全行业
    leaders_per_industry: int = 1                           # 每行业取龙头数量，universe_source=industry 时使用
    selection_mode: str = "none"                            # 选股模式: none | factor_top_n
    selection_top_n: int = 10                               # 因子选股取前 N 只
    selection_interval: int = 0                              # 多少交易日重选一次，0 表示仅初选一次
    score_weights: Optional[Dict[str, float]] = None        # 选股因子权重，如 {"momentum": 0.4, "quality": 0.35}
    factor_set: str = "hybrid"                               # 因子集合: style | trading | hybrid（风格与估值|情绪与交易）
    strategy_meta: Dict[str, Any] = field(default_factory=dict)   # 策略元信息
    factor_profile: Dict[str, Any] = field(default_factory=dict)  # 因子评估快照

    def __post_init__(self):
        """初始化后从共享配置获取默认参数"""
        risk_config = RISK_PREFERENCES.get(self.risk_preference, RISK_PREFERENCES["balanced"])
        
        if self.signal_threshold is None:
            self.signal_threshold = SIGNAL_STRENGTH_THRESHOLDS.get("弱信号", 0.4)
        if self.stop_loss_pct is None:
            self.stop_loss_pct = risk_config.get("stop_loss_pct", 0.05) + 0.03  # 组合止损略宽松
        if self.take_profit_pct is None:
            base_tp = risk_config.get("take_profit_pct", 0.12)
            # 如果是激进策略（无固定止盈），组合止盈设为25%
            self.take_profit_pct = 0.25 if base_tp > 100 else base_tp * 2
        if self.single_stop_loss_pct is None:
            self.single_stop_loss_pct = risk_config.get("stop_loss_pct", 0.05)
        if self.single_take_profit_pct is None:
            base_tp = risk_config.get("take_profit_pct", 0.12)
            # 如果是激进策略，单股止盈设为15%
            self.single_take_profit_pct = 0.15 if base_tp > 100 else base_tp


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    stock_code: str
    stock_name: str
    action: str
    shares: int
    price: float
    amount: float
    reason: str
    signal_strength: float
    market_state: str = ""
    before_weight: float = 0.0
    after_weight: float = 0.0
    # 盈亏信息（卖出时记录）
    avg_cost: float = 0.0          # 持仓均价
    profit_amount: float = 0.0     # 盈亏金额
    profit_pct: float = 0.0        # 盈亏比例


@dataclass
class DailySnapshot:
    """每日快照"""
    date: str
    total_value: float
    cash: float
    position_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    positions: Dict[str, Dict] = field(default_factory=dict)
    signals: Dict[str, Dict] = field(default_factory=dict)


# ==================== 股票质量评估 ====================

def evaluate_stock_quality(stock_code: str, df: pd.DataFrame, signal: Dict[str, Any], 
                           min_signal_strength: float = 0.4) -> Tuple[float, bool, str]:
    """
    评估股票质量，判断是否适合买入
    
    参数:
        stock_code: 股票代码
        df: 股票数据DataFrame（包含技术指标）
        signal: 股票信号字典
        min_signal_strength: 最小信号强度阈值
    
    返回:
        (质量评分0-1, 是否可买入, 评估原因)
    """
    if df.empty or len(df) < 20:
        return 0.0, False, "数据不足"
    
    quality_score = 0.0
    reasons = []
    
    # 1. 信号强度评估（40%权重）
    signal_strength = signal.get('signal_strength', 0.0)
    signal_type = signal.get('signal_type', 'Neutral')
    
    if signal_type != 'LongOpen':
        return 0.0, False, f"信号类型为{signal_type}，非买入信号"
    
    if signal_strength < min_signal_strength:
        return 0.0, False, f"信号强度{signal_strength:.2f}低于阈值{min_signal_strength}"
    
    signal_score = min(1.0, signal_strength / 0.8)  # 归一化到0-1
    quality_score += signal_score * 0.4
    reasons.append(f"信号强度{signal_strength:.2f}")
    
    # 2. 技术指标健康度评估（30%权重）
    latest = df.iloc[-1]
    ma5 = latest.get('MA5', latest['close'])
    ma10 = latest.get('MA10', latest['close'])
    ma20 = latest.get('MA20', latest['close'])
    ma60 = latest.get('MA60', ma20)
    rsi = latest.get('RSI', 50)
    macd_hist = latest.get('MACD_Hist', 0)
    price = latest['close']
    
    tech_score = 0.0
    
    # 均线排列（多头排列加分）
    if ma5 > ma10 > ma20:
        tech_score += 0.4
        reasons.append("均线多头排列")
    elif ma5 > ma20 and ma10 > ma20:
        tech_score += 0.2
        reasons.append("部分均线多头")
    
    # 价格位置（在MA20以上加分）
    if price > ma20:
        tech_score += 0.2
        reasons.append("价格站上MA20")
    
    # RSI评估（30-70之间较好）
    if 30 <= rsi <= 70:
        tech_score += 0.2
        reasons.append(f"RSI合理({rsi:.1f})")
    elif rsi < 30 or rsi > 70:
        tech_score -= 0.1  # 超买超卖扣分
    
    # MACD评估
    if macd_hist > 0:
        tech_score += 0.2
        reasons.append("MACD金叉")
    
    tech_score = max(0.0, min(1.0, tech_score))  # 限制在0-1
    quality_score += tech_score * 0.3
    
    # 3. 历史表现评估（20%权重）
    if len(df) >= 20:
        # 近期涨跌幅（最近20天）
        recent_return = (df.iloc[-1]['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close'] * 100
        
        # 波动率（最近20天）
        returns = df['close'].pct_change().tail(20).dropna()
        volatility = returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 0
        
        # 收益率越高越好，波动率越低越好
        return_score = min(1.0, max(0.0, (recent_return + 10) / 30))  # -10%到20%映射到0-1
        volatility_score = max(0.0, min(1.0, 1 - volatility / 50))  # 波动率越低越好
        
        perf_score = (return_score * 0.6 + volatility_score * 0.4)
        quality_score += perf_score * 0.2
        
        if recent_return > 5:
            reasons.append(f"近期涨幅{recent_return:.1f}%")
        elif recent_return < -10:
            reasons.append(f"近期跌幅{recent_return:.1f}%")
    
    # 4. 流动性评估（10%权重）
    volume = latest.get('volume', 0)
    if len(df) >= 20:
        avg_volume = df['volume'].tail(20).mean()
        if volume > 0 and avg_volume > 0:
            volume_ratio = volume / avg_volume
            liquidity_score = min(1.0, volume_ratio / 2.0)  # 成交量是均值的2倍以上得满分
            quality_score += liquidity_score * 0.1
            
            if volume_ratio > 1.5:
                reasons.append("成交量放大")
    
    # 综合评分
    quality_score = max(0.0, min(1.0, quality_score))
    
    # 判断是否可买入（质量评分 >= 0.5 且信号强度达标）
    can_buy = quality_score >= 0.5 and signal_strength >= min_signal_strength
    
    reason_text = ", ".join(reasons[:3]) if reasons else "基础评估"
    
    return quality_score, can_buy, reason_text


# ==================== 选股逻辑：排除 ST/北交所、市值范围 ====================

def _is_bse_code(code: str) -> bool:
    """北交所股票：43/82/83/87/88/920 开头（6 位）。"""
    c = str(code).strip().zfill(6)
    if len(c) < 6:
        return False
    return c.startswith(("43", "82", "83", "87", "88", "920"))


def _is_st_name(name: str) -> bool:
    """是否 ST、*ST 等风险警示名称。"""
    if not name or not isinstance(name, str):
        return False
    n = name.strip().upper()
    return "ST" in n or "*ST" in n or n.startswith("S*ST") or n.startswith("SST")


def filter_stock_universe(
    codes: List[str],
    cap_scope: str = "none",
    small_cap_threshold_billion: float = 30.0,
    code_to_name: Optional[Dict[str, str]] = None,
    code_to_mv: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    选股过滤：排除 ST、*ST、北交所；用户可选市值适用范围。
    
    参数:
        codes: 候选股票代码列表
        cap_scope: "none" 不按市值过滤；"only_small_cap" 仅保留小市值（市值 < 阈值）；"exclude_small_cap" 排除小市值（市值 >= 阈值）
        small_cap_threshold_billion: 小市值阈值（亿元）
        code_to_name: 代码->名称，用于过滤 ST；若为 None 且需要过滤 ST 则内部拉取
        code_to_mv: 代码->总市值（元），若为 None 且 cap_scope 非 none 则内部拉取
    
    返回:
        过滤后的代码列表
    """
    if not codes:
        return []
    out = []
    need_name = True
    need_mv = cap_scope and cap_scope != "none"
    if need_mv or need_name:
        if code_to_name is None or (need_mv and code_to_mv is None):
            try:
                spot = ak.stock_zh_a_spot_em()
                if spot is not None and not spot.empty:
                    if "代码" in spot.columns:
                        spot = spot.copy()
                        spot["_code"] = spot["代码"].astype(str).str.strip().str.zfill(6)
                        if code_to_name is None and "名称" in spot.columns:
                            code_to_name = dict(zip(spot["_code"], spot["名称"].astype(str)))
                        if need_mv and code_to_mv is None and "总市值" in spot.columns:
                            code_to_mv = dict(
                                zip(
                                    spot["_code"],
                                    pd.to_numeric(spot["总市值"], errors="coerce").fillna(0).tolist(),
                                )
                            )
            except Exception:
                pass
        if code_to_name is None:
            code_to_name = {}
        if code_to_mv is None:
            code_to_mv = {}
    threshold = float(small_cap_threshold_billion) * 1e8
    for c in codes:
        code = str(c).strip().zfill(6)
        if len(code) != 6:
            continue
        if _is_bse_code(code):
            continue
        if need_name:
            name = code_to_name.get(code) or code_to_name.get(c, "")
            if _is_st_name(name):
                continue
        if cap_scope == "only_small_cap":
            mv = code_to_mv.get(code) or code_to_mv.get(c, 0)
            if mv >= threshold:
                continue
        elif cap_scope == "exclude_small_cap":
            mv = code_to_mv.get(code) or code_to_mv.get(c, 0)
            if mv < threshold:
                continue
        out.append(code)
    return out


# ==================== 行业/指数选股（分行业龙头） ====================

def list_industry_names() -> List[str]:
    """
    获取东方财富行业板块名称列表
    
    返回:
        行业名称列表
    """
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty and "板块名称" in df.columns:
            return df["板块名称"].dropna().astype(str).tolist()
    except Exception as e:
        print(f"[选股] 获取行业列表失败: {e}")
    return []


def get_industry_constituents(industry_name: str) -> pd.DataFrame:
    """
    获取指定行业的成分股（东方财富）
    
    参数:
        industry_name: 行业名称（如 "银行"、"电力"）
    
    返回:
        成分股 DataFrame，含列 代码、名称、成交额 等
    """
    try:
        df = ak.stock_board_industry_cons_em(symbol=industry_name)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[选股] 获取行业 {industry_name} 成分失败: {e}")
    return pd.DataFrame()


def _select_industry_leaders_by_cons_only(
    industries: List[str],
    leaders_per_industry: int,
) -> Tuple[List[str], Dict[str, str]]:
    """
    仅用行业成分表选龙头：按成分表内「成交额」排序取前 k 名。
    当全市场行情 stock_zh_a_spot_em 不可用时使用此降级方案。
    """
    def _norm_code(c):
        s = str(c).strip()
        if "." in s:
            s = s.split(".")[0]
        return s[:6] if len(s) >= 6 else s.zfill(6)

    code_to_industry: Dict[str, str] = {}
    all_codes: List[str] = []

    for ind_name in industries:
        try:
            cons = get_industry_constituents(ind_name)
            if cons.empty or "代码" not in cons.columns:
                continue
            cons = cons.copy()
            cons["_code"] = cons["代码"].apply(_norm_code)
            # 按成交额排序（东方财富行业成分表含「成交额」列）
            if "成交额" in cons.columns:
                cons["_sort"] = pd.to_numeric(cons["成交额"], errors="coerce").fillna(0)
            else:
                cons["_sort"] = 0
            cons = cons.sort_values("_sort", ascending=False)
            top = cons.head(leaders_per_industry)
            for _, row in top.iterrows():
                code = row["_code"]
                if code and code not in code_to_industry:
                    code_to_industry[code] = ind_name
                    all_codes.append(code)
        except Exception as e:
            print(f"[选股] 行业 {ind_name} 取龙头失败: {e}")

    return all_codes, code_to_industry


def select_industry_leaders(
    industry_list: Optional[List[str]] = None,
    leaders_per_industry: int = 1,
    cap_scope: str = "none",
    small_cap_threshold_billion: float = 30.0,
) -> Tuple[List[str], Dict[str, str]]:
    """
    分行业选龙头：每行业按总市值排序取前 k 名，合并为初选池；并做选股过滤（排除 ST/北交所、可选市值范围）。
    
    参数:
        industry_list: 行业名称列表，空或 None 表示全行业
        leaders_per_industry: 每行业取龙头数量
        cap_scope: "none" | "only_small_cap" | "exclude_small_cap"
        small_cap_threshold_billion: 小市值阈值（亿元）
    
    返回:
        (股票代码列表, 代码->行业名称 映射)
    """
    industries = industry_list if industry_list else list_industry_names()
    if not industries:
        return [], {}

    def _norm_code(c):
        s = str(c).strip()
        if "." in s:
            s = s.split(".")[0]
        return s[:6] if len(s) >= 6 else s.zfill(6)

    spot_df = None
    for attempt in range(2):
        try:
            spot_df = ak.stock_zh_a_spot_em()
            if spot_df is not None and not spot_df.empty and "代码" in spot_df.columns:
                break
        except Exception as e:
            print(f"[选股] 获取全市场行情失败(尝试 {attempt + 1}/2): {e}")
            if attempt == 0:
                time.sleep(1.0)
            spot_df = None

    if spot_df is None or spot_df.empty or "代码" not in spot_df.columns:
        print("[选股] 全市场行情不可用，改用行业成分表按成交额选龙头")
        all_codes, code_to_industry = _select_industry_leaders_by_cons_only(industries, leaders_per_industry)
        all_codes = filter_stock_universe(all_codes, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_threshold_billion)
        code_to_industry = {c: code_to_industry[c] for c in all_codes if c in code_to_industry}
        return all_codes, code_to_industry

    spot_df = spot_df.copy()
    spot_df["_code"] = spot_df["代码"].apply(_norm_code)
    if "总市值" not in spot_df.columns:
        spot_df["_mv"] = spot_df.get("成交额", pd.Series(0.0, index=spot_df.index))
    else:
        spot_df["_mv"] = pd.to_numeric(spot_df["总市值"], errors="coerce").fillna(0)

    code_to_industry = {}
    all_codes = []

    for ind_name in industries:
        try:
            cons = get_industry_constituents(ind_name)
            if cons.empty or "代码" not in cons.columns:
                continue
            cons = cons.copy()
            cons["_code"] = cons["代码"].apply(_norm_code)
            merged = cons.merge(
                spot_df[["_code", "_mv"]],
                on="_code",
                how="left",
            )
            merged["_mv"] = pd.to_numeric(merged["_mv"], errors="coerce").fillna(0)
            merged = merged.sort_values("_mv", ascending=False)
            top = merged.head(leaders_per_industry)
            for _, row in top.iterrows():
                code = row["_code"]
                if code and code not in code_to_industry:
                    code_to_industry[code] = ind_name
                    all_codes.append(code)
        except Exception as e:
            print(f"[选股] 行业 {ind_name} 取龙头失败: {e}")

    if not all_codes:
        print("[选股] 按总市值未选出任何股票，改用行业成分表按成交额选龙头")
        all_codes, code_to_industry = _select_industry_leaders_by_cons_only(industries, leaders_per_industry)
        all_codes = filter_stock_universe(all_codes, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_threshold_billion)
        code_to_industry = {c: code_to_industry[c] for c in all_codes if c in code_to_industry}
        return all_codes, code_to_industry

    all_codes = filter_stock_universe(all_codes, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_threshold_billion)
    code_to_industry = {c: code_to_industry[c] for c in all_codes if c in code_to_industry}
    return all_codes, code_to_industry


def get_index_constituents(
    index_code: str,
    cap_scope: str = "none",
    small_cap_threshold_billion: float = 30.0,
) -> List[str]:
    """
    获取指数当前成分股代码列表（akshare），并做选股过滤：排除 ST/*ST/北交所，可选市值范围。
    
    index_code 如 "000300"（沪深300）、"000016"（上证50）。
    cap_scope: "none" | "only_small_cap" | "exclude_small_cap"
    
    返回:
        6 位股票代码列表
    """
    symbol = str(index_code).strip()
    codes: List[str] = []
    for fn, sym in [
        (ak.index_stock_cons_csindex, symbol),
        (ak.index_stock_cons, symbol),
    ]:
        try:
            df = fn(symbol=sym)
            if df is not None and not df.empty:
                col = None
                for c in ("品种代码", "成分券代码", "成分代码", "code", "代码"):
                    if c in df.columns:
                        col = c
                        break
                if col is None and len(df.columns) > 0:
                    col = df.columns[0]
                if col:
                    codes = [c for c in df[col].astype(str).str.replace(r"\D", "", regex=True).str[-6:].tolist() if len(c) == 6]
                    break
        except Exception:
            continue
    if not codes:
        try:
            print(f"[选股] 获取指数 {index_code} 成分失败")
        except Exception:
            pass
        return []
    return filter_stock_universe(codes, cap_scope=cap_scope, small_cap_threshold_billion=small_cap_threshold_billion)


# ==================== 选股池管理器 ====================

class StockPoolManager:
    """
    选股池管理器
    
    管理选股池的股票列表、持仓、交易和回测
    """
    
    def __init__(self, config: PoolConfig = None):
        """
        初始化选股池管理器
        
        参数:
            config: 选股池配置，如果为None则使用默认配置
        """
        self.config = config or PoolConfig()
        self.positions: Dict[str, StockPosition] = {}  # 当前持仓
        self.cash: float = self.config.initial_capital  # 现金
        self.trades: List[TradeRecord] = []  # 交易记录
        self.daily_snapshots: List[DailySnapshot] = []  # 每日快照
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}  # 股票数据缓存
        
    @property
    def total_value(self) -> float:
        """组合总价值"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def position_value(self) -> float:
        """持仓市值"""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def cash_ratio(self) -> float:
        """现金比例"""
        total = self.total_value
        return self.cash / total if total > 0 else 1.0
    
    def add_stock(self, stock_code: str, target_weight: float = 0.0):
        """添加股票到选股池"""
        code = get_stock_code_from_name(stock_code)
        if code not in self.config.stocks:
            self.config.stocks.append(code)
        if target_weight > 0:
            self.config.custom_weights[code] = target_weight
        print(f"[选股池] 添加股票: {code}，目标权重: {target_weight*100:.1f}%")
    
    def remove_stock(self, stock_code: str):
        """从选股池移除股票"""
        code = get_stock_code_from_name(stock_code)
        if code in self.config.stocks:
            self.config.stocks.remove(code)
        if code in self.config.custom_weights:
            del self.config.custom_weights[code]
        print(f"[选股池] 移除股票: {code}")
    
    def set_weights(self, weights: Dict[str, float]):
        """设置自定义权重"""
        total = sum(weights.values())
        if total > 1.0:
            # 归一化
            weights = {k: v/total for k, v in weights.items()}
        self.config.custom_weights = weights
        self.config.allocation_method = "custom"
        print(f"[选股池] 设置自定义权重: {weights}")
    
    def generate_signals(self, df_dict: Dict[str, pd.DataFrame], 
                         use_ai: bool = False, llm=None) -> Dict[str, Dict]:
        """
        为选股池中的所有股票生成交易信号
        
        参数:
            df_dict: 股票代码到数据DataFrame的映射
            use_ai: 是否使用AI深度分析
            llm: LLM实例
        
        返回:
            股票代码到信号的映射
        """
        signals = {}
        
        for stock_code in self.config.stocks:
            if stock_code not in df_dict:
                print(f"[选股池] 警告: {stock_code} 无数据，跳过")
                continue
            
            df = df_dict[stock_code]
            if len(df) < 60:
                print(f"[选股池] 警告: {stock_code} 数据不足，跳过")
                continue
            
            try:
                # 生成AI审计友好的信号
                signal = generate_ai_audit_signal(
                    df,
                    risk_preference=self.config.risk_preference,
                    use_ai=use_ai,
                    llm=llm
                )
                
                signals[stock_code] = {
                    'signal_type': signal['SignalType'],
                    'signal_strength': signal['SignalStrength']['总分'],
                    'strength_level': signal['SignalStrength']['强度等级'],
                    'market_state': signal['MarketState'],
                    'strategy_mode': signal['StrategyMode'],
                    'risk_assessment': signal['RiskAssessment'],
                    'suggestion': signal['TradingSuggestion'],
                    'explanation': signal['Explanation'],
                    'confidence': signal['ConfidenceLevel'],
                    'full_signal': signal
                }
                
            except Exception as e:
                print(f"[选股池] 生成 {stock_code} 信号失败: {e}")
                signals[stock_code] = {
                    'signal_type': 'Neutral',
                    'signal_strength': 0.0,
                    'strength_level': '无效',
                    'error': str(e)
                }
        
        return signals
    
    def calculate_target_weights(self, signals: Dict[str, Dict], 
                                 df_dict: Dict[str, pd.DataFrame] = None) -> Dict[str, float]:
        """
        根据信号计算目标权重（过滤烂股）
        
        参数:
            signals: 股票信号字典
            df_dict: 股票数据字典（用于质量评估，可选）
        
        返回:
            股票代码到目标权重的映射
        """
        method = self.config.allocation_method
        target_weights = {}
        
        # 质量过滤：过滤掉不符合买入条件的股票
        filtered_signals = {}
        filtered_stocks = []
        
        for code, sig in signals.items():
            signal_type = sig.get('signal_type', 'Neutral')
            signal_strength = sig.get('signal_strength', 0.0)
            
            # 基本过滤：必须是买入信号且强度达标
            if signal_type != 'LongOpen' or signal_strength < self.config.signal_threshold:
                continue
            
            # 如果有数据字典，进行质量评估（缠论策略不做烂股过滤）
            if self.config.strategy_type != 'chanlun' and df_dict and code in df_dict:
                quality_score, can_buy, reason = evaluate_stock_quality(
                    code, df_dict[code], sig, self.config.signal_threshold
                )
                if not can_buy:
                    print(f"[选股池] 过滤烂股 {code}: {reason} (质量评分{quality_score:.2f})")
                    filtered_stocks.append(code)
                    continue
            
            filtered_signals[code] = sig
        
        if filtered_stocks:
            print(f"[选股池] 共过滤 {len(filtered_stocks)} 只不符合条件的股票: {filtered_stocks}")
        
        if method == "custom" and self.config.custom_weights:
            # 使用自定义权重（但也要过滤）
            for code, weight in self.config.custom_weights.items():
                if code in filtered_signals:
                    target_weights[code] = weight
            return target_weights
        
        elif method == "equal_weight":
            # 等权重分配
            n_stocks = len(filtered_signals)
            if n_stocks > 0:
                weight = min(self.config.max_total_position / n_stocks, 
                            self.config.max_position_per_stock)
                for code in filtered_signals.keys():
                    target_weights[code] = weight
        
        elif method == "signal_strength":
            # 按信号强度分配（使用过滤后的信号）
            valid_signals = filtered_signals
            
            if valid_signals:
                total_strength = sum(abs(s['signal_strength']) for s in valid_signals.values())
                
                if total_strength > 0:
                    for code, sig in valid_signals.items():
                        raw_weight = abs(sig['signal_strength']) / total_strength
                        # 应用仓位限制
                        weight = raw_weight * self.config.max_total_position
                        weight = min(weight, self.config.max_position_per_stock)
                        weight = max(weight, self.config.min_position_per_stock)
                        target_weights[code] = weight
        
        elif method == "risk_parity":
            # 风险平价分配（使用过滤后的信号）
            valid_signals = filtered_signals
            
            if valid_signals:
                # 从风险评估中获取波动率信息，使用共享映射
                risk_scores = {}
                for code, sig in valid_signals.items():
                    risk = sig.get('risk_assessment', {})
                    vol_level = risk.get('波动率等级', '中')
                    # 使用共享的波动率权重映射
                    risk_scores[code] = VOLATILITY_WEIGHT_MAP.get(vol_level, 1.0)
                
                total_score = sum(risk_scores.values())
                if total_score > 0:
                    for code in valid_signals:
                        raw_weight = risk_scores[code] / total_score
                        weight = raw_weight * self.config.max_total_position
                        weight = min(weight, self.config.max_position_per_stock)
                        weight = max(weight, self.config.min_position_per_stock)
                        target_weights[code] = weight

        elif method == "kelly":
            # 凯利公式：权重 ∝ max(0, 期望收益/方差)。有信号时若 Kelly 全为 0 则回退等权，避免空仓率过高
            valid_signals = filtered_signals
            if valid_signals and df_dict:
                kelly_raw = {}
                for code, sig in valid_signals.items():
                    if code not in df_dict or df_dict[code] is None or len(df_dict[code]) < 20:
                        vol_level = sig.get('risk_assessment', {}).get('波动率等级', '中')
                        sigma = KELLY_VOLATILITY_SIGMA.get(vol_level, 0.02)
                    else:
                        df = df_dict[code]
                        if 'close' in df.columns and len(df) >= 20:
                            ret = df['close'].pct_change().dropna().tail(60)
                            sigma = float(ret.std())
                            if sigma <= 0 or np.isnan(sigma):
                                sigma = KELLY_VOLATILITY_SIGMA.get(
                                    sig.get('risk_assessment', {}).get('波动率等级', '中'), 0.02)
                        else:
                            sigma = KELLY_VOLATILITY_SIGMA.get(
                                sig.get('risk_assessment', {}).get('波动率等级', '中'), 0.02)
                    strength = max(0.0, sig.get('signal_strength', 0))
                    mu = max(0.001, 0.0005 + 0.0015 * min(1.0, strength))  # 日收益下限 0.1%，避免 f 过小
                    sigma_sq = max(sigma ** 2, 1e-8)
                    f = mu / sigma_sq
                    kelly_raw[code] = max(0.0, min(f, 2.0))
                total_kelly = sum(kelly_raw.values())
                if total_kelly > 0:
                    for code in valid_signals:
                        raw_weight = kelly_raw[code] / total_kelly
                        weight = raw_weight * self.config.max_total_position
                        weight = min(weight, self.config.max_position_per_stock)
                        weight = max(weight, self.config.min_position_per_stock)
                        target_weights[code] = weight
                else:
                    # Kelly 全为 0 时回退等权，避免长期空仓
                    n = len(valid_signals)
                    w = min(self.config.max_total_position / n, self.config.max_position_per_stock)
                    for code in valid_signals:
                        target_weights[code] = w
            elif valid_signals:
                kelly_raw = {}
                for code, sig in valid_signals.items():
                    vol_level = sig.get('risk_assessment', {}).get('波动率等级', '中')
                    sigma = KELLY_VOLATILITY_SIGMA.get(vol_level, 0.02)
                    strength = max(0.0, sig.get('signal_strength', 0))
                    mu = max(0.001, 0.0005 + 0.0015 * min(1.0, strength))
                    sigma_sq = max(sigma ** 2, 1e-8)
                    f = mu / sigma_sq
                    kelly_raw[code] = max(0.0, min(f, 2.0))
                total_kelly = sum(kelly_raw.values())
                if total_kelly > 0:
                    for code in valid_signals:
                        raw_weight = kelly_raw[code] / total_kelly
                        weight = raw_weight * self.config.max_total_position
                        weight = min(weight, self.config.max_position_per_stock)
                        weight = max(weight, self.config.min_position_per_stock)
                        target_weights[code] = weight
                else:
                    n = len(valid_signals)
                    w = min(self.config.max_total_position / n, self.config.max_position_per_stock)
                    for code in valid_signals:
                        target_weights[code] = w
        
        # 确保总权重不超过限制
        total_weight = sum(target_weights.values())
        if total_weight > self.config.max_total_position:
            scale = self.config.max_total_position / total_weight
            target_weights = {k: v * scale for k, v in target_weights.items()}
        
        return target_weights
    
    def generate_rebalance_orders(self, signals: Dict[str, Dict],
                                   current_prices: Dict[str, float],
                                   df_dict: Dict[str, pd.DataFrame] = None) -> List[Dict]:
        """
        生成再平衡订单（过滤烂股）
        
        参数:
            signals: 股票信号字典
            current_prices: 当前价格字典
            df_dict: 股票数据字典（用于质量评估，可选）
        
        返回:
            订单列表
        """
        orders = []
        target_weights = self.calculate_target_weights(signals, df_dict)
        total_value = self.total_value
        
        # 更新当前持仓的市值和权重
        for code, pos in self.positions.items():
            if code in current_prices:
                pos.update_market_value(current_prices[code])
                pos.weight = pos.market_value / total_value if total_value > 0 else 0
        
        # 检查卖出信号和止损/止盈
        for code, pos in list(self.positions.items()):
            if pos.shares == 0:
                continue
                
            sig = signals.get(code, {})
            signal_type = sig.get('signal_type', 'Neutral')
            
            should_sell = False
            sell_reason = ""
            
            # 检查卖出信号
            if signal_type == 'ShortOpen':
                should_sell = True
                sell_reason = f"卖出信号（强度{sig.get('signal_strength', 0):.2f}）"
            
            # 检查单股止损
            elif pos.profit_loss_pct <= -self.config.single_stop_loss_pct * 100:
                should_sell = True
                sell_reason = f"触发单股止损（亏损{pos.profit_loss_pct:.1f}%）"
            
            # 检查单股止盈
            elif pos.profit_loss_pct >= self.config.single_take_profit_pct * 100:
                should_sell = True
                sell_reason = f"触发单股止盈（盈利{pos.profit_loss_pct:.1f}%）"
            
            # 检查目标权重为0（不在目标持仓中）
            elif code not in target_weights:
                should_sell = True
                sell_reason = "目标权重为0，清仓"
            
            if should_sell:
                orders.append({
                    'action': 'SELL',
                    'stock_code': code,
                    'shares': pos.shares,
                    'price': current_prices.get(code, pos.current_price),
                    'reason': sell_reason,
                    'signal_strength': sig.get('signal_strength', 0),
                    'current_weight': pos.weight,
                    'target_weight': 0
                })
        
        # 检查买入信号和加仓（只买入通过质量评估的股票）
        for code, target_weight in target_weights.items():
            if code not in current_prices:
                continue
            
            # 再次进行质量检查（买入前最后一道防线；缠论策略不做烂股过滤）
            if self.config.strategy_type != 'chanlun' and df_dict and code in df_dict:
                sig = signals.get(code, {})
                quality_score, can_buy, reason = evaluate_stock_quality(
                    code, df_dict[code], sig, self.config.signal_threshold
                )
                if not can_buy:
                    print(f"[选股池] 买入前过滤 {code}: {reason} (质量评分{quality_score:.2f})")
                    continue
            
            price = current_prices[code]
            current_pos = self.positions.get(code)
            current_weight = current_pos.weight if current_pos else 0
            
            weight_diff = target_weight - current_weight
            
            # 如果权重差异超过再平衡阈值，则调仓
            if weight_diff > self.config.rebalance_threshold:
                target_value = total_value * target_weight
                current_value = current_pos.market_value if current_pos else 0
                buy_value = target_value - current_value
                
                # 检查现金是否足够
                available_cash = self.cash - total_value * self.config.min_cash_ratio
                buy_value = min(buy_value, available_cash)
                
                if buy_value > 0:
                    shares = int(buy_value / price / 100) * 100  # A股最小单位100股
                    if shares > 0:
                        sig = signals.get(code, {})
                        orders.append({
                            'action': 'BUY',
                            'stock_code': code,
                            'shares': shares,
                            'price': price,
                            'reason': f"目标权重{target_weight*100:.1f}%，当前{current_weight*100:.1f}%",
                            'signal_strength': sig.get('signal_strength', 0),
                            'current_weight': current_weight,
                            'target_weight': target_weight
                        })
            
            elif weight_diff < -self.config.rebalance_threshold:
                # 需要减仓
                if current_pos and current_pos.shares > 0:
                    target_value = total_value * target_weight
                    sell_value = current_pos.market_value - target_value
                    shares = int(sell_value / price / 100) * 100
                    
                    if shares > 0:
                        sig = signals.get(code, {})
                        orders.append({
                            'action': 'SELL',
                            'stock_code': code,
                            'shares': min(shares, current_pos.shares),
                            'price': price,
                            'reason': f"减仓至目标权重{target_weight*100:.1f}%",
                            'signal_strength': sig.get('signal_strength', 0),
                            'current_weight': current_weight,
                            'target_weight': target_weight
                        })
        
        return orders
    
    def execute_order(self, order: Dict, date: str = None) -> bool:
        """
        执行订单
        
        参数:
            order: 订单字典
            date: 交易日期
        
        返回:
            是否执行成功
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        code = order['stock_code']
        action = order['action']
        shares = order['shares']
        price = order['price']
        
        if action == 'BUY':
            cost = shares * price
            if cost > self.cash:
                print(f"[选股池] 买入失败：现金不足，需要{cost:.2f}，可用{self.cash:.2f}")
                return False
            
            self.cash -= cost
            
            if code not in self.positions:
                self.positions[code] = StockPosition(
                    stock_code=code,
                    shares=shares,
                    avg_cost=price,
                    entry_date=date
                )
            else:
                pos = self.positions[code]
                total_cost = pos.shares * pos.avg_cost + cost
                pos.shares += shares
                pos.avg_cost = total_cost / pos.shares
            
            self.positions[code].update_market_value(price)
            self.positions[code].signal_strength = order.get('signal_strength', 0)
            self.positions[code].last_signal = 'BUY'
            
            # 买入时的盈亏信息（均为0）
            sell_avg_cost = 0
            sell_profit_amount = 0
            sell_profit_pct = 0
            
        elif action == 'SELL':
            if code not in self.positions or self.positions[code].shares < shares:
                print(f"[选股池] 卖出失败：持仓不足")
                return False
            
            revenue = shares * price
            self.cash += revenue
            
            pos = self.positions[code]
            # 计算卖出盈亏（在删除持仓前）
            sell_avg_cost = pos.avg_cost
            sell_profit_amount = (price - sell_avg_cost) * shares
            sell_profit_pct = (price - sell_avg_cost) / sell_avg_cost * 100 if sell_avg_cost > 0 else 0
            
            pos.shares -= shares
            if pos.shares == 0:
                del self.positions[code]
            else:
                pos.update_market_value(price)
            
        # 记录交易
        # 买入时盈亏为0，卖出时记录实际盈亏
        trade_avg_cost = sell_avg_cost if action == 'SELL' else order.get('avg_cost', 0)
        trade_profit_amount = sell_profit_amount if action == 'SELL' else 0
        trade_profit_pct = sell_profit_pct if action == 'SELL' else 0
        
        trade = TradeRecord(
            date=date,
            stock_code=code,
            stock_name=order.get('stock_name', code),
            action=action,
            shares=shares,
            price=price,
            amount=shares * price,
            reason=order.get('reason', ''),
            signal_strength=order.get('signal_strength', 0),
            market_state=order.get('market_state', ''),
            before_weight=order.get('current_weight', 0),
            after_weight=order.get('target_weight', 0),
            avg_cost=trade_avg_cost,
            profit_amount=trade_profit_amount,
            profit_pct=trade_profit_pct
        )
        self.trades.append(trade)
        
        return True
    
    def take_snapshot(self, date: str, signals: Dict[str, Dict] = None):
        """
        记录每日快照
        
        参数:
            date: 日期
            signals: 当日信号（可选）
        """
        positions_dict = {}
        for code, pos in self.positions.items():
            positions_dict[code] = {
                'shares': pos.shares,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'profit_loss_pct': pos.profit_loss_pct,
                'weight': pos.weight
            }
        
        # 计算收益
        prev_value = self.daily_snapshots[-1].total_value if self.daily_snapshots else self.config.initial_capital
        daily_return = (self.total_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
        cumulative_return = (self.total_value - self.config.initial_capital) / self.config.initial_capital * 100
        
        snapshot = DailySnapshot(
            date=date,
            total_value=self.total_value,
            cash=self.cash,
            position_value=self.position_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            positions=positions_dict,
            signals={k: {'signal_type': v.get('signal_type'), 
                        'signal_strength': v.get('signal_strength')} 
                    for k, v in (signals or {}).items()}
        )
        self.daily_snapshots.append(snapshot)


# ==================== 选股池回测引擎 ====================

class PoolBacktestEngine:
    """
    选股池回测引擎
    
    对选股池进行历史回测，计算组合收益和风险指标
    """
    
    def __init__(self, pool_config: PoolConfig):
        """
        初始化回测引擎
        
        参数:
            pool_config: 选股池配置
        """
        self.config = pool_config
        self.manager = StockPoolManager(pool_config)
        self.benchmark_data = None  # 基准数据（如沪深300）
        self.benchmark_code = "510300"  # 默认使用沪深300ETF作为基准
        self.benchmark_values = []  # 基准每日净值
        self._code_to_industry: Dict[str, str] = {}  # 分行业龙头模式下 代码->行业 映射，供 LLM 使用
    
    def load_data(self, days: int = 252, abort_check: Optional[Callable[[], bool]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有股票数据和基准数据
        
        参数:
            days: 回测天数
            abort_check: 可选，停止检查函数
        
        返回:
            股票代码到DataFrame的映射
        """
        # 根据 universe_source 解析实际要加载的股票列表（分行业龙头 / 指数成分 / 手动）
        stocks_to_load = self.config.stocks
        if getattr(self.config, "universe_source", "manual") == "index" and getattr(self.config, "universe_index", ""):
            idx_code = (self.config.universe_index or "").strip()
            if idx_code:
                stocks_to_load = get_index_constituents(idx_code)
                if not stocks_to_load:
                    raise ValueError(
                        f"指数选股失败：未能获取指数 {idx_code} 的成分股。请检查指数代码或网络后重试。"
                    )
                self.config.stocks = stocks_to_load
                print(f"[回测] 指数选股: {idx_code} -> {len(stocks_to_load)} 只成分股")
        elif getattr(self.config, "universe_source", "manual") == "industry":
            industry_list = getattr(self.config, "industry_list", None) or None
            leaders_per = getattr(self.config, "leaders_per_industry", 1) or 1
            stocks_to_load, self._code_to_industry = select_industry_leaders(industry_list, leaders_per)
            if not stocks_to_load:
                raise ValueError(
                    "分行业龙头选股失败：未能获取到任何股票。请检查网络后重试；若行业名称有误可先使用「预览龙头」确认。"
                )
            self.config.stocks = stocks_to_load
            print(f"[回测] 分行业龙头选股: {len(stocks_to_load)} 只（{len(self._code_to_industry)} 行业映射）")
        
        data_dict = {}
        extra_days = 80  # 技术指标（如 MA60）需要的额外数据
        # 请求天数 = 回测天数 + 指标预热，仅加 1.2 倍缓冲（避免 1 年回测请求 800+ 日历天）
        request_days = int((days + extra_days) * 1.2)
        # 数据获取与单股回测一致：先检查缓存是否已是最新（5 日内），若是则优先使用缓存，不请求网络（见 stock_data.fetch_stock_data）
        print(f"[回测] 开始加载 {len(stocks_to_load)} 只股票数据（请求约 {request_days} 个交易日，缓存已最新时优先使用本地缓存）...")
        
        failed_stocks = []
        for i, code in enumerate(stocks_to_load):
            # 在数据加载过程中检查是否停止
            if abort_check and abort_check():
                print(f"\n[回测] 用户停止数据加载 @ {i+1}/{len(stocks_to_load)}")
                raise InterruptedError("用户停止回测")
            
            # 在加载股票之间添加小延迟，避免请求过于频繁导致限流
            if i > 0:
                import time
                time.sleep(0.5)  # 每只股票之间延迟0.5秒
            
            print(f"[回测] ({i+1}/{len(stocks_to_load)}) 加载 {code}...")
            try:
                stock_data = get_stock_data(code, request_days, use_cache=True)
                if stock_data and len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    df = add_technical_indicators_to_df(df)
                    df = df.dropna().reset_index(drop=True)
                    data_dict[code] = df
                    print(f"[回测] ✓ {code} 加载成功，{len(df)} 条数据")
                else:
                    failed_stocks.append(code)
                    print(f"[回测] ✗ {code} 数据不足（需要>60条，实际{len(stock_data) if stock_data else 0}条），跳过")
            except Exception as e:
                failed_stocks.append(code)
                error_msg = str(e)
                if '网络连接' in error_msg or 'Connection' in error_msg:
                    print(f"[回测] ✗ {code} 加载失败: 网络连接错误，请检查网络或稍后重试")
                else:
                    print(f"[回测] ✗ {code} 加载失败: {error_msg}")
        
        # 统计加载结果
        success_count = len(data_dict)
        if failed_stocks:
            print(f"\n[回测] 数据加载完成: 成功 {success_count}/{len(stocks_to_load)} 只股票")
            print(f"[回测] 失败的股票: {', '.join(failed_stocks)}")
            if success_count == 0:
                raise ValueError("所有股票数据加载失败，无法进行回测。请检查网络连接或稍后重试。")
            elif success_count < len(self.config.stocks) // 2:
                print(f"[回测] 警告: 超过一半的股票加载失败，回测结果可能不准确")
        else:
            print(f"\n[回测] ✓ 所有股票数据加载成功 ({success_count} 只)")
        
        # 记录成功加载的股票列表（用于统计时判断）
        self._loaded_stocks = set(data_dict.keys())
        
        # 加载基准数据（沪深300ETF）
        print(f"[回测] 加载基准数据 {self.benchmark_code}（沪深300ETF）...")
        try:
            benchmark_data = get_stock_data(self.benchmark_code, request_days, use_cache=True)
            if benchmark_data and len(benchmark_data) > 60:
                df_benchmark = pd.DataFrame(benchmark_data)
                df_benchmark = df_benchmark.dropna().reset_index(drop=True)
                self.benchmark_data = df_benchmark
                print(f"[回测] 基准数据加载成功，{len(df_benchmark)} 条数据")
            else:
                print(f"[回测] 警告：基准数据不足，基准对比功能将不可用")
                self.benchmark_data = None
        except Exception as e:
            print(f"[回测] 基准数据加载失败: {e}，基准对比功能将不可用")
            self.benchmark_data = None
        
        return data_dict
    
    def run_backtest(self, days: int = 252, 
                     rebalance_interval: int = 5,
                     use_ai: bool = False,
                     use_llm_signals: bool = False,
                     llm_sample_rate: int = 10,
                     high_win_rate_mode: bool = False,
                     llm=None,
                     decision_callback=None,
                     abort_check: Optional[Callable[[], bool]] = None,
                     risk_control_mode: str = 'off',
                     selection_mode: Optional[str] = None,
                     selection_top_n: Optional[int] = None,
                     selection_interval: Optional[int] = None,
                     score_weights: Optional[Dict[str, float]] = None,
                     factor_set: Optional[str] = None,
                     weight_source: str = "manual",
                     model_name: str = "",
                     learned_weights: Optional[Dict[str, Dict[str, float]]] = None,
                     no_lookahead: bool = False,
                     start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        运行AI选股池回测
        
        不是固定持仓再平衡，而是每天对池中股票进行AI信号判断，
        动态决定买入/卖出哪些股票，类似单股AI回测但应用于多股票组合。
        
        参数:
            days: 回测天数
            rebalance_interval: 信号检查间隔（交易日），默认每5天检查一次全量信号
            use_ai: 是否使用AI深度分析
            use_llm_signals: 是否使用LLM生成交易信号
            llm_sample_rate: LLM采样频率（每N天调用一次）
            high_win_rate_mode: 是否启用高胜率模式
            llm: LLM实例
            decision_callback: 决策回调函数(event_type, data)，用于实时推送决策
            abort_check: 可选，无参可调用对象，返回 True 时立即停止回测并返回已完成的局部结果
            risk_control_mode: 风控模式。'off'=回测不启用；'warn'=记录警告但不阻断；'block'=阻断
            selection_mode: 选股模式，默认使用配置值
            selection_top_n: 因子选股 TopN，默认使用配置值
            selection_interval: 因子重选间隔交易日，默认使用配置值
            score_weights: 因子权重，默认使用配置值
            factor_set: 因子集合（style/trading/hybrid，风格与估值/情绪与交易），默认使用配置值
            weight_source: 权重来源（manual/learned）
            model_name: 学习模型名（展示用途）
            learned_weights: 学习权重（weight_source=learned 时使用）
            no_lookahead: 是否启用无前视模式
            start_date: 回测起始日期（YYYY-MM-DD）
        
        返回:
            回测结果字典（若被 abort_check 停止则含 aborted=True）
        """
        from modules.strategy_config import ai_daily_decision
        from modules.strategy_config import get_strategy_config
        
        start_time = time.time()
        effective_selection_mode = (selection_mode or self.config.selection_mode or "none").strip().lower()
        effective_selection_top_n = int(selection_top_n if selection_top_n is not None else self.config.selection_top_n or 10)
        effective_selection_interval = int(selection_interval if selection_interval is not None else self.config.selection_interval or 0)
        effective_score_weights = score_weights if score_weights is not None else self.config.score_weights
        effective_factor_set = (factor_set or self.config.factor_set or "hybrid").strip().lower()
        if effective_factor_set not in ("style", "trading", "hybrid", "momentum", "volatility", "volume", "reversal"):
            effective_factor_set = "hybrid"
        if effective_selection_mode not in ("none", "factor_top_n"):
            effective_selection_mode = "none"
        # 因子挖掘已移除，factor_top_n 按 none 处理（使用全池）
        if effective_selection_mode == "factor_top_n":
            effective_selection_mode = "none"
        
        # LLM模式初始化
        if use_llm_signals:
            from prompts import llm_generate_signal
            if llm is None:
                from llm import llm as default_llm
                llm = default_llm
        
        signal_mode = "LLM大模型" if use_llm_signals else "规则算法"
        win_rate_mode = "高胜率" if high_win_rate_mode else "标准"
        
        print(f"\n{'='*60}")
        print(f"[AI选股池回测] 开始回测")
        print(f"[AI选股池回测] 股票池: {self.config.stocks}")
        print(f"[AI选股池回测] 策略: {self.config.strategy_type} + {self.config.risk_preference}")
        print(f"[AI选股池回测] 信号模式: {signal_mode}")
        if high_win_rate_mode:
            print(f"[AI选股池回测] 🎯 高胜率模式已启用（严格入场条件）")
        if use_llm_signals:
            print(f"[AI选股池回测] 🧠 LLM采样频率: 每{llm_sample_rate}天调用一次")
        print(f"[AI选股池回测] 初始资金: ¥{self.config.initial_capital:,.0f}")
        print(f"[AI选股池回测] 回测天数: {days}，信号检查间隔: {rebalance_interval}天")
        print(f"[AI选股池回测] 仓位分配: {self.config.allocation_method}")
        print(f"[AI选股池回测] 📊 模式: AI动态选股（非固定持仓）")
        print(f"{'='*60}\n")
        
        # 获取策略配置（支持高胜率模式）
        strategy_config = get_strategy_config(
            self.config.strategy_type if self.config.strategy_type != 'adaptive' else 'trend',
            self.config.risk_preference,
            high_win_rate_mode=high_win_rate_mode
        )
        is_adaptive = self.config.strategy_type == 'adaptive'
        
        # 加载数据（传入停止检查函数）
        data_dict = self.load_data(days, abort_check=abort_check)
        if not data_dict:
            raise ValueError("没有可用的股票数据")
        self._backtest_data_dict = data_dict  # 供 _calculate_statistics 计算各股票期间涨跌幅
        
        # 找到共同的交易日期范围
        all_dates = None
        for code, df in data_dict.items():
            dates = set(df['date'].tolist())
            if all_dates is None:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)
        
        all_dates = sorted(list(all_dates))

        effective_start_date = (start_date or "").strip()
        if effective_start_date:
            all_dates = [
                d for d in all_dates
                if (
                    d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
                ) >= effective_start_date
            ]
        if no_lookahead and not effective_start_date:
            raise ValueError("无前视模式需要 start_date")
        if len(all_dates) < 20:
            raise ValueError(f"起始日期过滤后共同交易日不足，仅有 {len(all_dates)} 天")
        
        # 取最后days天
        if len(all_dates) > days:
            all_dates = all_dates[-days:]
        
        print(f"[AI选股池回测] 共同交易日: {all_dates[0]} ~ {all_dates[-1]}，共 {len(all_dates)} 天")
        
        # 重置管理器
        self.manager = StockPoolManager(self.config)
        
        # 初始化基准数据
        self.benchmark_values = []
        self.benchmark_initial_price = None
        if self.benchmark_data is not None and len(self.benchmark_data) > 0:
            # 找到第一个交易日的基准价格（支持多种日期格式）
            first_date = all_dates[0]
            try:
                if hasattr(first_date, 'strftime'):
                    first_date_str = first_date.strftime('%Y-%m-%d')
                else:
                    first_date_str = str(first_date)
                
                benchmark_df = self.benchmark_data.copy()
                benchmark_df['date_str'] = benchmark_df['date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                )
                first_benchmark_row = benchmark_df[benchmark_df['date_str'] == first_date_str]
                
                if not first_benchmark_row.empty:
                    self.benchmark_initial_price = first_benchmark_row.iloc[0]['close']
                    print(f"[回测] 基准初始价格: ¥{self.benchmark_initial_price:.2f}")
                else:
                    # 如果找不到精确匹配，使用最接近的日期
                    benchmark_df['date_ts'] = pd.to_datetime(benchmark_df['date']).apply(lambda x: x.timestamp())
                    first_date_ts = pd.to_datetime(first_date).timestamp() if hasattr(first_date, 'timestamp') else pd.to_datetime(first_date_str).timestamp()
                    closest_idx = (benchmark_df['date_ts'] - first_date_ts).abs().idxmin()
                    self.benchmark_initial_price = benchmark_df.iloc[closest_idx]['close']
                    print(f"[回测] 基准初始价格（使用最接近日期）: ¥{self.benchmark_initial_price:.2f}")
            except Exception as e:
                print(f"[回测] 警告：初始化基准价格失败: {e}")
        
        # 每只股票的持仓状态
        stock_states = {}
        for code in self.config.stocks:
            stock_states[code] = {
                'entry_price': 0,
                'entry_date': None,
                'holding_days': 0,
                'highest_price': 0,
                'last_market_state': None,
                # LLM模式缓存
                'llm_cached_action': 'HOLD',
                'llm_cached_confidence': 0.5,
                'llm_cached_reason': '等待LLM决策',
                'last_llm_call_idx': -999
            }
        
        # LLM调用统计
        llm_call_count = 0

        # 组合止损冷却期（防止频繁触发）
        stop_loss_cooldown = 0

        # 组合历史峰值净值（用于基于峰值的回撤判定，避免每次相对初始资金反复触发）
        peak_total_value = self.config.initial_capital
        
        # 逐日回测
        total_days = len(all_dates)
        progress_interval = max(1, total_days // 10)
        adaptive_check_interval = 20  # 自适应策略每20天重新评估
        backtest_aborted = False
        selected_universe = list(self.config.stocks)
        next_date_map = {
            (
                all_dates[i].strftime('%Y-%m-%d') if hasattr(all_dates[i], 'strftime') else str(all_dates[i])
            ): (
                all_dates[i + 1].strftime('%Y-%m-%d') if hasattr(all_dates[i + 1], 'strftime') else str(all_dates[i + 1])
            )
            for i in range(len(all_dates) - 1)
        }
        
        for idx, date in enumerate(all_dates):
            # 检查是否被用户停止
            if abort_check and abort_check():
                backtest_aborted = True
                print(f"\n[AI选股池回测] 用户停止回测 @ Day {idx+1}/{total_days}")
                break
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            
            # 进度显示
            if idx % progress_interval == 0 or idx == total_days - 1:
                progress = (idx + 1) / total_days * 100
                elapsed = time.time() - start_time
                bar_len = 20
                filled = int(bar_len * (idx + 1) / total_days)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r[AI选股池回测] 进度: [{bar}] {progress:.0f}% | Day {idx+1}/{total_days}", end='')
                
                # 通知前端：进度更新
                if decision_callback:
                    decision_callback('progress', {
                        'current': idx + 1,
                        'total': total_days,
                        'percent': round(progress, 1),
                        'date': date_str,
                        'elapsed': round(elapsed, 2)
                    })
            
            # 获取当日数据（在处理前检查是否停止）
            if abort_check and abort_check():
                backtest_aborted = True
                print(f"\n[AI选股池回测] 用户停止回测（数据处理前） @ Day {idx+1}/{total_days}")
                break
            
            current_prices = {}
            daily_df_dict = {}
            
            for code, df in data_dict.items():
                mask = df['date'] <= date
                df_window = df[mask].tail(80)
                if len(df_window) >= 60:
                    daily_df_dict[code] = df_window
                    current_prices[code] = df_window.iloc[-1]['close']

            # 更新持仓市值和持仓天数
            for code, pos in self.manager.positions.items():
                if code in current_prices:
                    pos.update_market_value(current_prices[code])
                    stock_states[code]['holding_days'] += 1
                    # 更新最高价
                    if current_prices[code] > stock_states[code]['highest_price']:
                        stock_states[code]['highest_price'] = current_prices[code]

            # 风控审计（回测默认不启用；模拟盘可记录警告或阻断）
            risk_no_new_buys = False
            if risk_control_mode != 'off':
                equity_curve_so_far = [s.total_value for s in self.manager.daily_snapshots]
                if len(equity_curve_so_far) >= 2:
                    recent_returns = get_recent_daily_returns_from_equity(equity_curve_so_far, lookback=10)
                    positions_for_risk = {
                        code: {'market_value': p.market_value}
                        for code, p in self.manager.positions.items()
                        if p.shares > 0
                    }
                    risk_input = RiskAuditInput(
                        cash=self.manager.cash,
                        total_value=self.manager.total_value,
                        initial_capital=self.config.initial_capital,
                        positions_value=self.manager.position_value,
                        positions=positions_for_risk,
                        recent_daily_returns=recent_returns,
                        peak_value=peak_total_value,
                        current_date=date_str,
                    )
                    risk_result = audit_risk(risk_input)
                    would_block = not risk_result.pass_audit and (risk_result.action in ('stop', 'pause_trading') or (risk_result.action == 'reduce_position' and 'over_weight_code' not in risk_result.details))
                    if risk_control_mode == 'block':
                        risk_no_new_buys = would_block
                    elif risk_control_mode == 'warn' and would_block:
                        if idx % 5 == 0 or risk_result.action == 'stop':
                            print(f"\n[AI选股池回测] {date_str} 风控警告（不阻断）: {risk_result.reason}")
                        if decision_callback:
                            decision_callback('risk_warning', {'date': date_str, 'reason': risk_result.reason, 'action': risk_result.action})

            
            # ========== 对每只股票进行AI决策 ==========
            daily_signals = {}
            holding_codes = {code for code, pos in self.manager.positions.items() if pos.shares > 0}
            decision_universe = sorted(set(selected_universe).union(holding_codes))
            
            # LLM组合决策模式：综合所有股票后一次性决策
            if use_llm_signals:
                # 检查是否需要调用LLM（基于采样间隔）
                should_call_llm = (idx % llm_sample_rate == 0) or (idx == 0)
                
                # 如果有持仓盈亏较大，也需要调用
                for code, state in stock_states.items():
                    if code in self.manager.positions and self.manager.positions[code].shares > 0:
                        if state['entry_price'] > 0 and code in current_prices:
                            pnl = (current_prices[code] - state['entry_price']) / state['entry_price'] * 100
                            if abs(pnl) > strategy_config.get('stop_loss_pct', 0.05) * 100 * 0.7:
                                should_call_llm = True
                                break
                
                if should_call_llm:
                    # 收集所有股票数据
                    stocks_data = []
                    for code in decision_universe:
                        if code not in daily_df_dict or code not in current_prices:
                            continue
                        
                        df_window = daily_df_dict[code]
                        row = df_window.iloc[-1]
                        price = current_prices[code]
                        state = stock_states[code]
                        
                        has_position = code in self.manager.positions and self.manager.positions[code].shares > 0
                        profit_pct = 0
                        if has_position and state['entry_price'] > 0:
                            profit_pct = (price - state['entry_price']) / state['entry_price'] * 100
                        
                        # 转换指标为字典
                        indicators = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
                        
                        item = {
                            'code': code,
                            'indicators': indicators,
                            'has_position': has_position,
                            'entry_price': state['entry_price'] if has_position else None,
                            'holding_days': state['holding_days'] if has_position else 0,
                            'highest_price': state['highest_price'] if has_position else None,
                            'profit_pct': profit_pct
                        }
                        code_to_ind = getattr(self, '_code_to_industry', None) or {}
                        if code_to_ind:
                            item['industry'] = code_to_ind.get(code, '')
                        stocks_data.append(item)
                    
                    # 获取当前持仓信息
                    positions_info = {}
                    for code, pos in self.manager.positions.items():
                        if pos.shares > 0:
                            positions_info[code] = {
                                'shares': pos.shares,
                                'value': pos.market_value,
                                'weight': pos.weight
                            }
                    
                    # 确定当前策略
                    current_strategy = self.config.strategy_type
                    if is_adaptive:
                        # 取第一个股票的市场状态
                        first_state = list(stock_states.values())[0] if stock_states else {}
                        mstate = first_state.get('last_market_state', '')
                        if mstate in ['BULLISH', 'TRENDING_UP']:
                            current_strategy = 'trend'
                        elif mstate in ['BEARISH', 'TRENDING_DOWN', 'RANGING', 'VOLATILE']:
                            current_strategy = 'mean_reversion'
                    
                    # 通知前端：开始组合决策
                    if decision_callback:
                        decision_callback('llm_start', {
                            'date': date_str,
                            'stock_code': '组合决策',
                            'strategy': current_strategy,
                            'stocks_count': len(stocks_data),
                            'price': 0
                        })
                    
                    # 获取当日A股市场环境（北向资金、成交量等）供LLM判断趋势
                    market_context = None
                    try:
                        # 在获取市场环境前检查是否停止
                        if abort_check and abort_check():
                            backtest_aborted = True
                            print(f"\n[AI选股池回测] 用户停止回测（获取市场环境前） @ Day {idx+1}/{total_days}")
                            break
                        from tools.stock_data import get_market_context_for_llm
                        market_context = get_market_context_for_llm(days=5, end_date=date_str)
                    except Exception as _:
                        pass
                    # 在LLM调用前再次检查是否停止
                    if abort_check and abort_check():
                        backtest_aborted = True
                        print(f"\n[AI选股池回测] 用户停止回测（LLM调用前） @ Day {idx+1}/{total_days}")
                        break
                    
                    # 调用组合决策（分行业龙头模式下传入行业信息供 LLM 考虑分散/轮动）
                    from prompts import llm_portfolio_decision
                    pool_mode = None
                    stock_industry_map = None
                    if getattr(self.config, 'universe_source', 'manual') == 'industry':
                        pool_mode = 'industry'
                        stock_industry_map = getattr(self, '_code_to_industry', None) or {}
                    portfolio_result = llm_portfolio_decision(
                        stocks_data=stocks_data,
                        positions=positions_info,
                        cash=self.manager.cash,
                        total_value=self.manager.total_value,
                        strategy_type=current_strategy,
                        risk_preference=self.config.risk_preference,
                        max_positions=min(len(decision_universe), int(1 / self.config.min_position_per_stock)),
                        high_win_rate_mode=high_win_rate_mode,
                        market_context=market_context,
                        pool_mode=pool_mode,
                        stock_industry_map=stock_industry_map,
                        llm=llm
                    )
                    
                    # LLM调用后再次检查是否停止
                    if abort_check and abort_check():
                        backtest_aborted = True
                        print(f"\n[AI选股池回测] 用户停止回测（LLM调用后） @ Day {idx+1}/{total_days}")
                        break
                    
                    llm_call_count += 1
                    
                    # 更新所有股票的缓存
                    for code in decision_universe:
                        decision = portfolio_result['decisions'].get(code, {'action': 'HOLD', 'confidence': 0.5, 'reason': '无决策'})
                        stock_states[code]['llm_cached_action'] = decision['action']
                        stock_states[code]['llm_cached_confidence'] = decision['confidence']
                        stock_states[code]['llm_cached_reason'] = decision['reason']
                        stock_states[code]['last_llm_call_idx'] = idx
                    
                    # 通知前端：组合决策完成
                    if decision_callback:
                        decision_callback('llm_decision', {
                            'date': date_str,
                            'stock_code': '组合',
                            'action': f"买{len(portfolio_result['priority_buy'])}卖{len(portfolio_result['priority_sell'])}",
                            'confidence': 0.8,
                            'reason': portfolio_result['analysis'],
                            'price': 0,
                            'has_position': len(positions_info) > 0,
                            'call_count': llm_call_count,
                            'priority_buy': portfolio_result['priority_buy'],
                            'priority_sell': portfolio_result['priority_sell']
                        })
                    
                    print(f"\n[AI组合决策] {date_str} {portfolio_result['analysis']}")
                    if portfolio_result['priority_sell']:
                        print(f"  📉 建议卖出: {', '.join(portfolio_result['priority_sell'])}")
                    if portfolio_result['priority_buy']:
                        print(f"  📈 建议买入: {', '.join(portfolio_result['priority_buy'])}")
                
                # 从缓存构建daily_signals
                for code in decision_universe:
                    if code not in daily_df_dict or code not in current_prices:
                        continue
                    
                    state = stock_states[code]
                    action = state['llm_cached_action']
                    confidence = state['llm_cached_confidence']
                    reason = f"[LLM组合] {state['llm_cached_reason']}"
                    
                    daily_signals[code] = {
                        'signal_type': 'LongOpen' if action == 'BUY' else ('ShortOpen' if action == 'SELL' else 'Neutral'),
                        'signal_strength': confidence,
                        'action': action,
                        'reason': reason,
                        'market_state': state['last_market_state'] if is_adaptive else self.config.strategy_type
                    }
            else:
                # 规则模式：独立决策每只股票
                # 在规则模式循环前检查是否停止
                if abort_check and abort_check():
                    backtest_aborted = True
                    print(f"\n[AI选股池回测] 用户停止回测（规则模式循环前） @ Day {idx+1}/{total_days}")
                    break
                
                for code in decision_universe:
                    # 在每只股票处理前检查是否停止
                    if abort_check and abort_check():
                        backtest_aborted = True
                        print(f"\n[AI选股池回测] 用户停止回测（处理股票{code}时） @ Day {idx+1}/{total_days}")
                        break
                    
                    if code not in daily_df_dict or code not in current_prices:
                        continue
                    
                    df_window = daily_df_dict[code]
                    row = df_window.iloc[-1]
                    price = current_prices[code]
                    state = stock_states[code]
                    
                    has_position = code in self.manager.positions and self.manager.positions[code].shares > 0
                    
                    # 自适应策略：定期评估市场状态
                    if is_adaptive and (idx == 0 or idx % adaptive_check_interval == 0):
                        if len(df_window) >= 20:
                            market_analysis = analyze_market_state(df_window, use_ai=False)
                            state['last_market_state'] = market_analysis['market_state']
                            adaptive_config, _ = get_adaptive_strategy_config(
                                df_window, self.config.risk_preference, use_ai=False
                            )
                            strategy_config.update(adaptive_config)
                    
                    # 规则决策
                    # 计算当前仓位比例
                    current_pos_ratio = 0.0
                    if code in self.manager.positions and self.manager.positions[code].shares > 0:
                        pos_value = self.manager.positions[code].market_value
                        current_pos_ratio = pos_value / self.manager.total_value if self.manager.total_value > 0 else 0
                    
                    # 缠论策略需传入近期K线窗口，供分型/笔/中枢/背驰分析
                    call_kwargs = {}
                    if self.config.strategy_type == 'chanlun' and code in daily_df_dict:
                        call_kwargs['df_window'] = daily_df_dict[code]
                    result = ai_daily_decision(
                        row, 
                        has_position, 
                        state['entry_price'],
                        state['holding_days'],
                        strategy_config,
                        state['highest_price'],
                        current_pos_ratio,
                        **call_kwargs
                    )
                    # 兼容新旧返回格式，处理 None 情况
                    if result is None:
                        action, confidence, reason = 'HOLD', 0.5, '决策异常'
                    elif len(result) == 4:
                        action, confidence, reason, _ = result
                    else:
                        action, confidence, reason = result
                    
                    daily_signals[code] = {
                        'signal_type': 'LongOpen' if action in ['BUY', 'ADD'] else ('ShortOpen' if action in ['SELL', 'REDUCE'] else 'Neutral'),
                        'signal_strength': confidence,
                        'action': action,
                        'reason': reason,
                        'market_state': state['last_market_state'] if is_adaptive else self.config.strategy_type
                    }

            # ========== 执行卖出/减仓决策 ==========
            for code, sig in daily_signals.items():
                if code not in current_prices:
                    continue
                
                action = sig['action']
                confidence = sig['signal_strength']
                reason = sig['reason']
                price = current_prices[code]
                state = stock_states[code]
                has_position = code in self.manager.positions and self.manager.positions[code].shares > 0
                
                # 全部卖出（加入时间因子：持仓不足 min_holding_days 且非止损时不卖，避免频繁调仓磨损）
                min_holding_days = strategy_config.get('min_holding_days', 5)
                stop_loss_pct_threshold = strategy_config.get('stop_loss_pct', 0.03) * 100  # 如 3% -> 触发止损为盈亏<=-3%
                profit_pct_pre = (price - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] > 0 else 0
                allow_short_hold_sell = profit_pct_pre <= -stop_loss_pct_threshold or state['holding_days'] >= min_holding_days
                
                if action == 'SELL' and has_position and confidence >= strategy_config.get('confidence_threshold', 0.6) and allow_short_hold_sell:
                    pos = self.manager.positions[code]
                    sell_shares = pos.shares  # 先保存卖出股数，因为execute_order后会变成0
                    order = {
                        'action': 'SELL',
                        'stock_code': code,
                        'shares': sell_shares,
                        'price': price,
                        'reason': reason,
                        'signal_strength': confidence,
                        'market_state': state['last_market_state'] or self.config.strategy_type,
                        'current_weight': pos.weight if pos else 0,
                        'target_weight': 0
                    }
                    
                    if self.manager.execute_order(order, date_str):
                        profit_pct = (price - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] > 0 else 0
                        print(f"\n[AI选股池回测] {date_str} 卖出 {code} {sell_shares}股 @ ¥{price:.2f}，盈亏{profit_pct:+.1f}%，原因: {reason}")
                        
                        # 通知前端：执行卖出
                        if decision_callback:
                            decision_callback('trade', {
                                'date': date_str,
                                'stock_code': code,
                                'action': 'SELL',
                                'shares': sell_shares,
                                'price': round(price, 2),
                                'profit_pct': round(profit_pct, 2),
                                'reason': reason
                            })
                        
                        # 重置状态（保留LLM缓存字段）
                        stock_states[code] = {
                            'entry_price': 0, 'entry_date': None, 'holding_days': 0,
                            'highest_price': 0, 'last_market_state': state['last_market_state'],
                            'llm_cached_action': 'HOLD',
                            'llm_cached_confidence': 0.5,
                            'llm_cached_reason': '卖出后观望',
                            'last_llm_call_idx': idx
                        }
                
                # 部分减仓
                elif action == 'REDUCE' and has_position and confidence >= strategy_config.get('confidence_threshold', 0.6):
                    pos = self.manager.positions[code]
                    reduce_ratio = strategy_config.get('reduce_position_ratio', 0.5)
                    reduce_shares = int(pos.shares * reduce_ratio / 100) * 100  # 减仓股数（整手）
                    
                    if reduce_shares > 0 and reduce_shares < pos.shares:
                        order = {
                            'action': 'SELL',  # 对于manager来说，减仓也是卖出
                            'stock_code': code,
                            'shares': reduce_shares,
                            'price': price,
                            'reason': f"[减仓] {reason}",
                            'signal_strength': confidence,
                            'market_state': state['last_market_state'] or self.config.strategy_type,
                            'current_weight': pos.weight if pos else 0,
                            'target_weight': pos.weight * (1 - reduce_ratio) if pos else 0
                        }
                        
                        if self.manager.execute_order(order, date_str):
                            profit_pct = (price - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] > 0 else 0
                            remaining_shares = pos.shares  # execute_order后的剩余股数
                            print(f"\n[AI选股池回测] {date_str} 减仓 {code} {reduce_shares}股 @ ¥{price:.2f}，剩余{remaining_shares}股，盈亏{profit_pct:+.1f}%，原因: {reason}")
                            
                            # 通知前端：执行减仓
                            if decision_callback:
                                decision_callback('trade', {
                                    'date': date_str,
                                    'stock_code': code,
                                    'action': 'REDUCE',
                                    'shares': reduce_shares,
                                    'remaining_shares': remaining_shares,
                                    'price': round(price, 2),
                                    'profit_pct': round(profit_pct, 2),
                                    'reason': reason
                                })
            
            # 在执行交易前检查是否停止
            if abort_check and abort_check():
                backtest_aborted = True
                print(f"\n[AI选股池回测] 用户停止回测（执行交易前） @ Day {idx+1}/{total_days}")
                break
            
            # ========== 执行买入决策 ==========
            # 收集所有买入信号，按信号强度(confidence)排序，表现好的可提高仓位
            buy_signals = [
                (code, sig) for code, sig in daily_signals.items()
                if sig['action'] == 'BUY' 
                and sig['signal_strength'] >= strategy_config.get('confidence_threshold', 0.6)
                and code not in self.manager.positions
            ]
            buy_signals.sort(key=lambda x: x[1]['signal_strength'], reverse=True)
            
            # 计算可用于买入的资金和最大持仓数
            available_cash = self.manager.cash
            current_position_count = len([p for p in self.manager.positions.values() if p.shares > 0])
            max_positions = min(len(decision_universe), int(1 / self.config.min_position_per_stock))
            
            position_size = strategy_config.get('position_size', 0.3)
            single_position_capital = self.config.initial_capital * min(position_size, self.config.max_position_per_stock)
            # 使用配置的仓位分配方法（等权/信号强度/风险平价/凯利公式），与实盘一致
            buy_capitals_pre = []
            if buy_signals:
                buy_signals_dict = {code: sig for code, sig in buy_signals}
                target_weights = self.manager.calculate_target_weights(buy_signals_dict, daily_df_dict)
                total_weight_sum = sum(target_weights.values())
                if total_weight_sum > 0:
                    for code, _ in buy_signals:
                        w = target_weights.get(code, 0)
                        target_value = self.manager.total_value * w
                        buy_capitals_pre.append(min(target_value, single_position_capital))
                    total_desired = sum(buy_capitals_pre)
                    if total_desired > available_cash * 0.95:
                        scale = (available_cash * 0.95) / total_desired
                        buy_capitals_pre = [x * scale for x in buy_capitals_pre]
                else:
                    # 回退：凯利等全为 0 时按信号强度分配
                    total_confidence = sum(sig['signal_strength'] for _, sig in buy_signals)
                    if total_confidence > 0:
                        for code, sig in buy_signals:
                            share = sig['signal_strength'] / total_confidence
                            buy_capitals_pre.append(min(available_cash * share, single_position_capital))
                    else:
                        buy_capitals_pre = [single_position_capital] * len(buy_signals)
            
            for (code, sig), buy_capital_pre in zip(buy_signals, buy_capitals_pre):
                # 风控：本日审计不通过则不开新仓
                if risk_no_new_buys:
                    continue
                # 检查是否还能买入
                if current_position_count >= max_positions:
                    break
                if available_cash < single_position_capital * 0.3:
                    break
                
                # 质量过滤：买入前检查股票质量（缠论策略不做烂股过滤）
                if self.config.strategy_type != 'chanlun' and code in daily_df_dict:
                    quality_score, can_buy, quality_reason = evaluate_stock_quality(
                        code, daily_df_dict[code], sig, self.config.signal_threshold
                    )
                    if not can_buy:
                        print(f"\n[AI选股池回测] {date_str} 过滤烂股 {code}: {quality_reason} (质量评分{quality_score:.2f})")
                        continue
                
                price = current_prices[code]
                # 按预分配金额买入（表现好的 confidence 高，分配到的 buy_capital_pre 更大）
                buy_capital = min(buy_capital_pre, available_cash * 0.95)
                shares = int(buy_capital / price / 100) * 100  # 按手(100股)取整
                
                if shares < 100:
                    continue
                
                order = {
                    'action': 'BUY',
                    'stock_code': code,
                    'shares': shares,
                    'price': price,
                    'reason': sig['reason'],
                    'signal_strength': sig['signal_strength'],
                    'market_state': sig['market_state'],
                    'current_weight': 0,
                    'target_weight': shares * price / self.manager.total_value
                }
                
                if self.manager.execute_order(order, date_str):
                    print(f"\n[AI选股池回测] {date_str} 买入 {code} {shares}股 @ ¥{price:.2f}，原因: {sig['reason']}")
                    
                    # 通知前端：执行买入
                    if decision_callback:
                        decision_callback('trade', {
                            'date': date_str,
                            'stock_code': code,
                            'action': 'BUY',
                            'shares': shares,
                            'price': round(price, 2),
                            'profit_pct': 0,
                            'reason': sig['reason']
                        })
                    
                    # 更新状态（保留LLM缓存字段）
                    stock_states[code] = {
                        'entry_price': price,
                        'entry_date': date_str,
                        'holding_days': 0,
                        'highest_price': price,
                        'last_market_state': sig['market_state'],
                        'llm_cached_action': 'HOLD',
                        'llm_cached_confidence': 0.5,
                        'llm_cached_reason': '刚建仓，等待',
                        'last_llm_call_idx': idx
                    }
                    
                    available_cash -= shares * price
                    current_position_count += 1
            
            # 计算基准净值
            benchmark_value = None
            if self.benchmark_data is not None:
                try:
                    # 将日期转换为统一格式进行比较
                    if hasattr(date, 'strftime'):
                        date_str_for_match = date.strftime('%Y-%m-%d')
                    else:
                        date_str_for_match = str(date)
                    
                    # 找到基准数据中对应日期的收盘价（支持多种日期格式）
                    benchmark_df = self.benchmark_data.copy()
                    benchmark_df['date_str'] = benchmark_df['date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                    )
                    benchmark_row = benchmark_df[benchmark_df['date_str'] == date_str_for_match]
                    
                    if not benchmark_row.empty:
                        benchmark_close = benchmark_row.iloc[0]['close']
                        # 如果是第一天，记录初始价格
                        if len(self.benchmark_values) == 0:
                            self.benchmark_initial_price = benchmark_close
                        # 计算基准净值（相对于初始价格的收益率）
                        if self.benchmark_initial_price and self.benchmark_initial_price > 0:
                            benchmark_value = self.config.initial_capital * (benchmark_close / self.benchmark_initial_price)
                except Exception as e:
                    pass  # 如果计算失败，跳过
            
            # 记录基准净值
            if benchmark_value is not None:
                self.benchmark_values.append(benchmark_value)
            else:
                # 如果没有基准数据，使用上一个值或初始资金
                prev_benchmark = self.benchmark_values[-1] if self.benchmark_values else self.config.initial_capital
                self.benchmark_values.append(prev_benchmark)
            
            # 在记录快照前检查是否停止
            if abort_check and abort_check():
                backtest_aborted = True
                print(f"\n[AI选股池回测] 用户停止回测（记录快照前） @ Day {idx+1}/{total_days}")
                break
            
            # 记录每日快照
            self.manager.take_snapshot(date_str, daily_signals)
            # 更新历史峰值（供风控回撤计算）
            if self.manager.total_value > peak_total_value:
                peak_total_value = self.manager.total_value
        
        # 计算回测统计（含用户停止时的局部结果）
        elapsed_total = time.time() - start_time
        if backtest_aborted:
            print(f"\n[AI选股池回测] 已停止，返回局部结果，耗时: {elapsed_total:.2f}秒")
        else:
            print(f"\n[AI选股池回测] 回测完成！总耗时: {elapsed_total:.2f}秒")
        if use_llm_signals:
            print(f"[AI选股池回测] LLM调用次数: {llm_call_count}次")
        
        result = self._calculate_statistics()
        result['backtest_time'] = round(elapsed_total, 2)
        result['chart'] = self._generate_chart()
        if backtest_aborted:
            result['aborted'] = True
            result['aborted_message'] = '用户停止回测'
        
        # 添加LLM模式信息
        result['use_llm_signals'] = use_llm_signals
        result['signal_engine'] = "LLM大语言模型（DeepSeek）" if use_llm_signals else "规则算法（技术指标评分）"
        if use_llm_signals:
            result['llm_info'] = {
                'call_count': llm_call_count,
                'sample_rate': llm_sample_rate,
                'stocks_count': len(self.config.stocks)
            }
        result['factor_mining'] = {'enabled': False}
        return result
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算回测统计指标"""
        snapshots = self.manager.daily_snapshots
        if not snapshots:
            return {}
        
        # 权益曲线
        equity_values = [s.total_value for s in snapshots]
        dates = [s.date for s in snapshots]
        
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': equity_values
        })
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # 基本指标
        initial = self.config.initial_capital
        final = equity_values[-1]
        total_return = (final - initial) / initial * 100
        
        trading_days = len(snapshots)
        annual_return = ((final / initial) ** (252.0 / trading_days) - 1) * 100 if trading_days > 0 else 0
        
        # 夏普比率
        daily_rf = 0.03 / 252
        excess_returns = equity_df['daily_return'].dropna() - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].max()
        
        # ========== 基准对比统计 ==========
        benchmark_stats = {}
        if len(self.benchmark_values) == len(equity_values) and len(self.benchmark_values) > 0:
            benchmark_final = self.benchmark_values[-1]
            benchmark_total_return = (benchmark_final - initial) / initial * 100
            benchmark_annual_return = ((benchmark_final / initial) ** (252.0 / trading_days) - 1) * 100 if trading_days > 0 else 0
            
            # 基准收益率曲线
            benchmark_df = pd.DataFrame({
                'date': dates,
                'equity': self.benchmark_values
            })
            benchmark_df['daily_return'] = benchmark_df['equity'].pct_change()
            
            # 基准夏普比率
            benchmark_excess_returns = benchmark_df['daily_return'].dropna() - daily_rf
            benchmark_sharpe = np.sqrt(252) * benchmark_excess_returns.mean() / benchmark_excess_returns.std() if benchmark_excess_returns.std() > 0 else 0
            
            # 基准最大回撤
            benchmark_df['cummax'] = benchmark_df['equity'].cummax()
            benchmark_df['drawdown'] = (benchmark_df['cummax'] - benchmark_df['equity']) / benchmark_df['cummax'] * 100
            benchmark_max_drawdown = benchmark_df['drawdown'].max()
            
            # 超额收益（策略收益 - 基准收益）
            excess_return = total_return - benchmark_total_return
            excess_annual_return = annual_return - benchmark_annual_return
            
            benchmark_stats = {
                'benchmark_code': self.benchmark_code,
                'benchmark_name': '沪深300ETF',
                'benchmark_total_return': round(benchmark_total_return, 2),
                'benchmark_annual_return': round(benchmark_annual_return, 2),
                'benchmark_sharpe_ratio': round(benchmark_sharpe, 2),
                'benchmark_max_drawdown': round(benchmark_max_drawdown, 2),
                'excess_return': round(excess_return, 2),  # 超额收益
                'excess_annual_return': round(excess_annual_return, 2),  # 超额年化收益
                'benchmark_final_value': round(benchmark_final, 2)
            }
        
        # 交易统计
        trades = self.manager.trades
        sell_trades = [t for t in trades if t.action == 'SELL']
        
        # 简化胜率计算
        win_count = 0
        for t in sell_trades:
            # 找到对应的买入
            buy_trades_for_stock = [
                bt for bt in trades 
                if bt.action == 'BUY' and bt.stock_code == t.stock_code and bt.date < t.date
            ]
            if buy_trades_for_stock:
                last_buy = buy_trades_for_stock[-1]
                if t.price > last_buy.price:
                    win_count += 1
        
        win_rate = win_count / len(sell_trades) * 100 if sell_trades else 0
        
        # ========== 从快照构建各股票权益曲线，计算最大回撤 ==========
        stock_equity_curves = {}
        for s in snapshots:
            for code, pos in s.positions.items():
                if pos.get('shares', 0) > 0:
                    mv = pos.get('market_value', 0)
                    if code not in stock_equity_curves:
                        stock_equity_curves[code] = []
                    stock_equity_curves[code].append((s.date, mv))
        
        stock_max_drawdowns = {}
        for code, curve in stock_equity_curves.items():
            if len(curve) < 2:
                stock_max_drawdowns[code] = 0.0
                continue
            values = [v for _, v in curve]
            cummax = np.maximum.accumulate(values)
            drawdowns = (cummax - np.array(values)) / np.where(cummax > 0, cummax, 1) * 100
            stock_max_drawdowns[code] = float(np.max(drawdowns))
        
        # 各股票在回测期间的涨跌幅（期初→期末价格变动）
        stock_period_returns = {}
        data_dict = getattr(self, '_backtest_data_dict', {})
        start_date = dates[0] if dates else None
        end_date = dates[-1] if dates else None
        if start_date and end_date and data_dict:
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            for code in self.config.stocks:
                df = data_dict.get(code)
                if df is None or len(df) < 2:
                    stock_period_returns[code] = None
                    continue
                df = df.copy()
                df['date_str'] = df['date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                )
                start_row = df[df['date_str'] <= start_str].tail(1)
                end_row = df[df['date_str'] >= end_str].head(1)
                if start_row.empty or end_row.empty:
                    stock_period_returns[code] = None
                    continue
                p0 = float(start_row.iloc[0]['close'])
                p1 = float(end_row.iloc[0]['close'])
                if p0 > 0:
                    stock_period_returns[code] = round((p1 - p0) / p0 * 100, 2)
                else:
                    stock_period_returns[code] = None
        
        # ========== 统计所有历史成交股票的盈亏 ==========
        stock_performance = {}  # 每只股票的历史盈亏
        
        for t in trades:
            code = t.stock_code
            if code not in stock_performance:
                stock_performance[code] = {
                    'stock_code': code,
                    'buy_count': 0,
                    'sell_count': 0,
                    'total_buy_amount': 0.0,   # 总买入金额
                    'total_sell_amount': 0.0,  # 总卖出金额
                    'total_profit_amount': 0.0,  # 总盈亏金额
                    'trades': []  # 交易详情
                }
            
            perf = stock_performance[code]
            if t.action == 'BUY':
                perf['buy_count'] += 1
                perf['total_buy_amount'] += t.amount
                perf['trades'].append({
                    'date': t.date,
                    'action': 'BUY',
                    'shares': t.shares,
                    'price': round(t.price, 2),
                    'amount': round(t.amount, 2)
                })
            elif t.action == 'SELL':
                perf['sell_count'] += 1
                perf['total_sell_amount'] += t.amount
                perf['total_profit_amount'] += t.profit_amount
                perf['trades'].append({
                    'date': t.date,
                    'action': 'SELL',
                    'shares': t.shares,
                    'price': round(t.price, 2),
                    'amount': round(t.amount, 2),
                    'profit_amount': round(t.profit_amount, 2),
                    'profit_pct': round(t.profit_pct, 2)
                })
        
        # 计算每只股票的总收益率和当前状态
        # 确保所有配置的股票都出现在统计中，即使没有交易记录
        stock_summary = []
        all_configured_stocks = set(self.config.stocks)
        stocks_with_trades = set(stock_performance.keys())
        stocks_without_trades = all_configured_stocks - stocks_with_trades
        
        # 先处理有交易记录的股票
        for code, perf in stock_performance.items():
            # 当前是否仍有持仓
            current_pos = self.manager.positions.get(code)
            if current_pos and current_pos.shares > 0:
                # 有持仓：加上未实现盈亏
                unrealized_profit = current_pos.profit_loss
                status = '持仓中'
                current_value = current_pos.market_value
                current_shares = current_pos.shares
                current_price = current_pos.current_price
                current_profit_pct = current_pos.profit_loss_pct
            else:
                # 已清仓：只有已实现盈亏
                unrealized_profit = 0
                status = '已清仓'
                current_value = 0
                current_shares = 0
                current_price = 0
                current_profit_pct = 0
            
            # 正确计算总盈亏
            # 总盈亏 = 总卖出金额 + 当前持仓市值 - 总买入金额
            total_profit = perf['total_sell_amount'] + current_value - perf['total_buy_amount']
            
            # 已实现盈亏（卖出部分的盈亏）
            realized_profit = perf['total_profit_amount']
            
            # 计算收益率：总盈亏 / 平均单笔投入
            # 这样更能反映每笔交易的平均表现
            if perf['buy_count'] > 0:
                avg_buy_amount = perf['total_buy_amount'] / perf['buy_count']  # 平均每笔买入金额
                # 收益率 = 总盈亏 / 平均单笔投入
                total_profit_pct = (total_profit / avg_buy_amount) * 100
            else:
                total_profit_pct = 0
            
            max_dd = stock_max_drawdowns.get(code)
            period_ret = stock_period_returns.get(code)
            stock_summary.append({
                'stock_code': code,
                'status': status,
                'buy_count': perf['buy_count'],
                'sell_count': perf['sell_count'],
                'total_buy_amount': round(perf['total_buy_amount'], 2),
                'total_sell_amount': round(perf['total_sell_amount'], 2),
                'realized_profit': round(realized_profit, 2),  # 已实现盈亏
                'unrealized_profit': round(unrealized_profit, 2),  # 未实现盈亏
                'total_profit': round(total_profit, 2),  # 总盈亏
                'total_profit_pct': round(total_profit_pct, 2),  # 收益率
                'max_drawdown': round(max_dd, 2) if max_dd is not None else None,  # 持仓期间最大回撤
                'stock_period_return': period_ret,  # 该股票在回测期间的涨跌幅
                'current_shares': current_shares,
                'current_value': round(current_value, 2),
                'current_price': round(current_price, 2),
                'current_profit_pct': round(current_profit_pct, 2)  # 当前持仓收益率
            })
        
        # 处理没有交易记录的股票（显示为未操作）
        for code in stocks_without_trades:
            # 检查是否在数据字典中（数据加载成功）
            has_data = code in getattr(self, '_loaded_stocks', set())
            if has_data:
                # 数据加载成功但没有交易，可能是：
                # 1. 整个回测期间都没有产生买入信号
                # 2. 产生了信号但被质量过滤掉了
                # 3. 产生了信号但资金不足或达到最大持仓数
                status_text = "未操作（无交易信号）"
            else:
                status_text = "数据加载失败（数据不足或加载错误）"
            
            stock_summary.append({
                'stock_code': code,
                'status': status_text,
                'buy_count': 0,
                'sell_count': 0,
                'total_buy_amount': 0.0,
                'total_sell_amount': 0.0,
                'realized_profit': 0.0,
                'unrealized_profit': 0.0,
                'total_profit': 0.0,
                'total_profit_pct': 0.0,
                'max_drawdown': None,
                'stock_period_return': stock_period_returns.get(code),
                'current_shares': 0,
                'current_value': 0.0,
                'current_price': 0.0,
                'current_profit_pct': 0.0
            })
        
        # 按总盈亏金额排序（降序），未操作的排在最后
        stock_summary.sort(key=lambda x: (x['total_profit'] == 0, -x['total_profit']))
        
        # 持仓分析（仅当前持仓）
        final_positions = []
        for code, pos in self.manager.positions.items():
            final_positions.append({
                'stock_code': code,
                'shares': pos.shares,
                'market_value': round(pos.market_value, 2),
                'weight': round(pos.weight * 100, 1),
                'profit_loss_pct': round(pos.profit_loss_pct, 2)
            })
        
        result = {
            'pool_name': self.config.name,
            'stocks': self.config.stocks,
            'strategy': f"{self.config.strategy_type} + {self.config.risk_preference}",
            'allocation_method': self.config.allocation_method,
            'start_date': dates[0] if dates else '',
            'end_date': dates[-1] if dates else '',
            'trading_days': trading_days,
            'initial_capital': initial,
            'final_capital': round(final, 2),
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'final_positions': final_positions,
            # 所有历史成交股票的盈亏汇总
            'stock_summary': stock_summary,
            # 完整交易记录（用于展示决策和持仓变动）
            'all_trades': [
                {
                    'date': t.date,
                    'stock_code': t.stock_code,
                    'action': t.action,
                    'shares': t.shares,
                    'price': round(t.price, 2),
                    'amount': round(t.amount, 2),
                    'reason': t.reason,
                    'signal_strength': round(t.signal_strength, 4) if t.signal_strength else 0,
                    'market_state': t.market_state or '',
                    'before_weight': round(t.before_weight * 100, 1) if t.before_weight else 0,
                    'after_weight': round(t.after_weight * 100, 1) if t.after_weight else 0
                }
                for t in trades
            ],
            'recent_trades': [
                {
                    'date': t.date,
                    'stock_code': t.stock_code,
                    'action': t.action,
                    'shares': t.shares,
                    'price': round(t.price, 2),
                    'reason': t.reason
                }
                for t in trades[-10:]
            ],
            # 每日持仓快照（用于展示持仓变动）
            'position_snapshots': [
                {
                    'date': s.date,
                    'total_value': round(s.total_value, 2),
                    'cash': round(s.cash, 2),
                    'position_value': round(s.position_value, 2),
                    'daily_return': round(s.daily_return, 2),
                    'cumulative_return': round(s.cumulative_return, 2),
                    'positions': {
                        code: {
                            'shares': pos['shares'],
                            'weight': round(pos['weight'] * 100, 1),
                            'profit_loss_pct': round(pos['profit_loss_pct'], 2)
                        }
                        for code, pos in s.positions.items()
                    },
                    'signals': s.signals
                }
                for s in snapshots[::max(1, len(snapshots)//30)]  # 采样30个点
            ],
            'equity_curve': [
                {'date': s.date, 'value': round(s.total_value, 2)}
                for s in snapshots[::max(1, len(snapshots)//50)]  # 采样50个点
            ]
        }
        
        # 添加基准对比数据
        if benchmark_stats:
            result.update(benchmark_stats)
            # 添加基准净值曲线
            result['benchmark_curve'] = [
                {'date': dates[i], 'value': round(self.benchmark_values[i], 2)}
                for i in range(0, len(dates), max(1, len(dates)//50))  # 采样50个点
            ]
        
        return result
    
    def _generate_chart(self) -> str:
        """生成回测图表（Base64编码）"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64
            from utils.matplotlib_chinese import setup_chinese_font
            setup_chinese_font()
            
            snapshots = self.manager.daily_snapshots
            if not snapshots:
                return ""
            
            dates = [s.date for s in snapshots]
            equity = [s.total_value for s in snapshots]
            
            # 计算收益率
            returns = [(e / self.config.initial_capital - 1) * 100 for e in equity]
            
            # 持仓比例数据：收集所有出现过的股票代码，并计算每日现金比例与各股仓位比例
            all_codes = set()
            for s in snapshots:
                all_codes.update(s.positions.keys())
            all_codes = sorted(all_codes)
            cash_ratios = []
            weight_series = {code: [] for code in all_codes}
            for s in snapshots:
                tv = s.total_value if s.total_value > 0 else 1
                cash_ratios.append(s.cash / tv * 100)
                for code in all_codes:
                    pos = s.positions.get(code, {})
                    w = pos.get('weight')
                    if w is not None and w > 0:
                        weight_series[code].append(w * 100)
                    else:
                        mv = pos.get('market_value', 0)
                        weight_series[code].append(mv / tv * 100 if tv > 0 else 0)
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # 上图：权益曲线（含基准对比）
            ax1 = axes[0]
            ax1.plot(range(len(dates)), equity, 'b-', linewidth=1.5, label='组合净值')
            
            # 绘制基准曲线
            if len(self.benchmark_values) == len(equity):
                ax1.plot(range(len(dates)), self.benchmark_values, 'r--', linewidth=1.5, label='基准（沪深300ETF）', alpha=0.7)
            
            ax1.axhline(y=self.config.initial_capital, color='gray', linestyle='--', alpha=0.5, label='初始资金')
            ax1.fill_between(range(len(dates)), self.config.initial_capital, equity, 
                            where=[e >= self.config.initial_capital for e in equity],
                            color='green', alpha=0.3)
            ax1.fill_between(range(len(dates)), self.config.initial_capital, equity,
                            where=[e < self.config.initial_capital for e in equity],
                            color='red', alpha=0.3)
            ax1.set_ylabel('组合净值 (元)')
            ax1.set_title(f'选股池回测结果 - {self.config.name}（含基准对比）')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 中图：收益率曲线（含基准对比）
            ax2 = axes[1]
            ax2.fill_between(range(len(dates)), 0, returns, 
                            where=[r >= 0 for r in returns], color='green', alpha=0.3)
            ax2.fill_between(range(len(dates)), 0, returns,
                            where=[r < 0 for r in returns], color='red', alpha=0.3)
            ax2.plot(range(len(dates)), returns, 'b-', linewidth=1, label='组合收益率')
            
            # 绘制基准收益率曲线
            if len(self.benchmark_values) == len(equity):
                benchmark_returns = [(b / self.config.initial_capital - 1) * 100 for b in self.benchmark_values]
                ax2.plot(range(len(dates)), benchmark_returns, 'r--', linewidth=1, label='基准收益率', alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('累计收益率 (%)')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 下图：持仓比例图（堆叠面积）
            ax3 = axes[2]
            import numpy as np
            x = np.arange(len(dates))
            cash_arr = np.array(cash_ratios)
            stack_bottom = np.zeros(len(dates))
            ax3.fill_between(x, stack_bottom, cash_arr, label='现金', color='#cccccc', alpha=0.9)
            stack_bottom = cash_arr.copy()
            # 各股票用不同颜色（tab10 色板）
            try:
                cmap = plt.colormaps['tab10']
            except Exception:
                cmap = plt.cm.get_cmap('tab10', max(len(all_codes) + 1, 2))
            n_codes = max(len(all_codes), 1)
            for i, code in enumerate(all_codes):
                arr = np.array(weight_series[code])
                try:
                    color = cmap(i / n_codes)
                except Exception:
                    color = plt.cm.tab10(i % 10)
                ax3.fill_between(x, stack_bottom, stack_bottom + arr, label=code, color=color, alpha=0.85)
                stack_bottom = stack_bottom + arr
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('持仓比例 (%)')
            ax3.set_xlabel('交易日')
            ax3.set_title('持仓比例')
            ax3.legend(loc='upper left', fontsize=8, ncol=min(len(all_codes) + 1, 6))
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息（含基准对比）
            final_return = returns[-1] if returns else 0
            max_dd = max([(max(returns[:i+1]) - r) for i, r in enumerate(returns)]) if returns else 0
            
            stats_text = f'组合累计收益: {final_return:.1f}%  |  组合最大回撤: {max_dd:.1f}%'
            
            # 如果有基准数据，添加基准对比信息
            if len(self.benchmark_values) == len(equity):
                benchmark_final_return = (self.benchmark_values[-1] / self.config.initial_capital - 1) * 100
                benchmark_returns = [(b / self.config.initial_capital - 1) * 100 for b in self.benchmark_values]
                benchmark_max_dd = max([(max(benchmark_returns[:i+1]) - r) for i, r in enumerate(benchmark_returns)]) if benchmark_returns else 0
                excess_return = final_return - benchmark_final_return
                stats_text += f'  |  基准收益: {benchmark_final_return:.1f}%  |  超额收益: {excess_return:+.1f}%'
            
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12, hspace=0.3)
            
            # 保存为Base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return chart_base64
            
        except Exception as e:
            print(f"[回测] 生成图表失败: {e}")
            return ""


# ==================== 便捷函数 ====================

def create_stock_pool(stocks: List[str],
                      initial_capital: float = 1000000,
                      strategy_type: str = "adaptive",
                      risk_preference: str = "balanced",
                      allocation_method: str = "signal_strength",
                      name: str = "我的选股池",
                      universe_source: str = "manual",
                      universe_index: str = "",
                      industry_list: Optional[List[str]] = None,
                      leaders_per_industry: int = 1,
                      selection_mode: str = "none",
                      selection_top_n: int = 10,
                      selection_interval: int = 0,
                      score_weights: Optional[Dict[str, float]] = None,
                      factor_set: str = "hybrid",
                      strategy_meta: Optional[Dict[str, Any]] = None,
                      factor_profile: Optional[Dict[str, Any]] = None) -> PoolConfig:
    """
    创建选股池配置

    参数:
        stocks: 股票代码列表（universe_source 为 manual 时使用）
        initial_capital: 初始资金
        strategy_type: 策略类型
        risk_preference: 风险偏好
        allocation_method: 仓位分配方法
        name: 选股池名称
        universe_source: 股票来源 manual | index | industry
        universe_index: 指数代码（universe_source=index 时）
        industry_list: 行业名称列表（universe_source=industry 时），空为全行业
        leaders_per_industry: 每行业龙头数量（universe_source=industry 时）
        selection_mode: 选股模式 none | factor_top_n
        selection_top_n: 因子选股取前 N 只
        selection_interval: 重选间隔（0 表示仅初选）
        score_weights: 选股因子权重
        factor_set: 因子集合 style | trading | hybrid（风格与估值|情绪与交易）
        strategy_meta: 策略元信息
        factor_profile: 因子评估快照

    返回:
        PoolConfig实例
    """
    codes = [get_stock_code_from_name(s) for s in stocks] if stocks else []
    
    return PoolConfig(
        name=name,
        stocks=codes,
        initial_capital=initial_capital,
        strategy_type=strategy_type,
        risk_preference=risk_preference,
        allocation_method=allocation_method,
        universe_source=universe_source,
        universe_index=(universe_index or "").strip(),
        industry_list=industry_list,
        leaders_per_industry=leaders_per_industry,
        selection_mode=selection_mode,
        selection_top_n=selection_top_n,
        selection_interval=selection_interval,
        score_weights=score_weights,
        factor_set=factor_set,
        strategy_meta=strategy_meta or {},
        factor_profile=factor_profile or {},
    )


def backtest_stock_pool(stocks: List[str],
                        initial_capital: float = 1000000,
                        days: int = 252,
                        strategy_type: str = "adaptive",
                        risk_preference: str = "balanced",
                        allocation_method: str = "signal_strength",
                        rebalance_interval: int = 5,
                        use_ai: bool = False,
                        use_llm_signals: bool = False,
                        llm_sample_rate: int = 10,
                        high_win_rate_mode: bool = False,
                        llm=None,
                        decision_callback=None,
                        abort_check: Optional[Callable[[], bool]] = None,
                        universe_source: str = "manual",
                        universe_index: str = "",
                        industry_list: Optional[List[str]] = None,
                        leaders_per_industry: int = 1,
                        selection_mode: str = "none",
                        selection_top_n: int = 10,
                        selection_interval: int = 0,
                        score_weights: Optional[Dict[str, float]] = None,
                        factor_set: str = "hybrid",
                        weight_source: str = "manual",
                        model_name: str = "",
                        learned_weights: Optional[Dict[str, Dict[str, float]]] = None,
                        no_lookahead: bool = False,
                        start_date: str = "") -> Dict[str, Any]:
    """
    选股池回测便捷函数
    
    参数:
        stocks: 股票代码列表（universe_source=manual 时使用）
        initial_capital: 初始资金
        days: 回测天数
        strategy_type: 策略类型
        risk_preference: 风险偏好
        allocation_method: 仓位分配方法
        rebalance_interval: 再平衡间隔天数
        use_ai: 是否使用AI深度分析
        use_llm_signals: 是否使用LLM生成交易信号
        llm_sample_rate: LLM采样频率
        high_win_rate_mode: 是否启用高胜率模式
        llm: LLM实例
        decision_callback: 决策回调函数(event_type, data)，用于实时推送决策
        abort_check: 可选，无参可调用对象，返回 True 时停止回测并返回局部结果
        universe_source: 股票来源 manual | index | industry
        universe_index: 指数代码（universe_source=index 时）
        industry_list: 行业名称列表（universe_source=industry 时）
        leaders_per_industry: 每行业龙头数量（universe_source=industry 时）
        selection_mode: 选股模式 none | factor_top_n
        selection_top_n: 因子选股取前 N 只
        selection_interval: 因子重选间隔（0 表示仅初选）
        score_weights: 因子权重
        factor_set: 因子集合 style | trading | hybrid（风格与估值|情绪与交易）
        weight_source: 权重来源 manual | learned
        model_name: 学习模型名
        learned_weights: 学习模型权重
        no_lookahead: 是否启用无前视模式
        start_date: 回测起始日期（YYYY-MM-DD）
    
    返回:
        回测结果字典（若被停止则含 aborted=True）
    
    示例:
        result = backtest_stock_pool(
            stocks=['600519', '000858', '601318', '600036'],
            initial_capital=1000000,
            days=252,
            strategy_type='adaptive',
            risk_preference='balanced',
            use_llm_signals=True  # 使用LLM决策
        )
    """
    config = create_stock_pool(
        stocks=stocks or [],
        initial_capital=initial_capital,
        strategy_type=strategy_type,
        risk_preference=risk_preference,
        allocation_method=allocation_method,
        universe_source=universe_source,
        universe_index=universe_index,
        industry_list=industry_list,
        leaders_per_industry=leaders_per_industry,
        selection_mode=selection_mode,
        selection_top_n=selection_top_n,
        selection_interval=selection_interval,
        score_weights=score_weights,
        factor_set=factor_set,
    )
    
    engine = PoolBacktestEngine(config)
    return engine.run_backtest(
        days=days,
        rebalance_interval=rebalance_interval,
        use_ai=use_ai,
        use_llm_signals=use_llm_signals,
        llm_sample_rate=llm_sample_rate,
        high_win_rate_mode=high_win_rate_mode,
        llm=llm,
        decision_callback=decision_callback,
        abort_check=abort_check,
        selection_mode=selection_mode,
        selection_top_n=selection_top_n,
        selection_interval=selection_interval,
        score_weights=score_weights,
        factor_set=factor_set,
        weight_source=weight_source,
        model_name=model_name,
        learned_weights=learned_weights,
        no_lookahead=no_lookahead,
        start_date=start_date,
    )


def get_pool_signals(stocks: List[str],
                     risk_preference: str = "balanced",
                     use_ai: bool = False,
                     llm=None) -> Dict[str, Dict]:
    """
    获取选股池中所有股票的当前信号
    
    参数:
        stocks: 股票代码列表
        risk_preference: 风险偏好
        use_ai: 是否使用AI深度分析
        llm: LLM实例
    
    返回:
        股票代码到信号的映射
    """
    config = create_stock_pool(
        stocks=stocks,
        risk_preference=risk_preference
    )
    
    manager = StockPoolManager(config)
    
    # 加载数据（与因子挖掘一致：优先缓存；多只时略间隔以减轻限流）
    df_dict = {}
    for i, code in enumerate(config.stocks):
        if i > 0:
            time.sleep(0.4)
        try:
            stock_data = get_stock_data(code, 120, use_cache=True)
            if stock_data and len(stock_data) > 60:
                df = pd.DataFrame(stock_data)
                df = add_technical_indicators_to_df(df)
                df = df.dropna().reset_index(drop=True)
                df_dict[code] = df
        except Exception as e:
            print(f"[选股池] 加载 {code} 失败: {e}")
    
    # 生成信号
    signals = manager.generate_signals(df_dict, use_ai=use_ai, llm=llm)
    
    return signals


def get_pool_allocation(stocks: List[str],
                        total_capital: float = 1000000,
                        allocation_method: str = "signal_strength",
                        risk_preference: str = "balanced") -> Dict[str, Dict]:
    """
    获取选股池的建议持仓配置
    
    参数:
        stocks: 股票代码列表
        total_capital: 总资金
        allocation_method: 仓位分配方法
        risk_preference: 风险偏好
    
    返回:
        股票代码到配置建议的映射
    """
    # 获取信号
    signals = get_pool_signals(stocks, risk_preference)
    
    # 创建临时管理器计算权重
    config = create_stock_pool(
        stocks=stocks,
        initial_capital=total_capital,
        allocation_method=allocation_method,
        risk_preference=risk_preference
    )
    manager = StockPoolManager(config)
    
    # 计算目标权重
    target_weights = manager.calculate_target_weights(signals)
    
    # 获取当前价格
    current_prices = {}
    for code in config.stocks:
        try:
            stock_data = get_stock_data(code, 5, use_cache=True)
            if stock_data:
                current_prices[code] = stock_data[-1]['close']
        except:
            pass
    
    # 构建配置建议
    allocation = {}
    for code in config.stocks:
        weight = target_weights.get(code, 0)
        price = current_prices.get(code, 0)
        signal = signals.get(code, {})
        
        if weight > 0 and price > 0:
            target_value = total_capital * weight
            suggested_shares = int(target_value / price / 100) * 100
            
            allocation[code] = {
                'signal_type': signal.get('signal_type', 'Neutral'),
                'signal_strength': signal.get('signal_strength', 0),
                'strength_level': signal.get('strength_level', '无效'),
                'target_weight': round(weight * 100, 1),
                'target_value': round(target_value, 2),
                'current_price': round(price, 2),
                'suggested_shares': suggested_shares,
                'market_state': signal.get('market_state', ''),
                'risk_assessment': signal.get('risk_assessment', {}),
                'suggestion': signal.get('suggestion', {})
            }
        else:
            allocation[code] = {
                'signal_type': signal.get('signal_type', 'Neutral'),
                'signal_strength': signal.get('signal_strength', 0),
                'target_weight': 0,
                'suggested_shares': 0,
                'reason': '信号不满足买入条件' if signal.get('signal_type') != 'LongOpen' else '权重为0'
            }
    
    return allocation


# ==================== 选股池配置持久化 ====================

def save_pool_config(config: PoolConfig, filepath: str = None) -> str:
    """
    保存选股池配置到JSON文件
    
    参数:
        config: 选股池配置
        filepath: 文件路径，默认为 pools/{name}.json
    
    返回:
        保存的文件路径
    """
    import os
    
    if filepath is None:
        os.makedirs('pools', exist_ok=True)
        filepath = f"pools/{config.name.replace(' ', '_')}.json"
    
    config_dict = {
        'name': config.name,
        'stocks': config.stocks,
        'initial_capital': config.initial_capital,
        'strategy_type': config.strategy_type,
        'risk_preference': config.risk_preference,
        'allocation_method': config.allocation_method,
        'max_position_per_stock': config.max_position_per_stock,
        'min_position_per_stock': config.min_position_per_stock,
        'max_total_position': config.max_total_position,
        'min_cash_ratio': config.min_cash_ratio,
        'rebalance_threshold': config.rebalance_threshold,
        'signal_threshold': config.signal_threshold,
        'stop_loss_pct': config.stop_loss_pct,
        'take_profit_pct': config.take_profit_pct,
        'single_stop_loss_pct': config.single_stop_loss_pct,
        'single_take_profit_pct': config.single_take_profit_pct,
        'custom_weights': config.custom_weights,
        'selection_mode': config.selection_mode,
        'selection_top_n': config.selection_top_n,
        'selection_interval': config.selection_interval,
        'score_weights': config.score_weights,
        'factor_set': config.factor_set,
        'strategy_meta': config.strategy_meta or {},
        'factor_profile': config.factor_profile or {},
        'created_at': datetime.now().isoformat(),
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    print(f"[选股池] 配置已保存到: {filepath}")
    return filepath


def load_pool_config(filepath: str) -> PoolConfig:
    """
    从JSON文件加载选股池配置
    
    参数:
        filepath: 文件路径
    
    返回:
        PoolConfig实例
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = PoolConfig(
        name=config_dict.get('name', '默认选股池'),
        stocks=config_dict.get('stocks', []),
        initial_capital=config_dict.get('initial_capital', 1000000),
        strategy_type=config_dict.get('strategy_type', 'adaptive'),
        risk_preference=config_dict.get('risk_preference', 'balanced'),
        allocation_method=config_dict.get('allocation_method', 'signal_strength'),
        max_position_per_stock=config_dict.get('max_position_per_stock', 0.3),
        min_position_per_stock=config_dict.get('min_position_per_stock', 0.05),
        max_total_position=config_dict.get('max_total_position', 0.9),
        min_cash_ratio=config_dict.get('min_cash_ratio', 0.1),
        rebalance_threshold=config_dict.get('rebalance_threshold', 0.05),
        signal_threshold=config_dict.get('signal_threshold', 0.4),
        stop_loss_pct=config_dict.get('stop_loss_pct', 0.08),
        take_profit_pct=config_dict.get('take_profit_pct', 0.25),
        single_stop_loss_pct=config_dict.get('single_stop_loss_pct', 0.05),
        single_take_profit_pct=config_dict.get('single_take_profit_pct', 0.15),
        custom_weights=config_dict.get('custom_weights', {}),
        selection_mode=config_dict.get('selection_mode', 'none'),
        selection_top_n=config_dict.get('selection_top_n', 10),
        selection_interval=config_dict.get('selection_interval', 0),
        score_weights=config_dict.get('score_weights', None),
        factor_set=config_dict.get('factor_set', 'hybrid'),
        strategy_meta=config_dict.get('strategy_meta', {}),
        factor_profile=config_dict.get('factor_profile', {}),
    )
    
    print(f"[选股池] 配置已加载: {config.name}")
    return config


def list_saved_pools(directory: str = 'pools') -> List[Dict]:
    """
    列出所有保存的选股池配置
    
    参数:
        directory: 配置文件目录
    
    返回:
        选股池配置列表
    """
    import os
    
    pools = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    pools.append({
                        'name': config.get('name', 'Unknown'),
                        'stocks': config.get('stocks', []),
                        'initial_capital': config.get('initial_capital', 0),
                        'strategy_type': config.get('strategy_type', ''),
                        'filepath': filepath,
                        'created_at': config.get('created_at', ''),
                        'factor_set': config.get('factor_set', 'hybrid'),
                        'selection_mode': config.get('selection_mode', 'none'),
                        'selection_top_n': config.get('selection_top_n', 10),
                        'strategy_meta': config.get('strategy_meta', {}),
                    })
                except:
                    pass
    
    return pools


# ==================== AI审计友好的选股池报告 ====================

def generate_pool_report(stocks: List[str],
                         total_capital: float = 1000000,
                         risk_preference: str = "balanced") -> Dict[str, Any]:
    """
    生成AI审计友好的选股池分析报告
    
    参数:
        stocks: 股票代码列表
        total_capital: 总资金
        risk_preference: 风险偏好
    
    返回:
        完整的分析报告字典
    """
    print(f"\n{'='*60}")
    print(f"  选股池分析报告")
    print(f"{'='*60}")
    print(f"股票池: {stocks}")
    print(f"总资金: ¥{total_capital:,.0f}")
    print(f"风险偏好: {risk_preference}")
    print(f"{'='*60}\n")
    
    # 获取信号和配置建议
    signals = get_pool_signals(stocks, risk_preference)
    allocation = get_pool_allocation(stocks, total_capital, 'signal_strength', risk_preference)
    
    # 分析汇总
    buy_signals = []
    sell_signals = []
    neutral_signals = []
    
    for code, signal in signals.items():
        signal_type = signal.get('signal_type', 'Neutral')
        if signal_type == 'LongOpen':
            buy_signals.append(code)
        elif signal_type == 'ShortOpen':
            sell_signals.append(code)
        else:
            neutral_signals.append(code)
    
    # 计算配置汇总
    total_weight = sum(a.get('target_weight', 0) for a in allocation.values())
    total_investment = sum(a.get('target_value', 0) for a in allocation.values())
    
    report = {
        'report_time': datetime.now().isoformat(),
        'pool_summary': {
            'total_stocks': len(stocks),
            'total_capital': total_capital,
            'risk_preference': risk_preference,
        },
        'signal_summary': {
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'neutral_signals': len(neutral_signals),
            'buy_stocks': buy_signals,
            'sell_stocks': sell_signals,
            'neutral_stocks': neutral_signals,
        },
        'allocation_summary': {
            'total_weight': round(total_weight, 1),
            'total_investment': round(total_investment, 2),
            'cash_reserve': round(total_capital - total_investment, 2),
            'cash_ratio': round((total_capital - total_investment) / total_capital * 100, 1),
        },
        'stock_details': [],
        'recommendations': [],
    }
    
    # 每只股票的详细信息
    for code in stocks:
        signal = signals.get(code, {})
        alloc = allocation.get(code, {})
        
        detail = {
            'stock_code': code,
            'signal_type': signal.get('signal_type', 'Neutral'),
            'signal_strength': round(signal.get('signal_strength', 0), 4),
            'strength_level': signal.get('strength_level', '无效'),
            'market_state': signal.get('market_state', 'UNKNOWN'),
            'target_weight': alloc.get('target_weight', 0),
            'target_value': alloc.get('target_value', 0),
            'suggested_shares': alloc.get('suggested_shares', 0),
            'current_price': alloc.get('current_price', 0),
        }
        
        # 添加风险评估
        risk = signal.get('risk_assessment', {})
        if risk:
            detail['risk'] = {
                'max_drawdown': risk.get('最大回撤预测', 'N/A'),
                'volatility': risk.get('波动率等级', 'N/A'),
                'liquidity': risk.get('流动性评分', 0),
            }
        
        report['stock_details'].append(detail)
    
    # 生成建议
    if buy_signals:
        report['recommendations'].append({
            'type': '买入建议',
            'stocks': buy_signals,
            'message': f"以下{len(buy_signals)}只股票发出买入信号：{', '.join(buy_signals)}"
        })
    
    if sell_signals:
        report['recommendations'].append({
            'type': '卖出建议',
            'stocks': sell_signals,
            'message': f"以下{len(sell_signals)}只股票发出卖出信号：{', '.join(sell_signals)}"
        })
    
    if total_weight == 0:
        report['recommendations'].append({
            'type': '观望建议',
            'stocks': [],
            'message': "当前市场信号不明确，建议保持现金观望"
        })
    
    # 打印报告摘要
    print("【信号汇总】")
    print(f"  买入信号: {len(buy_signals)} 只 {buy_signals}")
    print(f"  卖出信号: {len(sell_signals)} 只 {sell_signals}")
    print(f"  中性信号: {len(neutral_signals)} 只")
    
    print("\n【配置建议】")
    print(f"  建议总仓位: {total_weight:.1f}%")
    print(f"  建议投资额: ¥{total_investment:,.0f}")
    print(f"  现金保留: ¥{total_capital - total_investment:,.0f} ({report['allocation_summary']['cash_ratio']:.1f}%)")
    
    print("\n【股票明细】")
    for detail in report['stock_details']:
        print(f"  {detail['stock_code']}: "
              f"{detail['signal_type']} "
              f"(强度{detail['signal_strength']:.2f}, {detail['strength_level']}) "
              f"→ 权重{detail['target_weight']:.1f}%, {detail['suggested_shares']}股")
    
    print("\n【操作建议】")
    for rec in report['recommendations']:
        print(f"  • {rec['message']}")
    
    return report


# ==================== 模拟仓调仓功能 ====================

def rebalance_account(account, current_prices: Dict[str, float],
                      df_dict: Dict[str, pd.DataFrame] = None,
                      risk_preference: str = "balanced",
                      use_llm: bool = False,
                      llm=None) -> Dict[str, Any]:
    """
    对模拟仓账户执行调仓（规则模式）
    
    参数:
        account: StockPoolSimAccount 账户对象
        current_prices: 当前价格字典 {stock_code: price}
        df_dict: 股票数据字典（用于质量评估，可选）
        risk_preference: 风险偏好
        use_llm: 是否使用LLM调仓
        llm: LLM实例
    
    返回:
        调仓结果字典
    """
    from modules.stock_pool.sim_account import StockPoolSimAccount
    
    if use_llm:
        return llm_rebalance_account(account, current_prices, df_dict, risk_preference, llm)
    
    # 规则模式调仓
    # 1. 创建临时管理器
    config = create_stock_pool(
        stocks=account.stock_pool or [],
        initial_capital=account.initial_capital,
        risk_preference=risk_preference
    )
    manager = StockPoolManager(config)
    
    # 2. 恢复账户状态到管理器
    manager.cash = account.cash
    for code, pos_data in account.positions.items():
        manager.positions[code] = StockPosition(
            stock_code=code,
            shares=pos_data['shares'],
            avg_cost=pos_data['avg_cost'],
            entry_date=pos_data.get('entry_date', '')
        )
        if code in current_prices:
            manager.positions[code].update_market_value(current_prices[code])
    
    # 3. 生成信号
    if df_dict is None:
        df_dict = {}
        for i, code in enumerate(account.stock_pool):
            if i > 0:
                time.sleep(0.4)
            try:
                from tools.stock_data import get_stock_data
                stock_data = get_stock_data(code, 120, use_cache=True)
                if stock_data and len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    from tools.technical_indicators import add_technical_indicators_to_df
                    df = add_technical_indicators_to_df(df)
                    df = df.dropna().reset_index(drop=True)
                    df_dict[code] = df
            except Exception as e:
                print(f"[调仓] 加载 {code} 数据失败: {e}")
    
    signals = manager.generate_signals(df_dict, use_ai=False)
    
    # 4. 生成调仓订单（使用质量过滤）
    orders = manager.generate_rebalance_orders(signals, current_prices, df_dict)
    
    # 5. 执行订单并更新账户
    executed_trades = []
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    for order in orders:
        if manager.execute_order(order, date_str):
            # 同步到账户
            code = order['stock_code']
            action = order['action']
            
            if action == 'BUY':
                result = account.buy(
                    code, order['shares'], order['price'],
                    reason=order.get('reason', '规则调仓买入')
                )
            elif action == 'SELL':
                result = account.sell(
                    code, order['shares'], order['price'],
                    reason=order.get('reason', '规则调仓卖出')
                )
            else:
                continue
            
            if result.get('success'):
                executed_trades.append(result['trade'])
    
    # 6. 更新账户最后调仓日期
    account.last_rebalance_date = datetime.now().isoformat()
    
    return {
        'success': True,
        'trades': executed_trades,
        'orders_count': len(orders),
        'executed_count': len(executed_trades),
        'account': account.to_dict()
    }


def llm_rebalance_account(account, current_prices: Dict[str, float],
                          df_dict: Dict[str, pd.DataFrame] = None,
                          risk_preference: str = "balanced",
                          llm=None) -> Dict[str, Any]:
    """
    使用LLM对模拟仓账户执行调仓
    
    参数:
        account: StockPoolSimAccount 账户对象
        current_prices: 当前价格字典 {stock_code: price}
        df_dict: 股票数据字典（用于LLM分析，可选）
        risk_preference: 风险偏好
        llm: LLM实例
    
    返回:
        调仓结果字典
    """
    from modules.stock_pool.sim_account import StockPoolSimAccount
    from prompts import llm_portfolio_decision
    
    if llm is None:
        from llm import llm as default_llm
        llm = default_llm
    
    # 1. 准备股票数据
    if df_dict is None:
        df_dict = {}
        for i, code in enumerate(account.stock_pool):
            if i > 0:
                time.sleep(0.4)
            try:
                from tools.stock_data import get_stock_data
                stock_data = get_stock_data(code, 120, use_cache=True)
                if stock_data and len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    from tools.technical_indicators import add_technical_indicators_to_df
                    df = add_technical_indicators_to_df(df)
                    df = df.dropna().reset_index(drop=True)
                    df_dict[code] = df
            except Exception as e:
                print(f"[LLM调仓] 加载 {code} 数据失败: {e}")
    
    # 2. 构建股票数据列表
    stocks_data = []
    for code in account.stock_pool:
        if code not in df_dict or code not in current_prices:
            continue
        
        df = df_dict[code]
        row = df.iloc[-1]
        price = current_prices[code]
        
        # 检查是否持仓
        has_position = code in account.positions and account.positions[code]['shares'] > 0
        entry_price = None
        holding_days = 0
        highest_price = None
        profit_pct = 0
        
        if has_position:
            pos = account.positions[code]
            entry_price = pos['avg_cost']
            entry_date = datetime.fromisoformat(pos['entry_date']) if isinstance(pos['entry_date'], str) else datetime.now()
            holding_days = (datetime.now() - entry_date).days
            highest_price = price  # 简化处理
            profit_pct = (price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        # 转换指标为字典
        indicators = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        
        stocks_data.append({
            'code': code,
            'indicators': indicators,
            'has_position': has_position,
            'entry_price': entry_price,
            'holding_days': holding_days,
            'highest_price': highest_price,
            'profit_pct': profit_pct
        })
    
    # 3. 获取当前持仓信息
    positions_info = {}
    for code, pos in account.positions.items():
        if pos['shares'] > 0 and code in current_prices:
            market_value = pos['shares'] * current_prices[code]
            total_value = account.get_total_equity(current_prices)
            weight = market_value / total_value if total_value > 0 else 0
            positions_info[code] = {
                'shares': pos['shares'],
                'value': market_value,
                'weight': weight
            }
    
    # 4. 获取A股市场环境（北向资金、成交量等）供LLM判断趋势
    market_context = None
    try:
        from tools.stock_data import get_market_context_for_llm
        market_context = get_market_context_for_llm(days=5)
    except Exception:
        pass
    # 5. 调用LLM组合决策
    total_value = account.get_total_equity(current_prices)
    portfolio_result = llm_portfolio_decision(
        stocks_data=stocks_data,
        positions=positions_info,
        cash=account.cash,
        total_value=total_value,
        strategy_type='adaptive',
        risk_preference=risk_preference,
        max_positions=min(len(account.stock_pool), 10),
        high_win_rate_mode=False,
        market_context=market_context,
        llm=llm
    )
    
    # 6. 执行LLM决策（加入时间因子：持仓不足 min_holding_days 且非止损时不卖，避免频繁调仓磨损）
    executed_trades = []
    date_str = datetime.now().strftime('%Y-%m-%d')
    min_holding_days = 5
    stop_loss_pct_threshold = 3.0  # 盈亏<=-3% 视为止损可破例卖出
    
    # 先执行卖出
    for code in portfolio_result.get('priority_sell', []):
        if code not in account.positions or account.positions[code]['shares'] <= 0 or code not in current_prices:
            continue
        pos = account.positions[code]
        entry_date = datetime.fromisoformat(pos['entry_date']) if isinstance(pos.get('entry_date'), str) else datetime.now()
        holding_days = (datetime.now() - entry_date).days
        entry_price = pos.get('avg_cost') or 0
        profit_pct = (current_prices[code] - entry_price) / entry_price * 100 if entry_price > 0 else 0
        # 短期持仓且未触发止损时不卖
        if holding_days < min_holding_days and profit_pct > -stop_loss_pct_threshold:
            continue
        decision = portfolio_result['decisions'].get(code, {})
        result = account.sell(
            code, None, current_prices[code],
            reason=f"[LLM] {decision.get('reason', 'LLM建议卖出')}"
        )
        if result.get('success'):
            executed_trades.append(result['trade'])
    
    # 再执行买入（按优先级；表现好的按 confidence 提高仓位）
    priority_buy = portfolio_result.get('priority_buy', [])
    if priority_buy and account.cash > 0:
        decisions = portfolio_result.get('decisions', {})
        # 按 confidence 分配可用资金：表现好的股票提高仓位
        confidences = [decisions.get(c, {}).get('confidence', 0.6) for c in priority_buy]
        total_conf = sum(confidences) if confidences else 0
        total_value = account.get_total_equity(current_prices)
        single_max_ratio = 0.35  # 单股最大仓位占比
        single_max_value = total_value * single_max_ratio if total_value > 0 else account.cash
        initial_cash = account.cash
        # 预计算每只股票的目标买入金额（按 confidence 比例，单股上限 single_max_value）
        if total_conf > 0:
            buy_values_pre = [min(initial_cash * (c / total_conf), single_max_value) for c in confidences]
        else:
            buy_values_pre = [min(initial_cash / len(priority_buy), single_max_value)] * len(priority_buy)
        available_cash = initial_cash
        
        for code, buy_value_pre in zip(priority_buy, buy_values_pre):
            if code not in current_prices:
                continue
            has_position = code in account.positions and account.positions[code]['shares'] > 0
            if has_position:
                continue
            buy_value = min(buy_value_pre, available_cash)
            if buy_value < 100 * current_prices[code]:  # 至少能买 100 股
                continue
            price = current_prices[code]
            shares = int(buy_value / price / 100) * 100  # A股最小单位100股
            if shares >= 100:
                decision = decisions.get(code, {})
                result = account.buy(
                    code, shares, price,
                    reason=f"[LLM] {decision.get('reason', 'LLM建议买入')}"
                )
                if result.get('success'):
                    executed_trades.append(result['trade'])
                    available_cash -= shares * price
    
    # 6. 更新账户最后调仓日期
    account.last_rebalance_date = datetime.now().isoformat()
    
    return {
        'success': True,
        'trades': executed_trades,
        'llm_analysis': portfolio_result.get('analysis', ''),
        'priority_buy': portfolio_result.get('priority_buy', []),
        'priority_sell': portfolio_result.get('priority_sell', []),
        'executed_count': len(executed_trades),
        'account': account.to_dict()
    }
