"""
AI审计友好的A股量化策略配置模块

本模块提供：
1. 策略类型配置（趋势跟踪/均值回归/AI自适应）
2. 风险偏好配置（激进/均衡/保守）
3. 市场状态分析（技术面/资金面/政策面）
4. 信号强度评估系统
5. 动态风险控制机制
6. AI审计友好的结构化输出

改进要点：
- 参数针对A股市场特性优化（高波动、政策影响）
- 多维度信号强度评分，量化评估信号可靠性
- 动态止损与仓位管理，根据市场状态自动调整
- 结构化JSON输出，便于AI审计系统解析和评估
- 完整的决策逻辑解释，增强策略可解释性
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ==================== 枚举类型定义 ====================

class MarketState(Enum):
    """市场状态枚举"""
    TRENDING = "TRENDING"      # 趋势市
    RANGING = "RANGING"        # 震荡市
    VOLATILE = "VOLATILE"      # 高波动市
    MIXED = "MIXED"            # 混合状态
    UNKNOWN = "UNKNOWN"        # 未知状态


class SignalType(Enum):
    """信号类型枚举"""
    LONG_OPEN = "LongOpen"     # 多头开仓
    SHORT_OPEN = "ShortOpen"   # 空头开仓（A股做空受限，主要用于减仓信号）
    NEUTRAL = "Neutral"        # 中性/观望
    LONG_CLOSE = "LongClose"   # 多头平仓
    SHORT_CLOSE = "ShortClose" # 空头平仓


class StrategyMode(Enum):
    """策略模式枚举"""
    TREND_FOLLOW = "TREND跟随"   # 趋势跟踪模式
    MEAN_REVERT = "MEAN回归"     # 均值回归模式
    ADAPTIVE = "AI自适应"        # AI自适应模式


# ==================== AI审计友好的策略参数配置 ====================

# 针对A股市场优化的策略参数
AI_AUDIT_STRATEGY_PARAMS = {
    # 移动平均线参数
    "MA周期": {
        "趋势市": [5, 21, 60],      # 短期、中期、长期均线
        "震荡市": [20, 60, 120],    # 中期、长期、超长期均线
        "高波动市": [5, 13, 30],    # 缩短参数提高灵敏度
        "低波动市": [10, 34, 100],  # 延长参数减少假信号
    },
    
    # 布林带参数
    "布林带": {
        "默认": {"周期": 20, "标准差": 2.0},
        "高波动市": {"周期": 14, "标准差": 3.0},
        "低波动市": {"周期": 26, "标准差": 1.5},
    },
    
    # RSI阈值
    "RSI阈值": {
        "默认": [30, 70],           # 超卖/超买阈值
        "高波动市": [25, 75],       # 放宽阈值减少误判
        "低波动市": [35, 65],       # 收紧阈值提高可靠性
    },
    
    # ATR止损倍数
    "ATR止损倍数": {
        "默认": 2.0,
        "主板": 2.0,
        "创业板": 2.5,
        "科创板": 3.0,
        "高波动市": 3.0,
        "低波动市": 1.5,
    },
    
    # 止损回撤阈值
    "止损回撤阈值": 0.05,  # 5%最大回撤控制
    
    # 动态权重配置 - 根据市场状态调整各指标权重
    "动态权重": {
        "趋势市": {
            "MA交叉": 0.7,
            "RSI": 0.1,
            "布林带位置": 0.1,
            "MACD柱状体": 0.1
        },
        "震荡市": {
            "MA交叉": 0.3,
            "RSI": 0.3,
            "布林带位置": 0.3,
            "MACD柱状体": 0.1
        },
        "高波动市": {
            "MA交叉": 0.2,
            "RSI": 0.2,
            "布林带位置": 0.5,
            "MACD柱状体": 0.1
        },
        "混合市": {
            "MA交叉": 0.4,
            "RSI": 0.2,
            "布林带位置": 0.2,
            "MACD柱状体": 0.2
        }
    }
}

# 市场状态判断参数
MARKET_STATE_PARAMS = {
    "布林带宽度阈值": {
        "趋势市": 0.10,      # 带宽>10%判定为趋势市
        "震荡市": 0.06,      # 带宽<6%判定为震荡市
        "极端行情": 0.15,    # 带宽>15%判定为极端行情
    },
    "ADX阈值": {
        "强趋势": 25,        # ADX>25表示强趋势
        "弱趋势": 20,        # ADX 20-25表示弱趋势
    },
    "ATR波动率阈值": {
        "高波动": 0.03,      # ATR/价格>3%为高波动
        "低波动": 0.015,     # ATR/价格<1.5%为低波动
    },
    "北向资金阈值": {
        "乐观情绪下沿": 10,  # 净流入>10亿为乐观
        "悲观情绪上沿": -10, # 净流出>10亿为悲观
    },
    "新闻情绪分位阈值": {
        "极度乐观": 0.9,     # 情绪分位>90%
        "极度悲观": 0.1,     # 情绪分位<10%
    }
}

# 信号强度评分阈值
SIGNAL_STRENGTH_THRESHOLDS = {
    "强信号": 0.8,           # 信号强度>0.8为强信号
    "中等信号": 0.6,         # 信号强度0.6-0.8为中等信号
    "弱信号": 0.4,           # 信号强度0.4-0.6为弱信号
    "无效信号": 0.4,         # 信号强度<0.4为无效信号
}

# 波动率等级权重映射（用于风险平价分配）
VOLATILITY_WEIGHT_MAP = {
    "低": 1.5,    # 低波动股票给更高权重
    "中": 1.0,    # 中等波动
    "高": 0.6,    # 高波动股票给更低权重
}

# 流动性评分阈值（日成交额，单位：元）
LIQUIDITY_THRESHOLDS = {
    "极高": 1e9,   # 10亿以上
    "高": 5e8,     # 5亿以上
    "中": 1e8,     # 1亿以上
    "低": 0,       # 1亿以下
}


def calculate_liquidity_score(daily_volume: float) -> float:
    """
    计算流动性评分
    
    参数:
        daily_volume: 日成交额（元）
    
    返回:
        流动性评分（0-1之间）
    """
    if daily_volume > LIQUIDITY_THRESHOLDS["极高"]:
        return 0.95
    elif daily_volume > LIQUIDITY_THRESHOLDS["高"]:
        return 0.85
    elif daily_volume > LIQUIDITY_THRESHOLDS["中"]:
        return 0.65
    else:
        return 0.35


def calculate_volatility_level(atr_ratio: float) -> str:
    """
    计算波动率等级
    
    参数:
        atr_ratio: ATR/收盘价比率
    
    返回:
        波动率等级（"高" | "中" | "低"）
    """
    if atr_ratio > MARKET_STATE_PARAMS["ATR波动率阈值"]["高波动"]:
        return "高"
    elif atr_ratio < MARKET_STATE_PARAMS["ATR波动率阈值"]["低波动"]:
        return "低"
    else:
        return "中"


# ==================== 策略类型配置 ====================

STRATEGY_TYPES = {
    'trend': {
        'name': '趋势跟踪',
        'description': '顺势而为，追涨杀跌，适合趋势明显的市场',
        # 趋势跟踪：看重均线排列和MACD方向
        'ma_weight': 2.5,        # 均线权重（高）
        'rsi_weight': 1.0,       # RSI权重（低）
        'macd_weight': 2.0,      # MACD权重（高）
        'bb_weight': 1.0,        # 布林带权重（低）
        'buy_on_breakout': True,  # 突破买入
        'sell_on_breakdown': True, # 跌破卖出
        'use_trailing_stop': True, # 使用移动止损
        # AI审计增强参数
        'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"]["趋势市"],
        'ma_periods': AI_AUDIT_STRATEGY_PARAMS["MA周期"]["趋势市"],
        'boll_params': AI_AUDIT_STRATEGY_PARAMS["布林带"]["默认"],
        'rsi_thresholds': AI_AUDIT_STRATEGY_PARAMS["RSI阈值"]["默认"],
    },
    'mean_reversion': {
        'name': '均值回归',
        'description': '逆势操作，低买高卖，适合震荡市场',
        # 均值回归：看重RSI超买超卖和布林带
        'ma_weight': 1.0,        # 均线权重（低）
        'rsi_weight': 2.5,       # RSI权重（高）
        'macd_weight': 1.0,      # MACD权重（低）
        'bb_weight': 2.5,        # 布林带权重（高）
        'buy_on_oversold': True,  # 超卖买入
        'sell_on_overbought': True, # 超买卖出
        'use_trailing_stop': False, # 不使用移动止损（等待回归）
        # AI审计增强参数
        'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"]["震荡市"],
        'ma_periods': AI_AUDIT_STRATEGY_PARAMS["MA周期"]["震荡市"],
        'boll_params': AI_AUDIT_STRATEGY_PARAMS["布林带"]["默认"],
        'rsi_thresholds': AI_AUDIT_STRATEGY_PARAMS["RSI阈值"]["默认"],
    },
    'adaptive': {
        'name': 'AI自适应',
        'description': 'AI自动判断市场状态，动态切换趋势跟踪或均值回归',
        # 自适应：权重根据市场状态动态调整
        'ma_weight': 1.5,        # 初始中等权重
        'rsi_weight': 1.5,       # 初始中等权重
        'macd_weight': 1.5,      # 初始中等权重
        'bb_weight': 1.5,        # 初始中等权重
        'buy_on_breakout': True,  # 根据市场状态动态调整
        'sell_on_breakdown': True,
        'buy_on_oversold': True,
        'sell_on_overbought': True,
        'use_trailing_stop': True,
        'is_adaptive': True,      # 标记为自适应策略
        # AI审计增强参数
        'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"],
        'ma_periods': AI_AUDIT_STRATEGY_PARAMS["MA周期"]["趋势市"],  # 初始使用趋势市参数
        'boll_params': AI_AUDIT_STRATEGY_PARAMS["布林带"]["默认"],
        'rsi_thresholds': AI_AUDIT_STRATEGY_PARAMS["RSI阈值"]["默认"],
    },
    'chanlun': {
        'name': '缠论',
        'description': '基于缠论思想（分型、笔、中枢、背驰与三类买卖点），偏结构+背驰',
        # 缠论：背驰用 MACD，结构看均线
        'ma_weight': 1.5,        # 均线权重（适中，趋势结构）
        'rsi_weight': 1.0,      # RSI权重
        'macd_weight': 2.5,     # MACD权重（高，背驰核心）
        'bb_weight': 1.0,       # 布林带权重
        'buy_on_breakout': True,
        'sell_on_breakdown': True,
        'use_trailing_stop': True,
        # AI审计增强参数
        'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"]["趋势市"],
        'ma_periods': AI_AUDIT_STRATEGY_PARAMS["MA周期"]["趋势市"],
        'boll_params': AI_AUDIT_STRATEGY_PARAMS["布林带"]["默认"],
        'rsi_thresholds': AI_AUDIT_STRATEGY_PARAMS["RSI阈值"]["默认"],
    }
}

# ==================== 风险偏好配置 ====================

RISK_PREFERENCES = {
    'aggressive': {
        'name': '激进进取',
        'description': '高仓位、宽止损、无固定止盈、仅用移动止盈',
        'position_size': 0.9,      # 90%仓位
        'stop_loss_pct': 0.08,     # 8%止损
        'take_profit_pct': 999.0,  # 不设固定止盈（设为无限大）
        'trailing_stop_pct': 0.08, # 8%移动止盈回撤（激进策略允许更大回撤）
        'trailing_activate_pct': 0.10, # 盈刚超过10%时激活移动止盈
        'buy_threshold': 3.5,      # 买入阈值（低，更容易触发）
        'sell_threshold': 4.5,     # 卖出阈值（高，更难触发）
        'confidence_threshold': 0.55, # 信心阈值（低）
        'max_holding_days': 60,    # 最大持仓天数（更长，让利润充分奖跑）
        'min_profit_for_timeout': 8, # 超时最低盈利要求
        # AI审计增强参数
        'atr_stop_multiplier': 3.0,   # ATR止损倍数（激进）
        'max_drawdown_limit': 0.15,   # 最大回撤限制15%
        'liquidity_threshold': 0.5,   # 流动性评分阈值（宽松）
    },
    'balanced': {
        'name': '均衡稳健',
        'description': '中等仓位、适度止损、平衡风险收益',
        'position_size': 0.7,      # 70%仓位
        'stop_loss_pct': 0.05,     # 5%止损
        'take_profit_pct': 0.12,   # 12%止盈
        'trailing_stop_pct': 0.05, # 5%移动止盈回撤
        'buy_threshold': 4.0,      # 买入阈值（中）
        'sell_threshold': 4.0,     # 卖出阈值（中）
        'confidence_threshold': 0.6, # 信心阈值（中）
        'max_holding_days': 25,    # 最大持仓天数（中）
        'min_profit_for_timeout': 3, # 超时最低盈利要求
        # AI审计增强参数
        'atr_stop_multiplier': 2.0,   # ATR止损倍数（均衡）
        'max_drawdown_limit': 0.10,   # 最大回撤限制10%
        'liquidity_threshold': 0.65,  # 流动性评分阈值（适中）
    },
    'conservative': {
        'name': '稳健保守',
        'description': '低仓位、严格止损、保本优先',
        'position_size': 0.5,      # 50%仓位
        'stop_loss_pct': 0.03,     # 3%止损
        'take_profit_pct': 0.08,   # 8%止盈
        'trailing_stop_pct': 0.03, # 3%移动止盈回撤（保守策略更紧的回撤控制）
        'buy_threshold': 5.0,      # 买入阈值（高，更难触发）
        'sell_threshold': 3.5,     # 卖出阈值（低，更容易止损）
        'confidence_threshold': 0.7, # 信心阈值（高）
        'max_holding_days': 15,    # 最大持仓天数（短）
        'min_profit_for_timeout': 2, # 超时最低盈利要求
        # AI审计增强参数
        'atr_stop_multiplier': 1.5,   # ATR止损倍数（保守）
        'max_drawdown_limit': 0.05,   # 最大回撤限制5%
        'liquidity_threshold': 0.8,   # 流动性评分阈值（严格）
    }
}


# ==================== AI审计友好的信号强度评估 ====================

class SignalStrengthEvaluator:
    """
    信号强度评估器
    
    提供多维度的信号强度评分，包括：
    - MA交叉评分
    - RSI评分
    - MACD评分
    - 布林带位置评分
    
    支持根据市场状态动态调整权重
    """
    
    def __init__(self, market_state: str = "混合市"):
        """
        初始化信号强度评估器
        
        参数:
            market_state: 市场状态 ("趋势市" | "震荡市" | "高波动市" | "混合市")
        """
        self.market_state = market_state
        self.weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"].get(
            market_state, 
            AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"]
        )
    
    def evaluate_ma_crossover(self, ma_short: np.ndarray, ma_mid: np.ndarray, 
                               ma_long: np.ndarray = None) -> Tuple[float, str]:
        """
        评估MA交叉信号
        
        参数:
            ma_short: 短期均线数组
            ma_mid: 中期均线数组
            ma_long: 长期均线数组（可选）
        
        返回:
            (评分, 信号说明)
        """
        if len(ma_short) < 2 or len(ma_mid) < 2:
            return 0.0, "数据不足"
        
        score = 0.0
        explanation = []
        
        # 金叉检测：短期均线上穿中期均线
        if ma_short[-1] > ma_mid[-1] and ma_short[-2] <= ma_mid[-2]:
            score = 1.0
            explanation.append("短期均线上穿中期均线（金叉）")
            
            # 如果长期均线也向上，额外加分
            if ma_long is not None and len(ma_long) >= 1:
                if ma_mid[-1] > ma_long[-1]:
                    score += 0.3
                    explanation.append("中期均线在长期均线上方，趋势增强")
        
        # 死叉检测：短期均线下穿中期均线
        elif ma_short[-1] < ma_mid[-1] and ma_short[-2] >= ma_mid[-2]:
            score = -1.0
            explanation.append("短期均线下穿中期均线（死叉）")
            
            if ma_long is not None and len(ma_long) >= 1:
                if ma_mid[-1] < ma_long[-1]:
                    score -= 0.3
                    explanation.append("中期均线在长期均线下方，下跌趋势增强")
        
        # 多头排列检测
        elif ma_short[-1] > ma_mid[-1]:
            if ma_long is not None and ma_mid[-1] > ma_long[-1]:
                score = 0.8
                explanation.append("均线多头排列（短>中>长）")
            else:
                score = 0.5
                explanation.append("短期均线在中期均线上方")
        
        # 空头排列检测
        elif ma_short[-1] < ma_mid[-1]:
            if ma_long is not None and ma_mid[-1] < ma_long[-1]:
                score = -0.8
                explanation.append("均线空头排列（短<中<长）")
            else:
                score = -0.5
                explanation.append("短期均线在中期均线下方")
        
        return score, "；".join(explanation) if explanation else "无明显信号"
    
    def evaluate_rsi(self, rsi_value: float, 
                     thresholds: List[int] = None) -> Tuple[float, str]:
        """
        评估RSI信号
        
        参数:
            rsi_value: RSI值
            thresholds: [超卖阈值, 超买阈值]，默认[30, 70]
        
        返回:
            (评分, 信号说明)
        """
        if thresholds is None:
            thresholds = AI_AUDIT_STRATEGY_PARAMS["RSI阈值"]["默认"]
        
        oversold, overbought = thresholds
        
        if rsi_value <= oversold:
            # 超卖区域
            score = 1.0 - (rsi_value / oversold)  # RSI越低，分数越高
            return score, f"RSI={rsi_value:.1f}，处于超卖区域（<{oversold}），可能反弹"
        
        elif rsi_value >= overbought:
            # 超买区域
            score = -1.0 * (1 - (100 - rsi_value) / (100 - overbought))
            return score, f"RSI={rsi_value:.1f}，处于超买区域（>{overbought}），可能回调"
        
        elif 40 <= rsi_value <= 60:
            # 中性区域
            return 0.0, f"RSI={rsi_value:.1f}，处于中性区域"
        
        elif oversold < rsi_value < 40:
            # 接近超卖
            score = 0.3
            return score, f"RSI={rsi_value:.1f}，接近超卖区域"
        
        else:
            # 接近超买
            score = -0.3
            return score, f"RSI={rsi_value:.1f}，接近超买区域"
    
    def evaluate_macd(self, macd_hist: np.ndarray) -> Tuple[float, str]:
        """
        评估MACD柱状体信号
        
        参数:
            macd_hist: MACD柱状体数组
        
        返回:
            (评分, 信号说明)
        """
        if len(macd_hist) < 2:
            return 0.0, "数据不足"
        
        current = macd_hist[-1]
        previous = macd_hist[-2]
        
        # MACD柱状体>0且放大
        if current > 0 and current > previous:
            return 1.0, f"MACD柱状体>0且放大（{current:.4f}），多头动量增强"
        
        # MACD柱状体>0但收缩
        elif current > 0 and current <= previous:
            return 0.5, f"MACD柱状体>0但收缩（{current:.4f}），多头动量减弱"
        
        # MACD柱状体<0但收缩（绝对值减小）
        elif current < 0 and abs(current) < abs(previous):
            return 0.3, f"MACD柱状体<0但收缩（{current:.4f}），空头动量减弱"
        
        # MACD柱状体<0且放大
        elif current < 0 and abs(current) >= abs(previous):
            return -1.0, f"MACD柱状体<0且放大（{current:.4f}），空头动量增强"
        
        return 0.0, f"MACD柱状体中性（{current:.4f}）"
    
    def evaluate_bollinger_position(self, close: float, 
                                     bb_upper: float, bb_lower: float, 
                                     bb_middle: float) -> Tuple[float, str]:
        """
        评估布林带位置信号
        
        参数:
            close: 当前收盘价
            bb_upper: 布林带上轨
            bb_lower: 布林带下轨
            bb_middle: 布林带中轨
        
        返回:
            (评分, 信号说明)
        """
        bb_width = bb_upper - bb_lower
        if bb_width == 0:
            return 0.0, "布林带宽度为0，数据异常"
        
        # 计算价格在布林带中的位置（0=下轨，1=上轨）
        position = (close - bb_lower) / bb_width
        
        if close > bb_upper:
            # 突破上轨
            return -0.8, f"价格突破布林带上轨（{bb_upper:.2f}），可能超买"
        
        elif close < bb_lower:
            # 突破下轨
            return 0.8, f"价格跌破布林带下轨（{bb_lower:.2f}），可能超卖"
        
        elif close > bb_middle and position > 0.7:
            # 在中轨上方靠近上轨
            return -0.3, f"价格在布林带上半区（位置{position:.1%}），偏强但需警惕"
        
        elif close < bb_middle and position < 0.3:
            # 在中轨下方靠近下轨
            return 0.3, f"价格在布林带下半区（位置{position:.1%}），偏弱但可能反弹"
        
        elif abs(close - bb_middle) / bb_middle < 0.01:
            # 接近中轨
            return 0.0, f"价格接近布林带中轨（{bb_middle:.2f}），方向不明"
        
        else:
            # 中性区域
            direction = "上方" if close > bb_middle else "下方"
            return 0.1 if close > bb_middle else -0.1, f"价格在中轨{direction}（位置{position:.1%}）"
    
    def calculate_total_score(self, component_scores: Dict[str, float]) -> float:
        """
        计算加权总分
        
        参数:
            component_scores: 各分项评分字典
        
        返回:
            加权总分（-1到1之间）
        """
        total = 0.0
        total_weight = 0.0
        
        weight_mapping = {
            "MA交叉": self.weights.get("MA交叉", 0.25),
            "RSI": self.weights.get("RSI", 0.25),
            "布林带位置": self.weights.get("布林带位置", 0.25),
            "MACD柱状体": self.weights.get("MACD柱状体", 0.25),
        }
        
        for key, score in component_scores.items():
            if key in weight_mapping:
                weight = weight_mapping[key]
                total += score * weight
                total_weight += weight
        
        if total_weight > 0:
            # 归一化到-1到1之间
            return max(-1.0, min(1.0, total / total_weight))
        
        return 0.0
    
    def evaluate(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合评估信号强度
        
        参数:
            indicators: 技术指标字典，包含：
                - ma_short, ma_mid, ma_long: 均线数组
                - rsi: RSI值
                - macd_hist: MACD柱状体数组
                - close: 当前收盘价
                - bb_upper, bb_lower, bb_middle: 布林带
        
        返回:
            包含评估结果的字典
        """
        component_scores = {}
        explanations = {}
        
        # MA交叉评估
        if all(k in indicators for k in ['ma_short', 'ma_mid']):
            ma_score, ma_exp = self.evaluate_ma_crossover(
                indicators['ma_short'],
                indicators['ma_mid'],
                indicators.get('ma_long')
            )
            component_scores["MA交叉"] = ma_score
            explanations["MA交叉"] = ma_exp
        
        # RSI评估
        if 'rsi' in indicators:
            rsi_score, rsi_exp = self.evaluate_rsi(
                indicators['rsi'],
                indicators.get('rsi_thresholds')
            )
            component_scores["RSI"] = rsi_score
            explanations["RSI"] = rsi_exp
        
        # MACD评估
        if 'macd_hist' in indicators:
            macd_score, macd_exp = self.evaluate_macd(indicators['macd_hist'])
            component_scores["MACD柱状体"] = macd_score
            explanations["MACD柱状体"] = macd_exp
        
        # 布林带位置评估
        if all(k in indicators for k in ['close', 'bb_upper', 'bb_lower', 'bb_middle']):
            bb_score, bb_exp = self.evaluate_bollinger_position(
                indicators['close'],
                indicators['bb_upper'],
                indicators['bb_lower'],
                indicators['bb_middle']
            )
            component_scores["布林带位置"] = bb_score
            explanations["布林带位置"] = bb_exp
        
        # 计算总分
        total_score = self.calculate_total_score(component_scores)
        
        # 确定信号强度等级
        abs_score = abs(total_score)
        if abs_score >= SIGNAL_STRENGTH_THRESHOLDS["强信号"]:
            strength_level = "强"
        elif abs_score >= SIGNAL_STRENGTH_THRESHOLDS["中等信号"]:
            strength_level = "中等"
        elif abs_score >= SIGNAL_STRENGTH_THRESHOLDS["弱信号"]:
            strength_level = "弱"
        else:
            strength_level = "无效"
        
        return {
            "总分": round(total_score, 4),
            "强度等级": strength_level,
            "分项评分": {k: round(v, 4) for k, v in component_scores.items()},
            "分项说明": explanations,
            "权重分配": self.weights,
            "市场状态": self.market_state
        }


# ==================== 动态风险控制 ====================

class DynamicRiskController:
    """
    动态风险控制器
    
    提供：
    - 动态止损计算（ATR止损、布林带止损、跟踪止损）
    - 仓位管理（根据信号强度、流动性、波动率调整）
    - 风险评估（最大回撤预测、波动率等级）
    """
    
    def __init__(self, risk_preference: str = "balanced", board_type: str = "主板"):
        """
        初始化风险控制器
        
        参数:
            risk_preference: 风险偏好 ("aggressive" | "balanced" | "conservative")
            board_type: 板块分类 ("主板" | "创业板" | "科创板")
        """
        self.risk_config = RISK_PREFERENCES.get(risk_preference, RISK_PREFERENCES["balanced"])
        self.board_type = board_type
    
    def calculate_stop_loss(self, current_price: float, atr_value: float,
                            market_state: str, signal_type: str,
                            bb_upper: float = None, bb_lower: float = None) -> Dict[str, Any]:
        """
        计算动态止损价格
        
        参数:
            current_price: 当前价格
            atr_value: ATR值
            market_state: 市场状态
            signal_type: 信号类型
            bb_upper: 布林带上轨（可选）
            bb_lower: 布林带下轨（可选）
        
        返回:
            止损计算结果字典
        """
        result = {
            "止损类型": "",
            "止损价格": 0.0,
            "止损比例": 0.0,
            "计算方法": "",
        }
        
        # 根据板块和波动率确定ATR倍数
        base_multiplier = AI_AUDIT_STRATEGY_PARAMS["ATR止损倍数"].get(
            self.board_type, 2.0
        )
        
        # 根据市场状态调整
        if market_state == "TRENDING":
            # 趋势市使用ATR止损
            multiplier = base_multiplier
            
            if signal_type in ["LongOpen", "多头开仓"]:
                stop_price = current_price - multiplier * atr_value
                result["止损类型"] = "ATR止损（趋势市多头）"
            else:
                stop_price = current_price + multiplier * atr_value
                result["止损类型"] = "ATR止损（趋势市空头）"
            
            result["止损价格"] = round(stop_price, 2)
            result["止损比例"] = round(abs(stop_price - current_price) / current_price, 4)
            result["计算方法"] = f"当前价格 {'减' if signal_type in ['LongOpen', '多头开仓'] else '加'} {multiplier}倍ATR（{atr_value:.2f}）"
        
        else:
            # 震荡市优先使用布林带止损
            if bb_lower is not None and bb_upper is not None:
                if signal_type in ["LongOpen", "多头开仓"]:
                    stop_price = bb_lower
                    result["止损类型"] = "布林带下轨止损（震荡市多头）"
                else:
                    stop_price = bb_upper
                    result["止损类型"] = "布林带上轨止损（震荡市空头）"
                
                result["止损价格"] = round(stop_price, 2)
                result["止损比例"] = round(abs(stop_price - current_price) / current_price, 4)
                result["计算方法"] = f"使用布林带{'下轨' if signal_type in ['LongOpen', '多头开仓'] else '上轨'}作为止损"
            else:
                # 没有布林带数据时使用ATR止损
                multiplier = base_multiplier * 0.8  # 震荡市适当收紧
                
                if signal_type in ["LongOpen", "多头开仓"]:
                    stop_price = current_price - multiplier * atr_value
                else:
                    stop_price = current_price + multiplier * atr_value
                
                result["止损类型"] = "ATR止损（震荡市）"
                result["止损价格"] = round(stop_price, 2)
                result["止损比例"] = round(abs(stop_price - current_price) / current_price, 4)
                result["计算方法"] = f"当前价格 {'减' if signal_type in ['LongOpen', '多头开仓'] else '加'} {multiplier}倍ATR"
        
        return result
    
    def calculate_position_size(self, signal_strength: float, 
                                 daily_volume: float,
                                 atr_ratio: float,
                                 max_drawdown_pred: float = 0.05) -> Dict[str, Any]:
        """
        计算建议仓位
        
        参数:
            signal_strength: 信号强度（-1到1）
            daily_volume: 日成交额（元）
            atr_ratio: ATR/收盘价比率
            max_drawdown_pred: 预测最大回撤
        
        返回:
            仓位计算结果字典
        """
        base_position = self.risk_config["position_size"]
        
        # 1. 根据信号强度调整
        strength_factor = 0.5 + abs(signal_strength) * 0.5  # 0.5-1.0
        position_ratio = base_position * strength_factor
        
        # 2. 根据流动性调整（使用共享函数）
        liquidity_score = calculate_liquidity_score(daily_volume)
        position_ratio *= liquidity_score
        
        # 3. 根据波动率调整（使用共享函数）
        volatility_level = calculate_volatility_level(atr_ratio)
        volatility_factor = {
            "高": 0.7,
            "中": 1.0,
            "低": 1.1  # 低波动可适当提高仓位
        }.get(volatility_level, 1.0)
        position_ratio *= volatility_factor
        
        # 4. 根据预测回撤调整
        max_allowed = self.risk_config["max_drawdown_limit"]
        if max_drawdown_pred > max_allowed:
            drawdown_factor = max_allowed / max_drawdown_pred
            position_ratio *= drawdown_factor
        
        # 5. 确保仓位在合理范围内
        position_ratio = max(0.1, min(1.0, position_ratio))
        
        return {
            "建议仓位比例": round(position_ratio, 2),
            "流动性评分": round(liquidity_score, 2),
            "波动率等级": volatility_level,
            "信号强度因子": round(strength_factor, 2),
            "调整说明": f"基础仓位{base_position:.0%}，经信号强度({strength_factor:.2f})、流动性({liquidity_score:.2f})、波动率({volatility_level})调整后为{position_ratio:.0%}"
        }
    
    def assess_risk(self, atr_ratio: float, daily_volume: float,
                    is_st: bool = False, board_type: str = "主板") -> Dict[str, Any]:
        """
        综合风险评估
        
        参数:
            atr_ratio: ATR/收盘价比率
            daily_volume: 日成交额
            is_st: 是否ST股票
            board_type: 板块分类
        
        返回:
            风险评估结果字典
        """
        # 波动率等级（使用共享函数）
        volatility_level = calculate_volatility_level(atr_ratio)
        
        # 根据波动率等级计算回撤预测
        drawdown_multiplier = {"高": 5, "中": 3, "低": 2}.get(volatility_level, 3)
        max_drawdown_pred = f"{atr_ratio * drawdown_multiplier * 100:.1f}%"
        
        # 流动性评分（使用共享函数）
        liquidity_score = calculate_liquidity_score(daily_volume)
        
        return {
            "最大回撤预测": max_drawdown_pred,
            "波动率等级": volatility_level,
            "流动性评分": round(liquidity_score, 2),
            "合规性指标": {
                "板块分类": board_type,
                "日成交额": f"{daily_volume/1e8:.1f}亿",
                "ST状态": str(is_st)
            }
        }


def get_strategy_config(strategy_type='trend', risk_preference='balanced', high_win_rate_mode=False):
    """
    获取策略配置
    
    参数：
        strategy_type: 策略类型 ('trend' | 'mean_reversion' | 'adaptive')
        risk_preference: 风险偏好 ('aggressive' | 'balanced' | 'conservative')
        high_win_rate_mode: 是否启用高胜率模式（更严格的入场条件）
    
    返回：
        合并后的策略配置字典
    """
    # 获取基础配置
    strategy = STRATEGY_TYPES.get(strategy_type, STRATEGY_TYPES['trend']).copy()
    risk = RISK_PREFERENCES.get(risk_preference, RISK_PREFERENCES['balanced']).copy()
    
    # 仓位管理参数（根据风险偏好调整）
    position_management = {
        'aggressive': {
            'max_position_ratio': 1.0,      # 最大仓位100%
            'min_position_ratio': 0.0,
            'add_position_threshold': 0.02,  # 2%盈利即可加仓
            'reduce_position_ratio': 0.5,    # 减仓比例50%
        },
        'balanced': {
            'max_position_ratio': 0.8,       # 最大仓位80%
            'min_position_ratio': 0.0,
            'add_position_threshold': 0.03,  # 3%盈利才加仓
            'reduce_position_ratio': 0.5,
        },
        'conservative': {
            'max_position_ratio': 0.6,       # 最大仓位60%
            'min_position_ratio': 0.0,
            'add_position_threshold': 0.05,  # 5%盈利才加仓
            'reduce_position_ratio': 0.4,    # 减仓比例40%
        }
    }
    
    pm = position_management.get(risk_preference, position_management['balanced'])
    
    # 合并配置（min_holding_days：最短持仓天数，避免频繁调仓磨损）
    config = {
        **strategy,
        **risk,
        **pm,  # 添加仓位管理参数
        'strategy_type': strategy_type,
        'risk_preference': risk_preference,
        'full_name': f"{strategy['name']} + {risk['name']}",
        'high_win_rate_mode': high_win_rate_mode,
        'min_holding_days': 5,  # 持仓不足此天数且未止损时不卖
    }
    
    # 高胜率模式：应用更严格的参数
    if high_win_rate_mode:
        config.update({
            # 更严格的止损止盈（2:1盈亏比）
            'stop_loss_pct': 0.03,         # 3%止损
            'take_profit_pct': 0.06,       # 6%止盈
            'trailing_stop_pct': 0.025,    # 2.5%移动止盈
            'trailing_activate_pct': 0.04, # 4%激活移动止盈
            
            # 更高的信号阈值
            'confidence_threshold': 0.70,  # 70%信心才入场
            'buy_threshold': 5.0,          # 更高的买入门槛
            'sell_threshold': 3.5,         # 更低的卖出门槛
            
            # 更短的持仓时间
            'max_holding_days': 15,
            'min_profit_for_timeout': 1,
            
            # 高胜率特有配置
            'min_bullish_indicators': 3,   # 至少3个指标看多
            'require_trend_alignment': True,
            'require_volume_confirm': True,
            
            'full_name': f"{strategy['name']} + {risk['name']} (高胜率模式)"
        })
    
    return config


# ==================== AI审计友好的市场状态分析 ====================

def analyze_market_state(df_window, use_ai=False, llm=None, 
                         north_flow=None, news_sentiment=None):
    """
    AI审计友好的市场状态分析
    
    分析当前市场是处于趋势状态还是震荡状态，结合技术面、资金面、政策面进行多维度判断
    
    参数：
        df_window: 用于分析的数据窗口（DataFrame，包含技术指标）
        use_ai: 是否使用AI进行深度分析（True时调用LLM，False时使用规则）
        llm: LLM实例（当use_ai=True时需要提供）
        north_flow: 北向资金净流入数据（可选，单位：亿元）
        news_sentiment: 新闻情绪分位数据（可选，0-1之间）
    
    返回：
        {
            'market_state': 'TRENDING' | 'RANGING' | 'VOLATILE' | 'MIXED',
            'suggested_strategy': 'trend' | 'mean_reversion',
            'confidence': 0.0-1.0,
            'analysis': 分析说明,
            'indicators': 关键指标数据,
            'multi_dimension_analysis': {  # AI审计友好的多维度分析
                '技术面': {...},
                '资金面': {...},
                '政策面': {...}
            },
            'signal_weights': 当前市场状态下的信号权重配置
        }
    """
    # 计算市场状态判断指标
    if len(df_window) < 20:
        return {
            'market_state': 'UNKNOWN',
            'suggested_strategy': 'trend',
            'confidence': 0.5,
            'analysis': '数据不足，默认使用趋势策略',
            'indicators': {},
            'multi_dimension_analysis': {},
            'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"]
        }
    
    close = df_window['close']
    high = df_window['high']
    low = df_window['low']
    volume = df_window['volume'] if 'volume' in df_window else None
    
    # 1. 布林带宽度（波动性指标）
    bb_upper = df_window['BB_Upper'].iloc[-1] if 'BB_Upper' in df_window else close.rolling(20).mean().iloc[-1] + 2 * close.rolling(20).std().iloc[-1]
    bb_lower = df_window['BB_Lower'].iloc[-1] if 'BB_Lower' in df_window else close.rolling(20).mean().iloc[-1] - 2 * close.rolling(20).std().iloc[-1]
    bb_middle = df_window['BB_Middle'].iloc[-1] if 'BB_Middle' in df_window else close.rolling(20).mean().iloc[-1]
    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0  # 布林带宽度比率
    bb_width_pct = bb_width * 100  # 转换为百分比
    
    # 2. ADX（趋势强度指标）- 简化计算
    price_changes = close.diff()
    up_moves = (price_changes > 0).sum()
    down_moves = (price_changes < 0).sum()
    direction_consistency = abs(up_moves - down_moves) / len(price_changes) * 100
    adx_estimate = min(50, direction_consistency * 2)
    
    # 3. 价格相对于均线的位置
    ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
    price = close.iloc[-1]
    price_vs_ma20 = (price - ma20) / ma20 * 100 if ma20 != 0 else 0
    price_vs_ma60 = (price - ma60) / ma60 * 100 if ma60 != 0 else 0
    
    # 4. 波动率（ATR百分比）
    atr = df_window['ATR'].iloc[-1] if 'ATR' in df_window else (high - low).rolling(14).mean().iloc[-1]
    atr_pct = atr / price * 100 if price != 0 else 0
    atr_ratio = atr / price if price != 0 else 0
    
    # 5. RSI偏离度
    rsi = df_window['RSI'].iloc[-1] if 'RSI' in df_window else 50
    rsi_deviation = abs(rsi - 50)
    
    # 6. MACD趋势
    macd_hist = df_window['MACD_Hist'].iloc[-1] if 'MACD_Hist' in df_window else 0
    macd_hist_prev = df_window['MACD_Hist'].iloc[-5] if 'MACD_Hist' in df_window and len(df_window) >= 5 else 0
    macd_expanding = abs(macd_hist) > abs(macd_hist_prev)
    
    # 7. 成交量变化（如果有数据）
    volume_change = 0
    if volume is not None and len(volume) >= 5:
        vol_ma5 = volume.rolling(5).mean().iloc[-1]
        vol_current = volume.iloc[-1]
        volume_change = (vol_current - vol_ma5) / vol_ma5 * 100 if vol_ma5 != 0 else 0
    
    indicators = {
        'bb_width': round(bb_width, 4),
        'bb_width_pct': round(bb_width_pct, 2),
        'adx_estimate': round(adx_estimate, 2),
        'price_vs_ma20': round(price_vs_ma20, 2),
        'price_vs_ma60': round(price_vs_ma60, 2),
        'atr_pct': round(atr_pct, 2),
        'atr_ratio': round(atr_ratio, 4),
        'rsi': round(rsi, 2),
        'rsi_deviation': round(rsi_deviation, 2),
        'macd_expanding': macd_expanding,
        'macd_hist': round(macd_hist, 4),
        'volume_change_pct': round(volume_change, 2),
        'current_price': round(price, 2),
        'ma5': round(ma5, 2),
        'ma20': round(ma20, 2),
        'ma60': round(ma60, 2),
        'bb_upper': round(bb_upper, 2),
        'bb_lower': round(bb_lower, 2),
        'bb_middle': round(bb_middle, 2),
    }
    
    # ========== 多维度市场状态判断 ==========
    trend_score = 0
    range_score = 0
    analysis_points = []
    
    # 技术面分析
    tech_analysis = {}
    
    # 布林带宽度判断
    trend_threshold = MARKET_STATE_PARAMS["布林带宽度阈值"]["趋势市"]
    range_threshold = MARKET_STATE_PARAMS["布林带宽度阈值"]["震荡市"]
    extreme_threshold = MARKET_STATE_PARAMS["布林带宽度阈值"]["极端行情"]
    
    if bb_width > trend_threshold:
        trend_score += 2
        tech_analysis["布林带宽度"] = f"{bb_width_pct:.1f}%>{trend_threshold*100:.0f}%，波动较大，适合趋势策略"
        analysis_points.append(f"布林带宽度{bb_width_pct:.1f}%>{trend_threshold*100:.0f}%，波动较大，适合趋势策略")
    elif bb_width < range_threshold:
        range_score += 2
        tech_analysis["布林带宽度"] = f"{bb_width_pct:.1f}%<{range_threshold*100:.0f}%，波动收窄，适合均值回归"
        analysis_points.append(f"布林带宽度{bb_width_pct:.1f}%<{range_threshold*100:.0f}%，波动收窄，适合均值回归")
    else:
        trend_score += 0.5
        range_score += 0.5
        tech_analysis["布林带宽度"] = f"{bb_width_pct:.1f}%处于中性区间"
        analysis_points.append(f"布林带宽度{bb_width_pct:.1f}%处于中性区间")
    
    # ADX估算值判断
    adx_strong = MARKET_STATE_PARAMS["ADX阈值"]["强趋势"]
    adx_weak = MARKET_STATE_PARAMS["ADX阈值"]["弱趋势"]
    
    if adx_estimate > adx_strong:
        trend_score += 2
        tech_analysis["趋势强度"] = f"ADX估算值{adx_estimate:.1f}>{adx_strong}，市场有明显趋势"
        analysis_points.append(f"趋势强度指标{adx_estimate:.1f}>{adx_strong}，市场有明显趋势")
    elif adx_estimate < 15:
        range_score += 2
        tech_analysis["趋势强度"] = f"ADX估算值{adx_estimate:.1f}<15，市场震荡"
        analysis_points.append(f"趋势强度指标{adx_estimate:.1f}<15，市场震荡")
    else:
        tech_analysis["趋势强度"] = f"ADX估算值{adx_estimate:.1f}，趋势中等"
    
    # 价格与均线关系
    if abs(price_vs_ma20) > 5:
        trend_score += 1.5
        direction = "上方" if price_vs_ma20 > 0 else "下方"
        tech_analysis["均线位置"] = f"价格偏离MA20达{abs(price_vs_ma20):.1f}%（{direction}），趋势明显"
        analysis_points.append(f"价格偏离MA20达{abs(price_vs_ma20):.1f}%（{direction}），趋势明显")
    else:
        range_score += 1
        tech_analysis["均线位置"] = f"价格接近MA20（偏离{abs(price_vs_ma20):.1f}%），震荡格局"
        analysis_points.append(f"价格接近MA20（偏离{abs(price_vs_ma20):.1f}%），震荡格局")
    
    # 均线排列判断
    if ma5 > ma20 > ma60:
        trend_score += 1.5
        tech_analysis["均线排列"] = "多头排列（MA5>MA20>MA60），上涨趋势"
    elif ma5 < ma20 < ma60:
        trend_score += 1.5
        tech_analysis["均线排列"] = "空头排列（MA5<MA20<MA60），下跌趋势"
    else:
        range_score += 1
        tech_analysis["均线排列"] = "均线交织，方向不明"
    
    # RSI极端值
    if rsi_deviation > 20:
        if bb_width < 0.08:
            range_score += 1.5
            tech_analysis["RSI状态"] = f"RSI={rsi:.0f}，在低波动市场适合均值回归"
            analysis_points.append(f"RSI达到{rsi:.0f}，在低波动市场适合均值回归")
        else:
            trend_score += 1
            tech_analysis["RSI状态"] = f"RSI={rsi:.0f}，在高波动市场可能继续趋势"
            analysis_points.append(f"RSI达到{rsi:.0f}，在高波动市场可能继续趋势")
    else:
        tech_analysis["RSI状态"] = f"RSI={rsi:.0f}，处于中性区间"
    
    # MACD动量
    if macd_expanding:
        trend_score += 1
        tech_analysis["MACD动量"] = f"MACD柱状体放大（{macd_hist:.4f}），动量增强"
        analysis_points.append("MACD柱状体放大，动量增强")
    else:
        range_score += 0.5
        tech_analysis["MACD动量"] = f"MACD柱状体收缩（{macd_hist:.4f}），动量减弱"
        analysis_points.append("MACD柱状体收缩，动量减弱")
    
    # 成交量分析
    if volume_change > 30:
        trend_score += 1
        tech_analysis["成交量"] = f"放量{volume_change:.0f}%，趋势可能延续"
    elif volume_change < -30:
        range_score += 0.5
        tech_analysis["成交量"] = f"缩量{abs(volume_change):.0f}%，动能减弱"
    else:
        tech_analysis["成交量"] = f"量能变化{volume_change:.0f}%，正常范围"
    
    # 资金面分析
    capital_analysis = {}
    if north_flow is not None:
        bullish_threshold = MARKET_STATE_PARAMS["北向资金阈值"]["乐观情绪下沿"]
        bearish_threshold = MARKET_STATE_PARAMS["北向资金阈值"]["悲观情绪上沿"]
        
        if north_flow > bullish_threshold:
            trend_score += 1
            capital_analysis["北向资金"] = f"净流入{north_flow:.1f}亿>{bullish_threshold}亿，外资情绪乐观"
        elif north_flow < bearish_threshold:
            trend_score += 0.5  # 流出也可能是下跌趋势
            capital_analysis["北向资金"] = f"净流出{abs(north_flow):.1f}亿，外资情绪悲观"
        else:
            capital_analysis["北向资金"] = f"净流入{north_flow:.1f}亿，外资情绪中性"
    
    # 政策面分析
    policy_analysis = {}
    if news_sentiment is not None:
        bullish_sentiment = MARKET_STATE_PARAMS["新闻情绪分位阈值"]["极度乐观"]
        bearish_sentiment = MARKET_STATE_PARAMS["新闻情绪分位阈值"]["极度悲观"]
        
        if news_sentiment > bullish_sentiment:
            trend_score += 0.5
            policy_analysis["新闻情绪"] = f"情绪分位{news_sentiment:.2f}>{bullish_sentiment}，政策面偏向积极"
        elif news_sentiment < bearish_sentiment:
            trend_score += 0.5
            policy_analysis["新闻情绪"] = f"情绪分位{news_sentiment:.2f}<{bearish_sentiment}，政策面偏向谨慎"
        else:
            policy_analysis["新闻情绪"] = f"情绪分位{news_sentiment:.2f}，政策面中性"
    
    # 综合判断
    if trend_score > range_score + 2:
        market_state = 'TRENDING'
        suggested_strategy = 'trend'
        confidence = min(0.9, 0.5 + (trend_score - range_score) * 0.08)
        signal_weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"]["趋势市"]
    elif range_score > trend_score + 2:
        market_state = 'RANGING'
        suggested_strategy = 'mean_reversion'
        confidence = min(0.9, 0.5 + (range_score - trend_score) * 0.08)
        signal_weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"]["震荡市"]
    else:
        market_state = 'MIXED'
        if atr_ratio > MARKET_STATE_PARAMS["ATR波动率阈值"]["高波动"]:
            suggested_strategy = 'trend'
            market_state = 'VOLATILE'
            signal_weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"]["高波动市"]
        else:
            suggested_strategy = 'mean_reversion'
            signal_weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"]
        confidence = 0.6
    
    # 极端行情判断
    if bb_width > extreme_threshold or atr_ratio > 0.05:
        if market_state == 'TRENDING':
            market_state = 'VOLATILE'
            signal_weights = AI_AUDIT_STRATEGY_PARAMS["动态权重"]["高波动市"]
        analysis_points.append(f"⚠️ 检测到极端行情（布林带宽度{bb_width_pct:.1f}%），建议谨慎操作")
    
    analysis = f"市场状态: {market_state}\n" + "\n".join(f"• {p}" for p in analysis_points)
    
    # 构建多维度分析结果
    multi_dimension_analysis = {
        "技术面": tech_analysis,
        "资金面": capital_analysis if capital_analysis else {"状态": "无数据"},
        "政策面": policy_analysis if policy_analysis else {"状态": "无数据"},
    }
    
    result = {
        'market_state': market_state,
        'suggested_strategy': suggested_strategy,
        'confidence': round(confidence, 2),
        'analysis': analysis,
        'indicators': indicators,
        'trend_score': round(trend_score, 2),
        'range_score': round(range_score, 2),
        'multi_dimension_analysis': multi_dimension_analysis,
        'signal_weights': signal_weights,
    }
    
    # ========== 可选：使用AI进行深度分析 ==========
    if use_ai and llm:
        try:
            ai_result = _ai_analyze_market_state(
                indicators, analysis_points, trend_score, range_score, 
                llm, north_flow, news_sentiment
            )
            if ai_result:
                result.update(ai_result)
        except Exception as e:
            print(f"[AI自适应] AI分析失败，使用规则判断: {e}")
    
    return result


def _ai_analyze_market_state(indicators, analysis_points, trend_score, range_score, 
                              llm, north_flow=None, news_sentiment=None):
    """
    使用LLM进行AI审计友好的市场状态深度分析
    """
    # 构建多维度分析数据
    capital_info = ""
    if north_flow is not None:
        capital_info = f"\n- 北向资金净流入: {north_flow:.1f}亿元"
    
    policy_info = ""
    if news_sentiment is not None:
        policy_info = f"\n- 新闻情绪分位: {news_sentiment:.2f}（0-1，越高越乐观）"
    
    ai_prompt = f"""
你是一位专业的A股量化交易市场分析师，需要判断当前市场状态并推荐最佳交易策略。
你的分析结果将用于AI审计系统的决策参考，请确保输出清晰、可追溯、逻辑完整。

**一、技术面指标数据**：
- 布林带宽度: {indicators['bb_width_pct']}%
  * >10%: 高波动，适合趋势跟踪
  * <6%: 低波动，适合均值回归
- 趋势强度(ADX估算): {indicators['adx_estimate']}
  * >25: 强趋势市场
  * <20: 震荡市场
- 价格vs MA20: {indicators['price_vs_ma20']:+.2f}%
- 价格vs MA60: {indicators['price_vs_ma60']:+.2f}%
- ATR波动率: {indicators['atr_pct']:.2f}%
  * >3%: 高波动
  * <1.5%: 低波动
- RSI: {indicators['rsi']:.1f}（偏离中值{indicators['rsi_deviation']:.1f}）
- MACD动量: {'放大中' if indicators['macd_expanding'] else '收缩中'}（{indicators['macd_hist']:.4f}）
- 成交量变化: {indicators['volume_change_pct']:+.1f}%

**二、资金面数据**：{capital_info if capital_info else "暂无数据"}

**三、政策面数据**：{policy_info if policy_info else "暂无数据"}

**四、规则分析结果**：
- 趋势得分: {trend_score}
- 震荡得分: {range_score}
- 分析要点: {'; '.join(analysis_points[:5])}

**判断任务**：
请综合以上多维度信息，输出AI审计友好的分析结果：

1. **市场状态判断**（TRENDING-趋势市 / RANGING-震荡市 / VOLATILE-高波动 / MIXED-混合）
2. **推荐策略**（trend-趋势跟踪 / mean_reversion-均值回归）
3. **判断置信度**（0.5-0.95）
4. **多维度分析摘要**（技术面、资金面、政策面各一句话）
5. **风险提示**（当前市场环境下的主要风险）
6. **参数适配建议**（针对当前市场状态的参数调整建议）

**输出JSON格式**（请严格按此格式输出）：
{{
    "market_state": "TRENDING/RANGING/VOLATILE/MIXED",
    "suggested_strategy": "trend/mean_reversion",
    "confidence": 0.5-0.95,
    "ai_reasoning": "综合判断理由（100字以内）",
    "multi_dimension_summary": {{
        "技术面": "一句话总结",
        "资金面": "一句话总结",
        "政策面": "一句话总结"
    }},
    "risk_warning": "主要风险提示",
    "parameter_suggestion": "参数调整建议"
}}
"""
    
    try:
        response = llm.invoke(ai_prompt)
        content = response.content
        
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            ai_decision = json.loads(json_match.group())
            
            # 根据AI判断更新信号权重
            ai_market_state = ai_decision.get('market_state', 'MIXED')
            weight_mapping = {
                'TRENDING': '趋势市',
                'RANGING': '震荡市',
                'VOLATILE': '高波动市',
                'MIXED': '混合市'
            }
            weight_key = weight_mapping.get(ai_market_state, '混合市')
            
            return {
                'market_state': ai_market_state,
                'suggested_strategy': ai_decision.get('suggested_strategy', 'trend'),
                'confidence': ai_decision.get('confidence', 0.6),
                'ai_reasoning': ai_decision.get('ai_reasoning', ''),
                'ai_multi_dimension_summary': ai_decision.get('multi_dimension_summary', {}),
                'ai_risk_warning': ai_decision.get('risk_warning', ''),
                'ai_parameter_suggestion': ai_decision.get('parameter_suggestion', ''),
                'signal_weights': AI_AUDIT_STRATEGY_PARAMS["动态权重"].get(
                    weight_key, 
                    AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"]
                )
            }
    except Exception as e:
        print(f"[AI自适应] AI解析失败: {e}")
    
    return None


def get_adaptive_strategy_config(df_window, base_risk_preference='balanced', use_ai=False, llm=None,
                                  north_flow=None, news_sentiment=None):
    """
    获取AI审计友好的自适应策略配置
    
    根据市场状态动态生成策略配置，输出标准化的AI审计友好格式
    
    参数：
        df_window: 数据窗口
        base_risk_preference: 基础风险偏好
        use_ai: 是否使用AI分析
        llm: LLM实例
        north_flow: 北向资金净流入（可选）
        news_sentiment: 新闻情绪分位（可选）
    
    返回：
        (策略配置字典, 市场分析结果)
    """
    # 分析市场状态
    market_analysis = analyze_market_state(
        df_window, use_ai=use_ai, llm=llm,
        north_flow=north_flow, news_sentiment=news_sentiment
    )
    suggested_strategy = market_analysis['suggested_strategy']
    market_state = market_analysis['market_state']
    confidence = market_analysis['confidence']
    signal_weights = market_analysis.get('signal_weights', AI_AUDIT_STRATEGY_PARAMS["动态权重"]["混合市"])
    
    # 获取建议策略的基础配置
    base_config = get_strategy_config(suggested_strategy, base_risk_preference)
    
    # 根据市场状态和置信度微调配置
    config = base_config.copy()
    config['is_adaptive'] = True
    config['detected_market_state'] = market_state
    config['strategy_confidence'] = confidence
    config['original_strategy'] = suggested_strategy
    config['signal_weights'] = signal_weights
    
    # 根据市场状态选择MA周期
    ma_periods_key = "趋势市" if market_state == "TRENDING" else \
                     "震荡市" if market_state == "RANGING" else \
                     "高波动市" if market_state == "VOLATILE" else "趋势市"
    config['ma_periods'] = AI_AUDIT_STRATEGY_PARAMS["MA周期"].get(ma_periods_key, [5, 21, 60])
    
    # 根据市场状态选择RSI阈值
    rsi_key = "高波动市" if market_state == "VOLATILE" else "默认"
    config['rsi_thresholds'] = AI_AUDIT_STRATEGY_PARAMS["RSI阈值"].get(rsi_key, [30, 70])
    
    # 根据市场状态选择布林带参数
    boll_key = "高波动市" if market_state == "VOLATILE" else \
               "低波动市" if market_state == "RANGING" else "默认"
    config['boll_params'] = AI_AUDIT_STRATEGY_PARAMS["布林带"].get(boll_key, {"周期": 20, "标准差": 2.0})
    
    # 根据置信度调整参数
    if confidence < 0.7:
        config['position_size'] = config['position_size'] * 0.8
        config['confidence_threshold'] = min(0.75, config['confidence_threshold'] + 0.1)
        config['stop_loss_pct'] = config['stop_loss_pct'] * 0.8
    
    # 混合市场时，平衡两种策略的权重
    if market_state == 'MIXED':
        trend_config = STRATEGY_TYPES['trend']
        mean_config = STRATEGY_TYPES['mean_reversion']
        config['ma_weight'] = (trend_config['ma_weight'] + mean_config['ma_weight']) / 2
        config['rsi_weight'] = (trend_config['rsi_weight'] + mean_config['rsi_weight']) / 2
        config['macd_weight'] = (trend_config['macd_weight'] + mean_config['macd_weight']) / 2
        config['bb_weight'] = (trend_config['bb_weight'] + mean_config['bb_weight']) / 2
    
    # 更新策略名称
    strategy_names = {'trend': '趋势跟踪', 'mean_reversion': '均值回归'}
    risk_names = {'aggressive': '激进进取', 'balanced': '均衡稳健', 'conservative': '稳健保守'}
    config['name'] = f"AI自适应({strategy_names.get(suggested_strategy, suggested_strategy)})"
    config['full_name'] = f"AI自适应 → {strategy_names.get(suggested_strategy, suggested_strategy)} + {risk_names.get(base_risk_preference, base_risk_preference)}"
    config['description'] = f"AI判断当前为{market_state}市场，自动采用{strategy_names.get(suggested_strategy, suggested_strategy)}策略"
    
    return config, market_analysis


# ==================== AI审计友好的策略输出生成器 ====================

class AIAuditStrategyOutputGenerator:
    """
    AI审计友好的策略输出生成器
    
    生成标准化的JSON格式输出，便于AI审计系统解析和评估
    
    输出包含：
    - 策略基本信息
    - 信号类型和强度
    - 市场状态分析
    - 风险评估
    - 决策逻辑解释
    - 交易建议
    """
    
    def __init__(self, strategy_config: Dict, market_analysis: Dict):
        """
        初始化输出生成器
        
        参数:
            strategy_config: 策略配置字典
            market_analysis: 市场分析结果字典
        """
        self.strategy_config = strategy_config
        self.market_analysis = market_analysis
        self.signal_evaluator = SignalStrengthEvaluator(
            market_analysis.get('market_state', 'MIXED')
        )
    
    def generate_signal(self, df_window, current_price: float = None,
                        daily_volume: float = None, is_st: bool = False,
                        board_type: str = "主板") -> Dict[str, Any]:
        """
        生成AI审计友好的完整交易信号
        
        参数:
            df_window: 数据窗口
            current_price: 当前价格（可选，默认从df_window获取）
            daily_volume: 日成交额（可选）
            is_st: 是否ST股票
            board_type: 板块分类
        
        返回:
            AI审计友好的完整信号字典
        """
        if current_price is None:
            current_price = df_window['close'].iloc[-1]
        
        # 获取技术指标
        indicators = self._extract_indicators(df_window, current_price)
        
        # 评估信号强度
        signal_strength_result = self.signal_evaluator.evaluate(indicators)
        
        # 确定信号类型
        signal_type, strategy_mode, signal_reason = self._determine_signal_type(
            indicators, signal_strength_result
        )
        
        # 风险控制
        risk_controller = DynamicRiskController(
            self.strategy_config.get('risk_preference', 'balanced'),
            board_type
        )
        
        # 计算止损
        stop_loss_result = risk_controller.calculate_stop_loss(
            current_price,
            indicators.get('atr', current_price * 0.02),
            self.market_analysis.get('market_state', 'MIXED'),
            signal_type,
            indicators.get('bb_upper'),
            indicators.get('bb_lower')
        )
        
        # 计算仓位
        position_result = risk_controller.calculate_position_size(
            signal_strength_result['总分'],
            daily_volume or 5e8,  # 默认5亿成交额
            indicators.get('atr', current_price * 0.02) / current_price,
            0.05  # 默认预测回撤5%
        )
        
        # 风险评估
        risk_assessment = risk_controller.assess_risk(
            indicators.get('atr', current_price * 0.02) / current_price,
            daily_volume or 5e8,
            is_st,
            board_type
        )
        
        # 计算止盈价格
        take_profit_pct = self.strategy_config.get('take_profit_pct', 0.12)
        if signal_type == "LongOpen":
            take_profit_price = round(current_price * (1 + take_profit_pct), 2)
        elif signal_type == "ShortOpen":
            take_profit_price = round(current_price * (1 - take_profit_pct), 2)
        else:
            take_profit_price = None
        
        # 生成决策逻辑解释
        explanation = self._generate_explanation(
            indicators, signal_strength_result, signal_type, strategy_mode
        )
        
        # 生成交易建议
        trading_suggestion = self._generate_trading_suggestion(
            signal_type, signal_strength_result, risk_assessment,
            stop_loss_result, take_profit_price, position_result
        )
        
        # 构建最终输出
        output = {
            "StrategyName": self.strategy_config.get('name', 'AI自适应策略'),
            "SignalType": signal_type,
            "MarketState": self.market_analysis.get('market_state', 'UNKNOWN'),
            "StrategyMode": strategy_mode,
            "Parameters": {
                "MA周期": self.strategy_config.get('ma_periods', [5, 21, 60]),
                "布林带": self.strategy_config.get('boll_params', {"周期": 20, "标准差": 2.0}),
                "RSI阈值": self.strategy_config.get('rsi_thresholds', [30, 70]),
                "ATR止损倍数": self.strategy_config.get('atr_stop_multiplier', 2.0)
            },
            "SignalStrength": {
                "总分": signal_strength_result['总分'],
                "强度等级": signal_strength_result['强度等级'],
                "分项评分": signal_strength_result['分项评分'],
                "权重分配": signal_strength_result['权重分配']
            },
            "RiskAssessment": {
                **risk_assessment,
                "止损价格": stop_loss_result['止损价格'],
                "止损类型": stop_loss_result['止损类型'],
                "止损比例": f"{stop_loss_result['止损比例']*100:.1f}%",
                "建议仓位比例": f"{position_result['建议仓位比例']*100:.0f}%",
                "止盈价格": take_profit_price
            },
            "Explanation": explanation,
            "TradingSuggestion": trading_suggestion,
            "Timestamp": pd.Timestamp.now().isoformat(),
            "ConfidenceLevel": self.market_analysis.get('confidence', 0.6)
        }
        
        # 添加AI分析结果（如果有）
        if 'ai_reasoning' in self.market_analysis:
            output["AIAnalysis"] = {
                "推理说明": self.market_analysis.get('ai_reasoning', ''),
                "多维度摘要": self.market_analysis.get('ai_multi_dimension_summary', {}),
                "风险提示": self.market_analysis.get('ai_risk_warning', ''),
                "参数建议": self.market_analysis.get('ai_parameter_suggestion', '')
            }
        
        return output
    
    def _extract_indicators(self, df_window, current_price: float) -> Dict[str, Any]:
        """从数据窗口提取技术指标"""
        indicators = {
            'close': current_price,
        }
        
        # 均线
        close = df_window['close']
        ma_periods = self.strategy_config.get('ma_periods', [5, 21, 60])
        
        if len(close) >= ma_periods[0]:
            indicators['ma_short'] = close.rolling(ma_periods[0]).mean().values
        if len(close) >= ma_periods[1]:
            indicators['ma_mid'] = close.rolling(ma_periods[1]).mean().values
        if len(close) >= ma_periods[2]:
            indicators['ma_long'] = close.rolling(ma_periods[2]).mean().values
        
        # RSI
        if 'RSI' in df_window:
            indicators['rsi'] = df_window['RSI'].iloc[-1]
            indicators['rsi_thresholds'] = self.strategy_config.get('rsi_thresholds', [30, 70])
        
        # MACD
        if 'MACD_Hist' in df_window:
            indicators['macd_hist'] = df_window['MACD_Hist'].values
        
        # 布林带
        if all(col in df_window for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            indicators['bb_upper'] = df_window['BB_Upper'].iloc[-1]
            indicators['bb_lower'] = df_window['BB_Lower'].iloc[-1]
            indicators['bb_middle'] = df_window['BB_Middle'].iloc[-1]
        
        # ATR
        if 'ATR' in df_window:
            indicators['atr'] = df_window['ATR'].iloc[-1]
        
        return indicators
    
    def _determine_signal_type(self, indicators: Dict, 
                                signal_strength: Dict) -> Tuple[str, str, str]:
        """确定信号类型、策略模式和原因"""
        total_score = signal_strength['总分']
        strength_level = signal_strength['强度等级']
        market_state = self.market_analysis.get('market_state', 'MIXED')
        
        # 信号强度过低，返回中性
        if strength_level == "无效":
            return "Neutral", "观望", "信号强度不足，建议观望"
        
        # 根据市场状态和信号强度确定策略模式
        if market_state in ['TRENDING', 'VOLATILE']:
            strategy_mode = "TREND跟随"
        else:
            strategy_mode = "MEAN回归"
        
        # 根据综合评分确定信号类型
        if total_score > 0.4:
            signal_type = "LongOpen"
            reason = f"综合评分{total_score:.2f}>0.4，{market_state}市场下建议做多"
        elif total_score < -0.4:
            signal_type = "ShortOpen"
            reason = f"综合评分{total_score:.2f}<-0.4，{market_state}市场下建议减仓/做空"
        else:
            signal_type = "Neutral"
            reason = f"综合评分{total_score:.2f}处于中性区间，建议观望"
        
        # RSI过滤
        rsi = indicators.get('rsi', 50)
        rsi_thresholds = indicators.get('rsi_thresholds', [30, 70])
        
        if signal_type == "LongOpen" and rsi >= rsi_thresholds[1]:
            signal_type = "Neutral"
            reason = f"RSI={rsi:.0f}处于超买区域，拒绝逆势做多"
        elif signal_type == "ShortOpen" and rsi <= rsi_thresholds[0]:
            signal_type = "Neutral"
            reason = f"RSI={rsi:.0f}处于超卖区域，拒绝逆势做空"
        
        return signal_type, strategy_mode, reason
    
    def _generate_explanation(self, indicators: Dict, signal_strength: Dict,
                               signal_type: str, strategy_mode: str) -> Dict[str, Any]:
        """生成决策逻辑解释"""
        explanation = {
            "SignalLogic": "",
            "MarketState": {},
            "ParameterReason": ""
        }
        
        # 信号逻辑说明
        logic_parts = []
        for indicator, exp in signal_strength.get('分项说明', {}).items():
            logic_parts.append(f"{indicator}: {exp}")
        
        explanation["SignalLogic"] = "；".join(logic_parts) if logic_parts else "综合技术指标分析"
        
        # 市场状态说明
        explanation["MarketState"] = self.market_analysis.get('multi_dimension_analysis', {
            "技术面": f"市场处于{self.market_analysis.get('market_state', 'UNKNOWN')}状态",
            "资金面": "暂无数据",
            "政策面": "暂无数据"
        })
        
        # 参数适配原因
        market_state = self.market_analysis.get('market_state', 'MIXED')
        explanation["ParameterReason"] = f"根据{market_state}市场状态，采用{strategy_mode}模式，" \
                                         f"MA周期设置为{self.strategy_config.get('ma_periods', [5,21,60])}，" \
                                         f"信号权重侧重于{'趋势指标' if market_state in ['TRENDING', 'VOLATILE'] else '震荡指标'}"
        
        return explanation
    
    def _generate_trading_suggestion(self, signal_type: str, signal_strength: Dict,
                                      risk_assessment: Dict, stop_loss: Dict,
                                      take_profit: float, position: Dict) -> Dict[str, Any]:
        """生成交易建议"""
        suggestion = {
            "建议方向": "",
            "建议仓位": position['建议仓位比例'],
            "止损价": stop_loss['止损价格'],
            "止盈价": take_profit,
            "建议原因": ""
        }
        
        if signal_type == "LongOpen":
            suggestion["建议方向"] = "买入"
        elif signal_type == "ShortOpen":
            suggestion["建议方向"] = "卖出/减仓"
        else:
            suggestion["建议方向"] = "观望"
        
        # 建议原因
        strength_level = signal_strength['强度等级']
        total_score = signal_strength['总分']
        max_drawdown = risk_assessment.get('最大回撤预测', '5%')
        
        if signal_type == "Neutral":
            suggestion["建议原因"] = f"信号强度{strength_level}（评分{total_score:.2f}），建议观望等待更明确的信号"
        else:
            suggestion["建议原因"] = f"综合技术指标显示{suggestion['建议方向']}信号，" \
                                    f"信号强度{strength_level}（评分{total_score:.2f}），" \
                                    f"风险评估显示最大回撤预测为{max_drawdown}，" \
                                    f"流动性评分{risk_assessment.get('流动性评分', 0.5):.2f}，" \
                                    f"建议仓位{position['建议仓位比例']*100:.0f}%"
        
        return suggestion


# ==================== 便捷函数 ====================

def generate_ai_audit_signal(df_window, risk_preference='balanced', use_ai=False, llm=None,
                              north_flow=None, news_sentiment=None,
                              daily_volume=None, is_st=False, board_type="主板") -> Dict[str, Any]:
    """
    生成AI审计友好的交易信号（便捷函数）
    
    参数:
        df_window: 数据窗口（DataFrame，包含技术指标）
        risk_preference: 风险偏好 ("aggressive" | "balanced" | "conservative")
        use_ai: 是否使用AI进行深度分析
        llm: LLM实例（当use_ai=True时需要提供）
        north_flow: 北向资金净流入（可选，单位：亿元）
        news_sentiment: 新闻情绪分位（可选，0-1之间）
        daily_volume: 日成交额（可选，单位：元）
        is_st: 是否ST股票
        board_type: 板块分类
    
    返回:
        AI审计友好的完整信号字典
    
    示例输出:
    {
        "StrategyName": "AI自适应(趋势跟踪)",
        "SignalType": "LongOpen",
        "MarketState": "TRENDING",
        "StrategyMode": "TREND跟随",
        "Parameters": {...},
        "SignalStrength": {...},
        "RiskAssessment": {...},
        "Explanation": {...},
        "TradingSuggestion": {...}
    }
    """
    # 获取自适应策略配置
    strategy_config, market_analysis = get_adaptive_strategy_config(
        df_window, risk_preference, use_ai, llm, north_flow, news_sentiment
    )
    
    # 创建输出生成器
    generator = AIAuditStrategyOutputGenerator(strategy_config, market_analysis)
    
    # 生成信号
    signal = generator.generate_signal(
        df_window,
        daily_volume=daily_volume,
        is_st=is_st,
        board_type=board_type
    )
    
    return signal


def get_signal_strength_evaluator(market_state: str = "混合市") -> SignalStrengthEvaluator:
    """
    获取信号强度评估器实例
    
    参数:
        market_state: 市场状态 ("趋势市" | "震荡市" | "高波动市" | "混合市")
    
    返回:
        SignalStrengthEvaluator实例
    """
    return SignalStrengthEvaluator(market_state)


def get_risk_controller(risk_preference: str = "balanced", 
                        board_type: str = "主板") -> DynamicRiskController:
    """
    获取动态风险控制器实例
    
    参数:
        risk_preference: 风险偏好 ("aggressive" | "balanced" | "conservative")
        board_type: 板块分类 ("主板" | "创业板" | "科创板")
    
    返回:
        DynamicRiskController实例
    """
    return DynamicRiskController(risk_preference, board_type)


# 子模块：每日决策与胜率优化
from .daily_decision import ai_daily_decision
from .win_rate_optimizer import high_win_rate_decision, get_optimized_config
