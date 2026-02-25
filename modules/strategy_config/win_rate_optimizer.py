"""
胜率优化模块 - 提高量化策略胜率的多种方法

核心优化方向：
1. 更严格的入场条件（多指标共振）
2. 趋势过滤器（大势判断）
3. 更合理的止损止盈比例
4. 仓位管理和资金曲线过滤
5. 信号确认机制
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


# ==================== 高胜率策略配置 ====================

HIGH_WIN_RATE_CONFIG = {
    # 更严格的入场条件
    'entry_conditions': {
        # 多指标共振要求
        'min_bullish_indicators': 3,  # 至少3个指标同时看多才买入
        'min_bearish_indicators': 2,  # 至少2个指标看空才卖出
        
        # 趋势过滤
        'require_trend_alignment': True,  # 要求与大趋势一致
        'trend_ma_period': 60,  # 60日均线判断大趋势
        
        # 信号确认
        'require_volume_confirm': True,  # 要求成交量确认
        'volume_ratio_threshold': 1.3,   # 成交量比过去5日均值高30%
        
        # 价格位置过滤
        'avoid_high_price': True,        # 避免追高
        'max_price_vs_ma20': 1.05,       # 价格不超过MA20的5%
    },
    
    # 更保守的止损止盈
    'risk_management': {
        'stop_loss_pct': 0.03,           # 3%止损（更严格）
        'take_profit_pct': 0.06,         # 6%止盈（2:1盈亏比）
        'trailing_stop_pct': 0.025,      # 2.5%移动止盈
        'trailing_activate_pct': 0.04,   # 4%启动移动止盈
    },
    
    # 信号强度阈值
    'signal_thresholds': {
        'min_buy_confidence': 0.70,      # 买入信心至少70%
        'min_sell_confidence': 0.60,     # 卖出信心至少60%
        'min_buy_score_margin': 2.0,     # 买入分数需比卖出分数高2分
    }
}


def calculate_trend_filter(df: pd.DataFrame, period: int = 60) -> pd.Series:
    """
    计算趋势过滤器
    
    只有在大趋势向上时才允许做多
    
    参数:
        df: 股票数据
        period: 趋势判断周期
    
    返回:
        趋势方向 Series (1=上涨, 0=横盘, -1=下跌)
    """
    ma_long = df['close'].rolling(window=period).mean()
    ma_slope = ma_long.diff(5) / ma_long.shift(5) * 100  # 5日斜率
    
    trend = pd.Series(0, index=df.index)
    trend[ma_slope > 0.5] = 1   # 斜率>0.5%为上涨趋势
    trend[ma_slope < -0.5] = -1  # 斜率<-0.5%为下跌趋势
    
    return trend


def calculate_volume_confirmation(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    计算成交量确认信号
    
    放量突破更可靠
    
    参数:
        df: 股票数据
        window: 成交量均值窗口
    
    返回:
        成交量确认 Series (True=放量, False=缩量)
    """
    if 'volume' not in df.columns:
        return pd.Series(True, index=df.index)
    
    avg_volume = df['volume'].rolling(window=window).mean()
    volume_ratio = df['volume'] / avg_volume
    
    return volume_ratio > HIGH_WIN_RATE_CONFIG['entry_conditions']['volume_ratio_threshold']


def count_bullish_indicators(row: dict) -> Tuple[int, List[str]]:
    """
    统计看多指标数量
    
    参数:
        row: 当日数据行
    
    返回:
        (看多指标数量, 看多指标列表)
    """
    bullish_count = 0
    bullish_indicators = []
    
    price = row.get('close', 0)
    ma5 = row.get('MA5', price)
    ma10 = row.get('MA10', price)
    ma20 = row.get('MA20', price)
    ma60 = row.get('MA60', ma20)
    rsi = row.get('RSI', 50)
    macd_hist = row.get('MACD_Hist', 0)
    bb_lower = row.get('BB_Lower', price * 0.95)
    bb_middle = row.get('BB_Middle', price)
    
    # 1. 均线多头排列
    if ma5 > ma10 > ma20:
        bullish_count += 1
        bullish_indicators.append("均线多头")
    
    # 2. 价格站上MA20
    if price > ma20:
        bullish_count += 1
        bullish_indicators.append("站上MA20")
    
    # 3. 价格站上MA60（大趋势向上）
    if price > ma60:
        bullish_count += 1
        bullish_indicators.append("站上MA60")
    
    # 4. MACD金叉且柱状图变大
    if macd_hist > 0:
        bullish_count += 1
        bullish_indicators.append("MACD金叉")
    
    # 5. RSI在合理区间且上升
    if 40 <= rsi <= 65:
        bullish_count += 1
        bullish_indicators.append(f"RSI适中({rsi:.0f})")
    
    # 6. 价格在布林带中轨上方
    if price > bb_middle:
        bullish_count += 1
        bullish_indicators.append("布林带中轨上方")
    
    # 7. MA5上穿MA10（短期金叉）
    if ma5 > ma10 and abs(ma5 - ma10) / ma10 < 0.01:  # 刚刚上穿
        bullish_count += 1
        bullish_indicators.append("短期金叉")
    
    return bullish_count, bullish_indicators


def count_bearish_indicators(row: dict, entry_price: float = None, profit_pct: float = 0) -> Tuple[int, List[str]]:
    """
    统计看空指标数量
    
    参数:
        row: 当日数据行
        entry_price: 开仓价格
        profit_pct: 当前盈亏百分比
    
    返回:
        (看空指标数量, 看空指标列表)
    """
    bearish_count = 0
    bearish_indicators = []
    
    price = row.get('close', 0)
    ma5 = row.get('MA5', price)
    ma10 = row.get('MA10', price)
    ma20 = row.get('MA20', price)
    rsi = row.get('RSI', 50)
    macd_hist = row.get('MACD_Hist', 0)
    bb_upper = row.get('BB_Upper', price * 1.05)
    
    # 1. 均线空头排列
    if ma5 < ma10 < ma20:
        bearish_count += 1
        bearish_indicators.append("均线空头")
    
    # 2. 价格跌破MA20
    if price < ma20:
        bearish_count += 1
        bearish_indicators.append("跌破MA20")
    
    # 3. MACD死叉
    if macd_hist < 0:
        bearish_count += 1
        bearish_indicators.append("MACD死叉")
    
    # 4. RSI超买
    if rsi > 70:
        bearish_count += 1
        bearish_indicators.append(f"RSI超买({rsi:.0f})")
    
    # 5. 价格触及布林带上轨
    if price > bb_upper * 0.98:
        bearish_count += 1
        bearish_indicators.append("触及布林上轨")
    
    # 6. 盈利回吐
    if entry_price and profit_pct > 3 and price < entry_price * 1.02:
        bearish_count += 1
        bearish_indicators.append("盈利回吐")
    
    return bearish_count, bearish_indicators


def high_win_rate_decision(row: dict, has_position: bool, 
                           entry_price: float = None,
                           holding_days: int = 0,
                           highest_price: float = None,
                           trend_direction: int = 0,
                           volume_confirmed: bool = True) -> Tuple[str, float, str]:
    """
    高胜率决策函数
    
    使用更严格的入场条件和风险管理
    
    参数:
        row: 当日数据行
        has_position: 是否持仓
        entry_price: 开仓价格
        holding_days: 持仓天数
        highest_price: 持仓期间最高价
        trend_direction: 趋势方向 (1=上涨, 0=横盘, -1=下跌)
        volume_confirmed: 成交量是否确认
    
    返回:
        (action, confidence, reason)
    """
    config = HIGH_WIN_RATE_CONFIG
    entry_cfg = config['entry_conditions']
    risk_cfg = config['risk_management']
    signal_cfg = config['signal_thresholds']
    
    price = row.get('close', 0)
    ma20 = row.get('MA20', price)
    
    if not has_position:
        # ========== 买入决策（更严格） ==========
        bullish_count, bullish_list = count_bullish_indicators(row)
        bearish_count, _ = count_bearish_indicators(row)
        
        # 条件1: 多指标共振
        if bullish_count < entry_cfg['min_bullish_indicators']:
            return 'HOLD', 0.4, f"看多指标不足({bullish_count}/{entry_cfg['min_bullish_indicators']})"
        
        # 条件2: 与大趋势一致
        if entry_cfg['require_trend_alignment'] and trend_direction < 0:
            return 'HOLD', 0.4, "大趋势向下，不宜做多"
        
        # 条件3: 成交量确认
        if entry_cfg['require_volume_confirm'] and not volume_confirmed:
            return 'HOLD', 0.45, "成交量未放大确认"
        
        # 条件4: 避免追高
        if entry_cfg['avoid_high_price']:
            if price > ma20 * entry_cfg['max_price_vs_ma20']:
                return 'HOLD', 0.45, f"价格偏离MA20过大({(price/ma20-1)*100:.1f}%)"
        
        # 条件5: 买入分数优势
        score_margin = bullish_count - bearish_count
        if score_margin < signal_cfg['min_buy_score_margin']:
            return 'HOLD', 0.45, f"多空分差不足({score_margin:.1f})"
        
        # 通过所有条件，买入
        confidence = min(0.90, 0.6 + bullish_count * 0.05)
        reason = f"高胜率买入: {', '.join(bullish_list[:3])}"
        if trend_direction > 0:
            reason += ", 顺势"
        if volume_confirmed:
            reason += ", 放量"
        
        return 'BUY', confidence, reason
    
    else:
        # ========== 卖出决策（保护盈利） ==========
        profit_pct = (price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        highest_profit = (highest_price - entry_price) / entry_price * 100 if highest_price and entry_price > 0 else profit_pct
        
        stop_loss = risk_cfg['stop_loss_pct'] * 100
        take_profit = risk_cfg['take_profit_pct'] * 100
        trailing_stop = risk_cfg['trailing_stop_pct'] * 100
        trailing_activate = risk_cfg['trailing_activate_pct'] * 100
        
        bearish_count, bearish_list = count_bearish_indicators(row, entry_price, profit_pct)
        
        # 止损（严格）
        if profit_pct < -stop_loss:
            return 'SELL', 0.92, f"止损: 亏损{profit_pct:.1f}%超过{stop_loss:.0f}%"
        
        # 移动止盈（保护盈利）
        if highest_profit > trailing_activate:
            drawdown = (highest_price - price) / highest_price * 100 if highest_price > 0 else 0
            if drawdown > trailing_stop:
                return 'SELL', 0.88, f"移动止盈: 最高{highest_profit:.1f}%回撤{drawdown:.1f}%"
        
        # 固定止盈
        if profit_pct > take_profit:
            return 'SELL', 0.85, f"止盈: 盈利{profit_pct:.1f}%达到目标{take_profit:.0f}%"
        
        # 技术面恶化
        if bearish_count >= entry_cfg['min_bearish_indicators']:
            if profit_pct > 1:  # 有盈利时更积极保护
                return 'SELL', 0.8, f"技术恶化止盈: {', '.join(bearish_list[:2])}"
            elif profit_pct < -stop_loss * 0.5:  # 亏损加速离场
                return 'SELL', 0.82, f"技术恶化止损: {', '.join(bearish_list[:2])}"
        
        # 持仓超时
        if holding_days > 15 and profit_pct < 2:
            if bearish_count >= 1:
                return 'SELL', 0.65, f"持仓{holding_days}天无盈利，技术面转弱"
        
        return 'HOLD', 0.5, "持有观望"


# ==================== 资金曲线过滤器 ====================

class EquityCurveFilter:
    """
    资金曲线过滤器
    
    当资金曲线跌破其均线时，暂停开新仓
    这是一种元策略，用于在策略表现不佳时减少交易
    """
    
    def __init__(self, lookback: int = 10):
        """
        参数:
            lookback: 资金曲线均线周期
        """
        self.lookback = lookback
        self.equity_history: List[float] = []
    
    def update(self, equity: float):
        """更新资金曲线"""
        self.equity_history.append(equity)
        if len(self.equity_history) > self.lookback * 2:
            self.equity_history = self.equity_history[-self.lookback * 2:]
    
    def allow_new_trade(self) -> bool:
        """
        是否允许开新仓
        
        当资金曲线在其均线上方时允许交易
        """
        if len(self.equity_history) < self.lookback:
            return True  # 数据不足，允许交易
        
        current = self.equity_history[-1]
        ma = np.mean(self.equity_history[-self.lookback:])
        
        return current >= ma * 0.98  # 允许2%的误差


# ==================== 信号确认机制 ====================

def confirm_signal_with_delay(signal_history: List[str], required_confirms: int = 2) -> bool:
    """
    信号确认机制
    
    要求连续N天出现同方向信号才确认
    
    参数:
        signal_history: 最近几天的信号历史 ['BUY', 'HOLD', 'BUY']
        required_confirms: 需要确认的天数
    
    返回:
        是否确认信号
    """
    if len(signal_history) < required_confirms:
        return False
    
    recent = signal_history[-required_confirms:]
    
    # 检查是否连续出现相同信号
    if all(s == 'BUY' for s in recent):
        return True
    if all(s == 'SELL' for s in recent):
        return True
    
    return False


# ==================== 提高胜率的建议总结 ====================

WIN_RATE_IMPROVEMENT_TIPS = """
提高胜率的关键方法：

1. **更严格的入场条件**
   - 要求至少3个技术指标同时看多
   - 与大趋势（60日均线方向）保持一致
   - 成交量放大确认突破有效性
   - 避免追高（价格不超过MA20的5%）

2. **更合理的盈亏比**
   - 止损3%，止盈6%（2:1盈亏比）
   - 即使胜率50%，盈亏比2:1也能盈利
   - 移动止盈保护已有盈利

3. **资金曲线过滤**
   - 当策略表现不佳（资金曲线跌破均线）时暂停交易
   - 避免在不利市场环境中连续亏损

4. **信号确认机制**
   - 要求信号连续2天确认才入场
   - 减少假突破和噪音交易

5. **时间过滤**
   - 避免在财报季、节假日前后交易
   - 选择流动性好的交易时段

6. **选股过滤**
   - 只交易流动性好的股票
   - 避免ST、*ST、刚上市的新股
   - 优先选择行业龙头

7. **仓位管理**
   - 单笔亏损不超过总资金的1-2%
   - 连续亏损后减少仓位
   - 盈利时逐步加仓

8. **LLM决策优化**
   - 提供更完整的市场背景信息
   - 强调保守决策，宁可错过不可做错
   - 要求LLM说明风险点
"""


def get_optimized_config(base_strategy: str = 'trend', 
                         risk_preference: str = 'balanced',
                         optimize_for_win_rate: bool = True) -> dict:
    """
    获取优化后的策略配置
    
    参数:
        base_strategy: 基础策略类型
        risk_preference: 风险偏好
        optimize_for_win_rate: 是否针对胜率优化
    
    返回:
        优化后的策略配置字典
    """
    from . import get_strategy_config
    
    config = get_strategy_config(base_strategy, risk_preference)
    
    if optimize_for_win_rate:
        # 应用高胜率优化
        risk_cfg = HIGH_WIN_RATE_CONFIG['risk_management']
        signal_cfg = HIGH_WIN_RATE_CONFIG['signal_thresholds']
        
        config.update({
            # 更严格的止损止盈
            'stop_loss_pct': risk_cfg['stop_loss_pct'],
            'take_profit_pct': risk_cfg['take_profit_pct'],
            'trailing_stop_pct': risk_cfg['trailing_stop_pct'],
            'trailing_activate_pct': risk_cfg['trailing_activate_pct'],
            
            # 更高的信号阈值
            'confidence_threshold': signal_cfg['min_buy_confidence'],
            'buy_threshold': 5.0,   # 更高的买入门槛
            'sell_threshold': 3.5,  # 更低的卖出门槛（更容易保护本金）
            
            # 更短的持仓时间
            'max_holding_days': 15,
            'min_profit_for_timeout': 1,
            
            # 高胜率特有配置
            'high_win_rate_mode': True,
            'min_bullish_indicators': HIGH_WIN_RATE_CONFIG['entry_conditions']['min_bullish_indicators'],
            'require_trend_alignment': True,
            'require_volume_confirm': True,
        })
    
    return config
