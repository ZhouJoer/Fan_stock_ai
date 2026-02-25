"""
每日AI决策模块
提供基于技术指标的每日交易决策逻辑
"""
import pandas as pd
from modules.strategy_config import get_strategy_config


def ai_daily_decision(row, has_position, entry_price=None, holding_days=0, config=None, highest_price=0, current_position_ratio=1.0, df_window=None):
    """
    每日 AI 决策逻辑
    
    基于当日技术指标综合判断交易操作
    返回: (action, confidence, reason, target_ratio) 或 (action, confidence, reason)
    
    action类型:
        - BUY: 建仓买入（空仓时）
        - ADD: 加仓（已持仓时继续买入）
        - SELL: 全部卖出
        - REDUCE: 部分减仓
        - HOLD: 持有观望
    
    参数：
        row: 当日数据行（DataFrame行或字典）
        has_position: 是否持仓
        entry_price: 开仓价格
        holding_days: 持仓天数
        config: 策略配置
        highest_price: 持仓期间最高价（用于移动止盈）
        current_position_ratio: 当前仓位比例(0-1)，用于判断加减仓空间
        df_window: 可选，近期K线 DataFrame（缠论策略时传入，需含 MACD_Hist）
    """
    if config is None:
        config = get_strategy_config('trend', 'balanced')
    
    # 缠论策略：使用缠论模块（分型/笔/中枢/背驰）生成信号
    if config.get('strategy_type') == 'chanlun' and df_window is not None and len(df_window) >= 20:
        try:
            from modules.chanlun import chanlun_signal
            action, confidence, reason = chanlun_signal(
                df_window,
                has_position=has_position,
                entry_price=entry_price,
                holding_days=holding_days,
                highest_price=highest_price,
                stop_loss_pct=config.get('stop_loss_pct', 0.05),
                take_profit_pct=config.get('take_profit_pct', 0.12),
                trailing_stop_pct=config.get('trailing_stop_pct', 0.05),
                trailing_activate_pct=config.get('trailing_activate_pct', 0.06),
            )
            target_ratio = 1.0 if action in ['BUY', 'ADD'] else (0.5 if action == 'REDUCE' else 0.0)
            return action, confidence, reason, target_ratio
        except Exception:
            pass  # 缠论模块异常时回退到下方规则打分
    
    # 仓位管理参数
    max_position_ratio = config.get('max_position_ratio', 1.0)  # 最大仓位
    min_position_ratio = config.get('min_position_ratio', 0.0)  # 最小仓位
    add_position_threshold = config.get('add_position_threshold', 0.3)  # 加仓阈值
    reduce_position_ratio = config.get('reduce_position_ratio', 0.5)  # 减仓比例
    
    # 高胜率模式：使用更严格的入场条件
    if config.get('high_win_rate_mode', False):
        from modules.strategy_config.win_rate_optimizer import high_win_rate_decision
        # 转换row为字典
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        result = high_win_rate_decision(
            row_dict, has_position, entry_price, holding_days, highest_price,
            trend_direction=1 if row_dict.get('close', 0) > row_dict.get('MA60', row_dict.get('MA20', 0)) else -1,
            volume_confirmed=True  # 简化处理
        )
        # 处理 None 或异常返回
        if result is None:
            return 'HOLD', 0.5, '高胜率决策异常', 0.0
        # 兼容旧版返回格式
        if len(result) == 3:
            return result[0], result[1], result[2], 1.0 if result[0] in ['BUY', 'ADD'] else (0.5 if result[0] == 'REDUCE' else 0.0)
        return result
    
    ma_weight = config.get('ma_weight', 2.0)
    rsi_weight = config.get('rsi_weight', 1.5)
    macd_weight = config.get('macd_weight', 1.5)
    bb_weight = config.get('bb_weight', 1.5)
    buy_threshold = config.get('buy_threshold', 4.0)
    sell_threshold = config.get('sell_threshold', 4.0)
    confidence_threshold = config.get('confidence_threshold', 0.6)
    stop_loss_pct = config.get('stop_loss_pct', 0.05)
    take_profit_pct = config.get('take_profit_pct', 0.12)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.05)  # 移动止盈回撤比例，默认5%
    max_holding_days = config.get('max_holding_days', 25)
    min_profit_for_timeout = config.get('min_profit_for_timeout', 3)
    is_trend_strategy = config.get('strategy_type') in ('trend', 'chanlun')
    use_trailing_stop = config.get('use_trailing_stop', True)  # 是否使用移动止盈
    
    price = row['close']
    ma5 = row['MA5']
    ma10 = row['MA10']
    ma20 = row['MA20']
    ma60 = row['MA60'] if pd.notna(row['MA60']) else ma20
    rsi = row['RSI']
    macd_hist = row['MACD_Hist']
    bb_upper = row['BB_Upper']
    bb_middle = row['BB_Middle']
    bb_lower = row['BB_Lower']
    
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    
    # 1. 均线系统评分
    if ma5 > ma10 > ma20:
        buy_score += ma_weight
        buy_reasons.append("均线多头排列(MA5>MA10>MA20)")
    elif ma5 < ma10 < ma20:
        sell_score += ma_weight
        sell_reasons.append("均线空头排列(MA5<MA10<MA20)")
    
    if price > ma20:
        buy_score += ma_weight * 0.5
        buy_reasons.append("价格站上MA20")
    else:
        sell_score += ma_weight * 0.5
        sell_reasons.append("价格跌破MA20")
        
    # 2. RSI 评分
    if rsi < 30:
        if is_trend_strategy:
            sell_score += rsi_weight * 0.3
        else:
            buy_score += rsi_weight
            buy_reasons.append(f"RSI超卖({rsi:.1f}<30)")
    elif rsi < 40:
        if not is_trend_strategy:
            buy_score += rsi_weight * 0.5
            buy_reasons.append(f"RSI偏低({rsi:.1f})")
    elif rsi > 70:
        if is_trend_strategy:
            buy_score += rsi_weight * 0.3
        else:
            sell_score += rsi_weight
            sell_reasons.append(f"RSI超买({rsi:.1f}>70)")
    elif rsi > 60:
        if not is_trend_strategy:
            sell_score += rsi_weight * 0.5
            sell_reasons.append(f"RSI偏高({rsi:.1f})")
        
    # 3. MACD 评分
    if macd_hist > 0:
        buy_score += macd_weight
        buy_reasons.append("MACD金叉")
    else:
        sell_score += macd_weight
        sell_reasons.append("MACD死叉")
        
    # 4. 布林带评分
    if price < bb_lower * 1.02:
        if is_trend_strategy:
            sell_score += bb_weight * 0.5
        else:
            buy_score += bb_weight
            buy_reasons.append("价格触及布林下轨")
    elif price > bb_upper * 0.98:
        if is_trend_strategy:
            buy_score += bb_weight * 0.5
        else:
            sell_score += bb_weight
            sell_reasons.append("价格触及布林上轨")
    elif price > bb_middle:
        buy_score += bb_weight * 0.3
    else:
        sell_score += bb_weight * 0.3
    
    if not has_position:
        if buy_score >= buy_threshold and buy_score > sell_score + 1:
            confidence = min(0.9, 0.5 + (buy_score - sell_score) * 0.1)
            if confidence >= confidence_threshold:
                reason = "买入信号: " + ", ".join(buy_reasons[:3]) if buy_reasons else "综合指标看多"
                # 根据信号强度决定初始仓位
                initial_ratio = min(max_position_ratio, 0.5 + (buy_score - buy_threshold) * 0.1)
                return 'BUY', confidence, reason, initial_ratio
        return 'HOLD', 0.5, "观望", 0.0
    
    # ========== 已持仓：先检查止盈止损，再检查加仓 ==========
    profit_pct = (price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    highest_profit_pct = (highest_price - entry_price) / entry_price * 100 if entry_price > 0 and highest_price > 0 else 0
    take_profit_threshold = take_profit_pct * 100
    stop_loss_threshold = stop_loss_pct * 100
    trailing_stop_threshold = trailing_stop_pct * 100  # 移动止盈回撤阈值
    
    # ========== 止损条件（优先级最高）==========
    if profit_pct < -stop_loss_threshold:
        return 'SELL', 0.9, f"止损: 亏损{profit_pct:.1f}%超过阈值({stop_loss_threshold:.0f}%)", 0.0
    
    partial_stop = stop_loss_threshold * 0.6
    if profit_pct < -partial_stop and sell_score > buy_score + 1:
        reason = f"部分止损: 亏损{profit_pct:.1f}%且趋势转弱"
        if sell_reasons:
            reason += f"({sell_reasons[0]})"
        # 亏损未到止损线但趋势转弱，先减仓一半
        if current_position_ratio > 0.3:
            return 'REDUCE', 0.8, reason, reduce_position_ratio
        return 'SELL', 0.8, reason, 0.0
    
    # ========== 移动止盈（Trailing Stop Profit）==========
    trailing_activate_threshold = config.get('trailing_activate_pct', take_profit_pct * 0.5) * 100
    
    if use_trailing_stop and highest_profit_pct > trailing_activate_threshold:
        drawdown_from_high = (highest_price - price) / highest_price * 100 if highest_price > 0 else 0
        
        if drawdown_from_high > trailing_stop_threshold:
            if drawdown_from_high > trailing_stop_threshold * 1.5:
                return 'SELL', 0.88, f"移动止盈: 最高盈利{highest_profit_pct:.1f}%，回撤{drawdown_from_high:.1f}%触发({trailing_stop_threshold:.0f}%)", 0.0
            elif current_position_ratio > 0.5:
                return 'REDUCE', 0.75, f"部分止盈: 最高盈利{highest_profit_pct:.1f}%，回撤{drawdown_from_high:.1f}%减仓", reduce_position_ratio
    
    # ========== 固定止盈 ==========
    if take_profit_threshold < 100:
        if profit_pct > take_profit_threshold * 1.5:
            return 'SELL', 0.9, f"止盈: 盈利{profit_pct:.1f}%达到上限({take_profit_threshold * 1.5:.0f}%)", 0.0
        
        partial_take_profit = take_profit_threshold * 0.8
        if profit_pct > partial_take_profit:
            if rsi > 70 or sell_score >= sell_threshold:
                reason = f"技术止盈: 盈利{profit_pct:.1f}%"
                if rsi > 70:
                    reason += f", RSI超买({rsi:.1f})"
                if rsi > 80 or profit_pct > take_profit_threshold * 1.2:
                    return 'SELL', 0.85, reason, 0.0
                elif current_position_ratio > 0.3:
                    return 'REDUCE', 0.75, f"部分减仓: {reason}", reduce_position_ratio
    
    # ========== 趋势反转 ==========
    if sell_score >= sell_threshold * 1.25 and sell_score > buy_score + 2:
        reason = "趋势反转: " + ", ".join(sell_reasons[:2]) if sell_reasons else "综合指标看空"
        if current_position_ratio > 0.5 and profit_pct > 0:
            return 'REDUCE', 0.7, f"减仓观望: {reason}", reduce_position_ratio
        return 'SELL', 0.75, reason, 0.0
    
    # ========== 持仓时间过长 ==========
    if holding_days > max_holding_days and profit_pct < min_profit_for_timeout:
        if sell_score > buy_score:
            if profit_pct > 0 and current_position_ratio > 0.3:
                return 'REDUCE', 0.6, f"持仓超时减仓: 持有{holding_days}天盈利仅{profit_pct:.1f}%", reduce_position_ratio
            return 'SELL', 0.65, f"持仓超时清仓: 持有{holding_days}天盈利仅{profit_pct:.1f}%", 0.0
    
    # ========== 判断加仓机会 ==========
    if current_position_ratio < max_position_ratio:
        if buy_score >= buy_threshold * 1.2 and buy_score > sell_score + 2:
            if profit_pct > add_position_threshold * 100:
                add_ratio = min(max_position_ratio - current_position_ratio, 0.3)
                if add_ratio >= 0.1:
                    confidence = min(0.85, 0.6 + (buy_score - buy_threshold) * 0.08)
                    reason = f"加仓信号: 盈利{profit_pct:.1f}%且趋势强化"
                    if buy_reasons:
                        reason += f"({buy_reasons[0]})"
                    return 'ADD', confidence, reason, add_ratio
    
    return 'HOLD', 0.5, "持有观望", current_position_ratio
