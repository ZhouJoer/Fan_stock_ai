"""
回测模块
提供 AI 决策策略回测和图表生成功能
"""
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

from tools.stock_data import get_stock_data
from tools.technical_indicators import add_technical_indicators_to_df
from modules.strategy_config import (
    STRATEGY_TYPES, RISK_PREFERENCES,
    get_strategy_config, analyze_market_state, get_adaptive_strategy_config
)
from modules.strategy_config import ai_daily_decision
from llm import llm


def ai_decision_backtest(stock_code, initial_capital=100000, days=252, 
                         strategy_type='trend', risk_preference='balanced',
                         use_llm_signals=False, llm_sample_rate=10,
                         high_win_rate_mode=False):
    """
    AI 决策策略回测
    
    基于历史数据逐日判断是否买入/卖出，得出一段时间的回测结果
    
    参数：
        stock_code: 股票代码
        initial_capital: 初始资金
        days: 回测天数
        high_win_rate_mode: 是否启用高胜率模式（更严格的入场条件）
        strategy_type: 策略类型
            - 'trend': 趋势跟踪（顺势而为，追涨杀跌）
            - 'mean_reversion': 均值回归（逆势操作，低买高卖）
            - 'adaptive': AI自适应（自动判断市场状态）
        risk_preference: 风险偏好
            - 'aggressive': 激进进取（高仓位、宽止损、追求高收益）
            - 'balanced': 均衡稳健（中等仓位、适度止损）
            - 'conservative': 稳健保守（低仓位、严格止损、保本优先）
        use_llm_signals: 是否使用LLM生成交易信号（默认False使用规则算法）
        llm_sample_rate: LLM模式下采样频率（默认10，即每10天调用一次LLM决策）
    
    返回：
        回测结果字典
        
    注意：
        - use_llm_signals=True时会调用LLM，速度较慢但决策更智能
        - LLM模式下，不同策略类型会使用不同的prompt模板
        - llm_sample_rate用于控制LLM调用频率，减少API调用
    """
    # 判断是否为自适应策略
    is_adaptive = (strategy_type == 'adaptive')
    
    # 策略参数配置
    if not is_adaptive:
        strategy_config = get_strategy_config(strategy_type, risk_preference)
    else:
        strategy_config = get_strategy_config('trend', risk_preference)
        strategy_config['is_adaptive'] = True
    
    strategy_names = {
        'trend': '趋势跟踪',
        'mean_reversion': '均值回归',
        'adaptive': 'AI自适应'
    }
    risk_names = {
        'aggressive': '激进进取',
        'balanced': '均衡稳健', 
        'conservative': '稳健保守'
    }
    
    strategy_desc = f"{strategy_names.get(strategy_type, '趋势跟踪')} + {risk_names.get(risk_preference, '均衡稳健')}"
    
    signal_mode = "LLM大模型" if use_llm_signals else "规则算法"
    print(f"[AI回测] 开始回测 {stock_code}，策略：{strategy_desc}，信号模式：{signal_mode}")
    if is_adaptive:
        print(f"[AI回测] 🤖 AI自适应模式：将自动判断市场状态并动态切换策略")
    if use_llm_signals:
        print(f"[AI回测] 🧠 LLM模式：使用DeepSeek生成交易信号，采样频率：每{llm_sample_rate}天")
        # 导入LLM信号生成函数
        from prompts import llm_generate_signal
        from llm import llm as default_llm
        llm = default_llm
    print(f"[AI回测] 回测天数：{days}，初始资金：{initial_capital}")
    print(f"[AI回测] 策略参数：仓位{strategy_config['position_size']*100:.0f}%，止损{strategy_config['stop_loss_pct']*100:.1f}%，止盈{strategy_config['take_profit_pct']*100:.1f}%")
    
    # 获取真实历史数据：request_days 为「交易日」数，get_stock_data 内部会按 1.35 倍+缓冲换算为日历天
    # 仅加 1.2 倍缓冲，避免 1 年回测请求 700+ 日历天
    extra_days_for_indicators = 80  # 技术指标计算需要的额外交易日
    request_days = int((days + extra_days_for_indicators) * 1.2)
    
    stock_data = get_stock_data(stock_code, request_days, use_cache=True)
    
    if not stock_data or len(stock_data) < 60:
        raise ValueError(f"获取 {stock_code} 数据失败或数据不足")
    
    df = pd.DataFrame(stock_data)
    print(f"[AI回测] 获取到 {len(df)} 条历史数据")
    print(f"[AI回测] 价格范围：¥{df['close'].min():.2f} ~ ¥{df['close'].max():.2f}")
    
    # 计算技术指标
    df = add_technical_indicators_to_df(df)
    
    # 去掉NaN行
    df = df.dropna().reset_index(drop=True)
    
    # 取最后 days 天用于回测
    if len(df) > days:
        df = df.iloc[-days:].reset_index(drop=True)
    
    print(f"[AI回测] 回测数据范围：{df.iloc[0]['date']} ~ {df.iloc[-1]['date']}，共 {len(df)} 个交易日")
    
    # ========== 逐日回测逻辑 ==========
    start_time = time.time()
    print(f"[AI回测] 开始逐日回测，共 {len(df)} 个交易日...")
    
    capital = initial_capital
    position = 0  # 持仓股数
    entry_price = 0  # 开仓价格
    entry_date = None  # 开仓日期
    holding_days = 0  # 持仓天数
    highest_price = 0  # 持仓期间最高价（用于移动止盈）
    
    trades = []  # 交易记录
    equity_curve = []  # 权益曲线
    trade_signals = []  # 交易信号点
    daily_decisions = []  # 每日决策记录
    market_state_history = []  # 市场状态历史（仅自适应策略）
    
    total_days = len(df)
    progress_interval = max(1, total_days // 10)
    adaptive_check_interval = 20  # 自适应策略每20天重新评估
    last_market_state = None
    
    # LLM模式的缓存决策
    llm_cached_action = 'HOLD'
    llm_cached_confidence = 0.5
    llm_cached_reason = "等待LLM决策"
    last_llm_call_idx = -999  # 上次LLM调用的index
    llm_call_count = 0  # LLM调用计数
    
    for idx, row in df.iterrows():
        # 进度显示
        if idx % progress_interval == 0 or idx == total_days - 1:
            progress = (idx + 1) / total_days * 100
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (total_days - idx - 1) if idx > 0 else 0
            bar_len = 20
            filled = int(bar_len * (idx + 1) / total_days)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r[AI回测] 进度: [{bar}] {progress:.0f}% | 已用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s", end='')
        
        price = row['close']
        date = row['date']
        
        # ========== 自适应策略：定期重新评估市场状态 ==========
        if is_adaptive and (idx == 0 or idx % adaptive_check_interval == 0):
            window_start = max(0, idx - 60)
            df_window = df.iloc[window_start:idx+1] if idx > 0 else df.iloc[:min(60, len(df))]
            
            if len(df_window) >= 20:
                market_analysis = analyze_market_state(df_window, use_ai=False)
                suggested_strategy = market_analysis['suggested_strategy']
                market_state = market_analysis['market_state']
                
                if market_state != last_market_state:
                    strategy_config, _ = get_adaptive_strategy_config(df_window, risk_preference, use_ai=False)
                    last_market_state = market_state
                    
                    if idx > 0:
                        print(f"\n[AI自适应] 第{idx}天 市场状态变化: {market_state} → 切换至 {suggested_strategy} 策略")
                    else:
                        print(f"\n[AI自适应] 初始市场状态: {market_state} → 采用 {suggested_strategy} 策略")
                
                market_state_history.append({
                    'day': idx,
                    'date': date,
                    'market_state': market_state,
                    'strategy': suggested_strategy,
                    'confidence': market_analysis['confidence']
                })
        
        has_position = position > 0
        if has_position:
            holding_days += 1
            # 更新持仓期间最高价（用于移动止盈）
            if price > highest_price:
                highest_price = price
        
        # ========== 决策逻辑：LLM模式 vs 规则模式 ==========
        if use_llm_signals:
            # LLM模式：使用大语言模型生成交易信号
            # 为避免过多API调用，采用采样策略：
            # 1. 每llm_sample_rate天调用一次LLM
            # 2. 持仓状态变化时立即调用LLM
            # 3. 其他时间使用缓存的决策
            
            should_call_llm = (
                idx - last_llm_call_idx >= llm_sample_rate or  # 达到采样间隔
                (has_position and not (position > 0)) or  # 刚开仓
                (not has_position and entry_price > 0)  # 刚平仓
            )
            
            # 如果持仓盈亏超过阈值，也需要及时调用LLM
            if has_position and entry_price > 0:
                current_pnl = (price - entry_price) / entry_price * 100
                if abs(current_pnl) > strategy_config['stop_loss_pct'] * 100 * 0.7:
                    should_call_llm = True
            
            if should_call_llm:
                # 确定当前使用的策略类型（自适应模式下可能动态变化）
                current_strategy = strategy_type
                if is_adaptive and market_state_history:
                    last_state = market_state_history[-1]
                    current_strategy = last_state.get('strategy', 'trend')
                
                action, confidence, reason = llm_generate_signal(
                    row,
                    strategy_type=current_strategy,
                    risk_preference=risk_preference,
                    has_position=has_position,
                    entry_price=entry_price if has_position else None,
                    holding_days=holding_days if has_position else 0,
                    highest_price=highest_price if has_position else None,
                    llm=llm
                )
                
                # 更新缓存
                llm_cached_action = action
                llm_cached_confidence = confidence
                llm_cached_reason = reason
                last_llm_call_idx = idx
                llm_call_count += 1
                
                # 标记这是LLM决策
                reason = f"[LLM] {reason}"
            else:
                # 使用缓存的LLM决策
                action = llm_cached_action
                confidence = llm_cached_confidence
                reason = f"[LLM缓存] {llm_cached_reason}"
        else:
            # 规则模式：使用传统算法生成交易信号
            current_position_ratio = (position * price) / (capital + position * price) if (capital + position * price) > 0 else 0
            result = ai_daily_decision(row, has_position, entry_price, holding_days, strategy_config, highest_price, current_position_ratio)
            # 兼容新旧返回格式，处理 None 情况
            if result is None:
                action, confidence, reason, target_ratio = 'HOLD', 0.5, '决策异常', 0.0
            elif len(result) == 4:
                action, confidence, reason, target_ratio = result
            else:
                action, confidence, reason = result
                target_ratio = 1.0 if action == 'BUY' else 0.0
        
        if is_adaptive and last_market_state:
            reason = f"[{last_market_state}] {reason}"
        
        daily_decisions.append({
            'date': date,
            'price': price,
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'has_position': has_position,
            'market_state': last_market_state if is_adaptive else None
        })
        
        # 执行交易
        total_value = capital + position * price
        
        # 买入/建仓
        if action == 'BUY' and confidence >= strategy_config['confidence_threshold'] and position == 0:
            available = capital * strategy_config['position_size'] * target_ratio
            shares = int(available / price / 100) * 100  # 整手买入
            if shares > 0:
                cost = shares * price
                capital -= cost
                position = shares
                entry_price = price
                entry_date = date
                holding_days = 0
                highest_price = price  # 重置最高价
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'type': 'BUY',
                    'price': round(price, 2),
                    'shares': shares,
                    'cost': round(cost, 2),
                    'confidence': round(confidence, 2),
                    'market_state': last_market_state if is_adaptive else None,
                    'reason': reason
                })
                trade_signals.append({'date': date, 'type': 'BUY', 'price': price})
                print(f"[AI回测] {date} 买入 {shares} 股 @ ¥{price:.2f}，原因: {reason}")
        
        # 加仓
        elif action == 'ADD' and confidence >= strategy_config['confidence_threshold'] and position > 0:
            # 计算加仓数量
            add_value = total_value * target_ratio
            add_shares = int(add_value / price / 100) * 100
            if add_shares > 0 and capital >= add_shares * price:
                cost = add_shares * price
                capital -= cost
                # 更新平均成本
                total_cost = position * entry_price + cost
                position += add_shares
                entry_price = total_cost / position  # 新的平均成本
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'type': 'ADD',
                    'price': round(price, 2),
                    'shares': add_shares,
                    'cost': round(cost, 2),
                    'confidence': round(confidence, 2),
                    'total_position': position,
                    'avg_cost': round(entry_price, 2),
                    'reason': reason
                })
                trade_signals.append({'date': date, 'type': 'ADD', 'price': price})
                print(f"[AI回测] {date} 加仓 {add_shares} 股 @ ¥{price:.2f}，持仓{position}股，均价¥{entry_price:.2f}，原因: {reason}")
        
        # 减仓
        elif action == 'REDUCE' and confidence >= strategy_config['confidence_threshold'] and position > 0:
            # 计算减仓数量
            reduce_shares = int(position * target_ratio / 100) * 100  # target_ratio是减仓比例
            if reduce_shares > 0:
                revenue = reduce_shares * price
                profit = revenue - (reduce_shares * entry_price)
                profit_pct = (price - entry_price) / entry_price * 100
                capital += revenue
                position -= reduce_shares
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'type': 'REDUCE',
                    'price': round(price, 2),
                    'shares': reduce_shares,
                    'revenue': round(revenue, 2),
                    'profit': round(profit, 2),
                    'profit_pct': round(profit_pct, 2),
                    'remaining_position': position,
                    'confidence': round(confidence, 2),
                    'reason': reason
                })
                trade_signals.append({'date': date, 'type': 'REDUCE', 'price': price})
                print(f"[AI回测] {date} 减仓 {reduce_shares} 股 @ ¥{price:.2f}，剩余{position}股，盈亏 {profit_pct:+.2f}%，原因: {reason}")
                
                # 如果减仓后仓位为0，重置状态
                if position == 0:
                    entry_price = 0
                    entry_date = None
                    holding_days = 0
                    highest_price = 0
        
        # 全部卖出
        elif action == 'SELL' and confidence >= strategy_config['confidence_threshold'] and position > 0:
            revenue = position * price
            profit = revenue - (position * entry_price)
            profit_pct = (price - entry_price) / entry_price * 100
            capital += revenue
            
            trades.append({
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'type': 'SELL',
                'price': round(price, 2),
                'shares': position,
                'revenue': round(revenue, 2),
                'profit': round(profit, 2),
                'profit_pct': round(profit_pct, 2),
                'holding_days': holding_days,
                'confidence': round(confidence, 2),
                'reason': reason
            })
            trade_signals.append({'date': date, 'type': 'SELL', 'price': price})
            print(f"[AI回测] {date} 卖出 {position} 股 @ ¥{price:.2f}，盈亏 {profit_pct:+.2f}%，原因: {reason}")
            
            position = 0
            entry_price = 0
            entry_date = None
            holding_days = 0
            highest_price = 0  # 重置最高价
        
        # 记录当日权益
        total_value = capital + position * price
        equity_curve.append({
            'date': date,
            'equity': total_value,
            'price': price,
            'position_value': position * price,
            'cash': capital
        })
    
    # 回测完成
    end_time = time.time()
    elapsed_total = end_time - start_time
    print(f"\n[AI回测] 回测完成！总耗时: {elapsed_total:.2f}秒，平均每日: {elapsed_total/total_days*1000:.2f}ms")
    if use_llm_signals:
        print(f"[AI回测] LLM调用次数: {llm_call_count}，平均调用间隔: {total_days/llm_call_count:.1f}天" if llm_call_count > 0 else "[AI回测] LLM调用次数: 0")
    
    # ========== 计算回测统计指标 ==========
    equity_df = pd.DataFrame(equity_curve)
    
    final_value = capital + position * df.iloc[-1]['close']
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    
    trading_days = len(equity_df)
    if trading_days > 0 and final_value > 0 and initial_capital > 0:
        annual_return = ((final_value / initial_capital) ** (252.0 / trading_days) - 1) * 100
    else:
        annual_return = 0
    
    print(f"[AI回测] 交易日数: {trading_days}，总收益: {total_return:.2f}%，年化收益: {annual_return:.2f}%")
    
    # 夏普比率
    daily_rf = 0.03 / 252
    excess_returns = equity_df['daily_return'].dropna() - daily_rf
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # 最大回撤
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax'] * 100
    max_drawdown = equity_df['drawdown'].max()
    
    # 胜率（包括全部卖出和减仓）
    exit_trades = [t for t in trades if t['type'] in ['SELL', 'REDUCE']]
    winning_trades = [t for t in exit_trades if t.get('profit', 0) > 0]
    win_rate = len(winning_trades) / len(exit_trades) * 100 if exit_trades else 0
    
    # 统计交易类型
    buy_count = len([t for t in trades if t['type'] == 'BUY'])
    add_count = len([t for t in trades if t['type'] == 'ADD'])
    reduce_count = len([t for t in trades if t['type'] == 'REDUCE'])
    sell_count = len([t for t in trades if t['type'] == 'SELL'])
    
    # 生成图表
    chart_base64 = generate_ai_backtest_chart(df, equity_df, trade_signals, stock_code)
    
    strategy_display = strategy_names.get(strategy_type, 'AI综合决策')
    if is_adaptive:
        strategy_display = 'AI自适应决策'
    
    # 信号引擎显示
    signal_engine = "LLM大语言模型（DeepSeek）" if use_llm_signals else "规则算法（技术指标评分）"
    
    result = {
        'stock_code': stock_code,
        'strategy': strategy_display,
        'strategy_type': strategy_type,
        'signal_engine': signal_engine,
        'use_llm_signals': use_llm_signals,
        'start_date': df.iloc[0]['date'].strftime('%Y-%m-%d'),
        'end_date': df.iloc[-1]['date'].strftime('%Y-%m-%d'),
        'trading_days': trading_days,
        'backtest_time': round(elapsed_total, 2),
        'initial_capital': initial_capital,
        'final_capital': round(final_value, 2),
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'trades': trades[-10:],
        'chart': chart_base64
    }
    
    # LLM模式额外信息
    if use_llm_signals:
        result['llm_info'] = {
            'call_count': llm_call_count,
            'sample_rate': llm_sample_rate,
            'avg_call_interval': round(total_days / llm_call_count, 1) if llm_call_count > 0 else 0
        }
    
    # 自适应策略额外信息
    if is_adaptive and market_state_history:
        state_counts = {}
        for h in market_state_history:
            state = h['market_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        total_checks = len(market_state_history)
        state_pcts = {k: round(v / total_checks * 100, 1) for k, v in state_counts.items()}
        
        result['adaptive_info'] = {
            'market_state_history': market_state_history[-5:],
            'state_distribution': state_pcts,
            'total_state_checks': total_checks,
            'check_interval': adaptive_check_interval
        }
        
        print(f"\n[AI自适应] 市场状态分布: {state_pcts}")
    
    return result




def generate_ai_backtest_chart(df, equity_df, trade_signals, stock_code):
    """
    生成 AI 决策回测图表
    
    参数：
        df: 股票数据 DataFrame
        equity_df: 权益曲线 DataFrame
        trade_signals: 交易信号列表
        stock_code: 股票代码
    
    返回：
        base64 编码的图表图片
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from io import BytesIO
    import base64
    from utils.matplotlib_chinese import setup_chinese_font
    setup_chinese_font()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'AI 决策策略回测 - {stock_code}', fontsize=14, fontweight='bold')
    
    # 图1：价格走势 + 买卖点
    ax1 = axes[0]
    ax1.plot(df['date'], df['close'], label='收盘价', color='#2196F3', linewidth=1.5)
    ax1.plot(df['date'], df['MA5'], label='MA5', color='#FF9800', linewidth=1, alpha=0.7)
    ax1.plot(df['date'], df['MA20'], label='MA20', color='#9C27B0', linewidth=1, alpha=0.7)
    ax1.fill_between(df['date'], df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray', label='布林带')
    
    # 标注买卖点
    for signal in trade_signals:
        if signal['type'] == 'BUY':
            ax1.scatter(signal['date'], signal['price'], marker='^', color='red', s=100, zorder=5)
            ax1.annotate(f"买\n{signal['price']:.2f}", (signal['date'], signal['price']),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')
        else:
            ax1.scatter(signal['date'], signal['price'], marker='v', color='green', s=100, zorder=5)
            ax1.annotate(f"卖\n{signal['price']:.2f}", (signal['date'], signal['price']),
                        textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='green')
    
    ax1.set_ylabel('价格')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('价格走势与买卖信号', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2：资金曲线
    ax2 = axes[1]
    ax2.plot(equity_df['date'], equity_df['equity'], label='账户权益', color='#4CAF50', linewidth=2)
    ax2.axhline(y=equity_df['equity'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax2.fill_between(equity_df['date'], equity_df['equity'].iloc[0], equity_df['equity'], 
                     where=(equity_df['equity'] >= equity_df['equity'].iloc[0]), 
                     color='#4CAF50', alpha=0.3)
    ax2.fill_between(equity_df['date'], equity_df['equity'].iloc[0], equity_df['equity'], 
                     where=(equity_df['equity'] < equity_df['equity'].iloc[0]), 
                     color='#F44336', alpha=0.3)
    ax2.set_ylabel('账户权益')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title('资金曲线', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 图3：回撤
    ax3 = axes[2]
    ax3.fill_between(equity_df['date'], 0, -equity_df['drawdown'], color='#F44336', alpha=0.5)
    ax3.set_ylabel('回撤 (%)')
    ax3.set_xlabel('日期')
    ax3.set_title('回撤曲线', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    
    # 转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    plt.close(fig)
    
    return chart_base64
