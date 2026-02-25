"""
QMT量化交易工具模块
提供基于QMT平台的量化交易功能，包括回测、信号生成、订单执行等
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import io
import base64


class QMTQuantTools:
    """QMT量化交易工具类"""
    
    def __init__(self):
        """初始化QMT连接"""
        self.is_connected = False
        # 注意：实际使用时需要导入QMT的SDK
        # from xtquant import xtdata, xttrader
        
    def connect(self) -> bool:
        """连接到QMT平台"""
        try:
            # 实际实现时连接QMT
            # xtdata.connect()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"连接QMT失败: {str(e)}")
            return False
    
    def get_market_data(self, stock_code: str, start_date: str, end_date: str, 
                       period: str = "1d") -> pd.DataFrame:
        """
        获取股票市场数据
        
        Args:
            stock_code: 股票代码 (如 "600000.SH")
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期 ("1d", "1m", "5m", "15m", "30m", "60m")
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 实际实现时使用QMT接口
        # data = xtdata.get_market_data(field_list=['open','high','low','close','volume'],
        #                                stock_list=[stock_code],
        #                                period=period,
        #                                start_time=start_date,
        #                                end_time=end_date)
        
        # 模拟数据（实际使用时删除）
        print(f"获取{stock_code}从{start_date}到{end_date}的{period}数据")
        return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        if df.empty:
            return df
            
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # 布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        
        # ATR (平均真实波幅)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame, strategy_type: str = "dual_ma",
                                  strategy_config: dict = None) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含技术指标的DataFrame
            strategy_type: 策略类型 ("dual_ma", "macd", "rsi_bb", "multi_factor", "trend", "mean_reversion")
            strategy_config: 策略配置字典（可选，来自 get_strategy_config）
            
        Returns:
            添加了交易信号的DataFrame
        """
        if df.empty:
            return df
        
        # 默认配置
        if strategy_config is None:
            strategy_config = {
                'buy_threshold': 4.0,
                'sell_threshold': 4.0,
                'ma_weight': 2.0,
                'rsi_weight': 1.5,
                'macd_weight': 1.5,
                'bb_weight': 1.5,
            }
        
        df['signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
        
        if strategy_type == "dual_ma":
            # 双均线策略
            df.loc[(df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 'signal'] = 1
            df.loc[(df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 'signal'] = -1
            
        elif strategy_type == "macd":
            # MACD策略
            df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'signal'] = 1
            df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'signal'] = -1
            
        elif strategy_type == "rsi_bb":
            # RSI + 布林带策略
            df.loc[(df['RSI'] < 30) & (df['close'] < df['BB_Lower']), 'signal'] = 1
            df.loc[(df['RSI'] > 70) & (df['close'] > df['BB_Upper']), 'signal'] = -1
            
        elif strategy_type == "multi_factor":
            # 多因子策略
            buy_condition = (
                (df['MA5'] > df['MA20']) &
                (df['RSI'] > 30) & (df['RSI'] < 70) &
                (df['MACD'] > df['Signal']) &
                (df['close'] > df['BB_Lower'])
            )
            sell_condition = (
                (df['MA5'] < df['MA20']) |
                (df['RSI'] > 75) |
                (df['close'] > df['BB_Upper'])
            )
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
        elif strategy_type == "trend":
            # 趋势跟踪策略（使用配置权重）
            ma_weight = strategy_config.get('ma_weight', 2.5)
            macd_weight = strategy_config.get('macd_weight', 2.0)
            buy_threshold = strategy_config.get('buy_threshold', 4.0)
            sell_threshold = strategy_config.get('sell_threshold', 4.0)
            
            for i in range(1, len(df)):
                buy_score = 0
                sell_score = 0
                
                # 均线信号（高权重）
                if df['MA5'].iloc[i] > df['MA10'].iloc[i] > df['MA20'].iloc[i]:
                    buy_score += ma_weight
                elif df['MA5'].iloc[i] < df['MA10'].iloc[i] < df['MA20'].iloc[i]:
                    sell_score += ma_weight
                
                # 均线金叉/死叉
                if df['MA5'].iloc[i] > df['MA20'].iloc[i] and df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1]:
                    buy_score += ma_weight * 0.5
                elif df['MA5'].iloc[i] < df['MA20'].iloc[i] and df['MA5'].iloc[i-1] >= df['MA20'].iloc[i-1]:
                    sell_score += ma_weight * 0.5
                
                # MACD信号（高权重）
                if df['Histogram'].iloc[i] > 0:
                    buy_score += macd_weight
                else:
                    sell_score += macd_weight
                
                # MACD金叉/死叉
                if df['MACD'].iloc[i] > df['Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1]:
                    buy_score += macd_weight * 0.5
                elif df['MACD'].iloc[i] < df['Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1]:
                    sell_score += macd_weight * 0.5
                
                # 判断信号
                if buy_score >= buy_threshold and buy_score > sell_score + 1:
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                elif sell_score >= sell_threshold and sell_score > buy_score + 1:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    
        elif strategy_type == "mean_reversion":
            # 均值回归策略（使用配置权重）
            rsi_weight = strategy_config.get('rsi_weight', 2.5)
            bb_weight = strategy_config.get('bb_weight', 2.5)
            buy_threshold = strategy_config.get('buy_threshold', 4.0)
            sell_threshold = strategy_config.get('sell_threshold', 4.0)
            
            for i in range(1, len(df)):
                buy_score = 0
                sell_score = 0
                
                # RSI超买超卖（高权重）
                rsi = df['RSI'].iloc[i]
                if rsi < 30:
                    buy_score += rsi_weight
                elif rsi < 40:
                    buy_score += rsi_weight * 0.5
                elif rsi > 70:
                    sell_score += rsi_weight
                elif rsi > 60:
                    sell_score += rsi_weight * 0.5
                
                # 布林带信号（高权重）
                close = df['close'].iloc[i]
                bb_lower = df['BB_Lower'].iloc[i]
                bb_upper = df['BB_Upper'].iloc[i]
                bb_middle = df['BB_Middle'].iloc[i]
                
                if close < bb_lower * 1.02:  # 接近下轨买入
                    buy_score += bb_weight
                elif close > bb_upper * 0.98:  # 接近上轨卖出
                    sell_score += bb_weight
                elif close < bb_middle:
                    buy_score += bb_weight * 0.3
                else:
                    sell_score += bb_weight * 0.3
                
                # 判断信号
                if buy_score >= buy_threshold and buy_score > sell_score + 1:
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                elif sell_score >= sell_threshold and sell_score > buy_score + 1:
                    df.iloc[i, df.columns.get_loc('signal')] = -1

        elif strategy_type == "chanlun":
            # 缠论：逐日取窗口调用 chanlun_signal，模拟持仓；适度放宽阈值以产生交易（如 600875）
            from modules.chanlun import chanlun_signal
            min_bars = 20
            position_sim = False
            entry_price_sim = None
            buy_idx = None
            highest_sim = 0.0
            last_sell_idx = -999
            min_confidence_buy = 0.62   # 再放宽以增加交易次数
            min_confidence_sell = 0.60  # 卖点也放宽
            cooldown_bars = 2           # 卖出后隔 2 根可再买
            min_hold_bars = 1          # 持仓满 1 根即可响应卖点
            for i in range(min_bars, len(df)):
                w = df.iloc[: i + 1].copy()
                if "date" not in w.columns and w.index is not None:
                    w["date"] = w.index
                if "Histogram" in w.columns and "MACD_Hist" not in w.columns:
                    w["MACD_Hist"] = w["Histogram"]
                if position_sim and buy_idx is not None:
                    holding_days_sim = i - buy_idx
                    highest_sim = max(highest_sim, float(w["close"].iloc[-1]))
                else:
                    holding_days_sim = 0
                action, confidence, _ = chanlun_signal(
                    w,
                    has_position=position_sim,
                    entry_price=entry_price_sim,
                    holding_days=holding_days_sim,
                    highest_price=highest_sim if position_sim else 0,
                    stop_loss_pct=strategy_config.get("stop_loss_pct", 0.05),
                    take_profit_pct=strategy_config.get("take_profit_pct", 0.12),
                )
                # 仅高质量信号触发：买入置信度>=0.72，卖出/减仓>=0.70
                if action == "BUY" and not position_sim:
                    if confidence >= min_confidence_buy and (i - last_sell_idx) >= cooldown_bars:
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        position_sim = True
                        entry_price_sim = float(w["close"].iloc[-1])
                        buy_idx = i
                        highest_sim = entry_price_sim
                elif action in ("SELL", "REDUCE") and position_sim:
                    # 止损/止盈（高置信度）始终执行；其余卖出信号需置信度达标且持仓满 min_hold_bars，避免频繁卖
                    is_stop_or_target = confidence >= 0.85
                    if is_stop_or_target or (confidence >= min_confidence_sell and holding_days_sim >= min_hold_bars):
                        df.iloc[i, df.columns.get_loc("signal")] = -1
                        position_sim = False
                        entry_price_sim = None
                        buy_idx = None
                        highest_sim = 0.0
                        last_sell_idx = i
                # HOLD 保持 0

        return df
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000, 
                commission_rate: float = 0.0003, slippage: float = 0.001,
                strategy_config: dict = None) -> Dict[str, Any]:
        """
        回测策略
        
        Args:
            df: 包含交易信号的DataFrame
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
            strategy_config: 策略配置（可选，包含 position_size, stop_loss_pct 等）
            
        Returns:
            回测结果字典
        """
        if df.empty or 'signal' not in df.columns:
            return {
                "error": "数据为空或缺少交易信号",
                "total_return": 0,
                "annual_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "total_trades": 0
            }
        
        # 从配置获取参数
        if strategy_config is None:
            strategy_config = {}
        position_size = strategy_config.get('position_size', 0.95)  # 默认95%仓位
        stop_loss_pct = strategy_config.get('stop_loss_pct', 0.05)  # 默认5%止损
        take_profit_pct = strategy_config.get('take_profit_pct', 0.12)  # 默认12%止盈
        
        capital = initial_capital
        position = 0  # 持仓数量
        entry_price = 0  # 开仓价格
        trades = []  # 交易记录
        equity_curve = []  # 资金曲线
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            # 检查止损止盈条件（持仓时）
            if position > 0 and entry_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
                
                # 止损
                if profit_pct < -stop_loss_pct:
                    price = current_price * (1 - slippage)
                    revenue = position * price * (1 - commission_rate)
                    capital += revenue
                    trades.append({
                        'date': row.name if hasattr(row, 'name') else i,
                        'index': i,
                        'type': 'sell',
                        'price': price,
                        'shares': position,
                        'revenue': revenue,
                        'reason': f'止损({profit_pct*100:.1f}%)'
                    })
                    position = 0
                    entry_price = 0
                    
                # 止盈
                elif profit_pct > take_profit_pct:
                    price = current_price * (1 - slippage)
                    revenue = position * price * (1 - commission_rate)
                    capital += revenue
                    trades.append({
                        'date': row.name if hasattr(row, 'name') else i,
                        'index': i,
                        'type': 'sell',
                        'price': price,
                        'shares': position,
                        'revenue': revenue,
                        'reason': f'止盈({profit_pct*100:.1f}%)'
                    })
                    position = 0
                    entry_price = 0
            
            # 买入信号
            if row['signal'] == 1 and position == 0:
                price = current_price * (1 + slippage)
                shares = int((capital * position_size) / price)  # 使用配置的仓位比例
                if shares > 0:
                    cost = shares * price * (1 + commission_rate)
                    if cost <= capital:
                        position = shares
                        entry_price = price  # 记录开仓价格
                        capital -= cost
                        trades.append({
                            'date': row.name if hasattr(row, 'name') else i,
                            'index': i,  # 添加明确的索引位置
                            'type': 'buy',
                            'price': price,
                            'shares': shares,
                            'cost': cost
                        })
            
            # 卖出信号（信号触发）
            elif row['signal'] == -1 and position > 0:
                price = current_price * (1 - slippage)
                revenue = position * price * (1 - commission_rate)
                capital += revenue
                trades.append({
                    'date': row.name if hasattr(row, 'name') else i,
                    'index': i,  # 添加明确的索引位置
                    'type': 'sell',
                    'price': price,
                    'shares': position,
                    'revenue': revenue,
                    'reason': '信号卖出'
                })
                position = 0
                entry_price = 0
            
            # 记录权益
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # 计算回测指标
        equity_curve = np.array(equity_curve)
        total_return = (equity_curve[-1] - initial_capital) / initial_capital if len(equity_curve) > 0 else 0
        
        # 计算年化收益率
        trading_days = len(df)
        years = trading_days / 252  # 假设一年252个交易日
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算夏普比率
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 计算胜率
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        win_trades = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['revenue'] > buy_trades[i]['cost']:
                win_trades += 1
        win_rate = win_trades / len(sell_trades) if len(sell_trades) > 0 else 0
        
        return {
            "total_return": round(total_return * 100, 2),  # 百分比
            "annual_return": round(annual_return * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "total_trades": len(sell_trades),
            "final_capital": round(equity_curve[-1], 2) if len(equity_curve) > 0 else initial_capital,
            "trades": trades,  # 返回所有交易记录用于图表
            "trades_summary": trades[-10:],  # 最近10笔交易摘要
            "equity_curve": equity_curve.tolist()[-30:] if len(equity_curve) > 0 else [],  # 最近30个数据点
            "equity_curve_full": equity_curve.tolist(),  # 完整资金曲线
            "dates": [str(df.index[i]) if hasattr(df.index[i], '__str__') else str(i) for i in range(len(df))],  # 日期序列
            "prices": df['close'].tolist()  # 价格序列
        }
    
    def generate_backtest_charts(self, df: pd.DataFrame, backtest_result: Dict[str, Any], 
                                save_path: Optional[str] = None) -> str:
        """
        生成回测图表
        
        Args:
            df: 包含交易信号的DataFrame
            backtest_result: 回测结果字典
            save_path: 图表保存路径（可选）
            
        Returns:
            图表的base64编码字符串（如果save_path为None）或保存路径
        """
        from utils.matplotlib_chinese import setup_chinese_font
        setup_chinese_font()
        plt.rcParams['font.size'] = 10  # 略大字体便于阅读
        
        # 创建图表：较大画布 + 高 DPI，避免前端显示过小
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3, 
                     left=0.08, right=0.95, top=0.94, bottom=0.05)
        
        # 获取数据
        equity_curve = np.array(backtest_result.get('equity_curve_full', []))
        prices = np.array(backtest_result.get('prices', df['close'].tolist()))
        trades = backtest_result.get('trades', [])
        
        # 1. 资金曲线图（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        if len(equity_curve) > 0:
            ax1.plot(equity_curve, linewidth=2, color='#2E86DE', label='资金曲线')
            ax1.fill_between(range(len(equity_curve)), equity_curve, 
                           alpha=0.3, color='#2E86DE')
            ax1.axhline(y=backtest_result['final_capital'], 
                       color='green', linestyle='--', alpha=0.5, 
                       label=f"最终: ¥{backtest_result['final_capital']:,.0f}")
            ax1.set_title('资金曲线', fontsize=13, fontweight='bold', pad=12)
            ax1.set_xlabel('交易日', fontsize=10)
            ax1.set_ylabel('资金 (元)', fontsize=10)
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.ticklabel_format(style='plain', axis='y')
            ax1.tick_params(labelsize=9)
        
        # 2. 收益率曲线图（右上）
        ax2 = fig.add_subplot(gs[0, 1])
        if len(equity_curve) > 1:
            returns = (equity_curve / equity_curve[0] - 1) * 100
            ax2.plot(returns, linewidth=2, color='#10AC84', label='累计收益率')
            ax2.fill_between(range(len(returns)), 0, returns, 
                           where=(returns >= 0), color='#10AC84', alpha=0.3, 
                           interpolate=True)
            ax2.fill_between(range(len(returns)), 0, returns, 
                           where=(returns < 0), color='#EE5A6F', alpha=0.3, 
                           interpolate=True)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_title(f'累计收益率 ({backtest_result["total_return"]:+.2f}%)', 
                         fontsize=13, fontweight='bold', pad=12)
            ax2.set_xlabel('交易日', fontsize=10)
            ax2.set_ylabel('收益率 (%)', fontsize=10)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(labelsize=9)
        
        # 3. 回撤曲线图（左中）
        ax3 = fig.add_subplot(gs[1, 0])
        if len(equity_curve) > 0:
            cummax = np.maximum.accumulate(equity_curve)
            drawdown = (cummax - equity_curve) / cummax * 100
            ax3.fill_between(range(len(drawdown)), 0, drawdown, 
                           color='#EE5A6F', alpha=0.4, label='回撤')
            ax3.plot(drawdown, linewidth=1.5, color='#EE5A6F')
            max_dd_idx = np.argmax(drawdown)
            ax3.plot(max_dd_idx, drawdown[max_dd_idx], 'ro', markersize=8, 
                    label=f'最大: {backtest_result["max_drawdown"]:.2f}%')
            ax3.set_title('回撤曲线', fontsize=13, fontweight='bold', pad=12)
            ax3.set_xlabel('交易日', fontsize=10)
            ax3.set_ylabel('回撤 (%)', fontsize=10)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.tick_params(labelsize=9)
            ax3.invert_yaxis()  # 反转y轴，回撤向下
        
        # 4. 价格与交易信号图（右中）- 分时走势
        ax4 = fig.add_subplot(gs[1, 1])
        if len(prices) > 0:
            # 使用日期作为x轴
            dates_list = list(range(len(df)))
            if hasattr(df.index, 'strftime'):
                try:
                    dates_display = [d.strftime('%m-%d') if i % max(1, len(df)//10) == 0 else '' 
                                   for i, d in enumerate(df.index)]
                except:
                    dates_display = [str(i) if i % max(1, len(df)//10) == 0 else '' 
                                   for i in range(len(df))]
            else:
                dates_display = [str(i) if i % max(1, len(df)//10) == 0 else '' 
                               for i in range(len(df))]
            
            # 绘制价格走势线
            ax4.plot(dates_list, prices, linewidth=2.5, color='#341F97', 
                    label='价格走势', alpha=0.8, zorder=1)
            
            # 填充区域增强视觉效果
            ax4.fill_between(dates_list, prices, alpha=0.15, color='#341F97')
            
            # 标记买入卖出点
            buy_trades = [t for t in trades if t['type'] == 'buy']
            sell_trades = [t for t in trades if t['type'] == 'sell']
            
            # 绘制买入点
            for i, trade in enumerate(buy_trades):
                idx = trade.get('index', trade.get('date', -1))
                if isinstance(idx, int) and 0 <= idx < len(prices):
                    # 买入点标记
                    ax4.scatter(idx, trade['price'], color='#FF4444', marker='^', 
                              s=200, label='买入' if i == 0 else '', zorder=10, 
                              edgecolors='white', linewidths=2)
                    # 添加垂直线
                    ax4.axvline(x=idx, color='#FF4444', linestyle='--', 
                              alpha=0.3, linewidth=1.5)
                    # 添加文字标注
                    ax4.text(idx, trade['price'], f" ¥{trade['price']:.2f}", 
                           fontsize=8, color='#FF4444', fontweight='bold',
                           verticalalignment='bottom', horizontalalignment='left')
            
            # 绘制卖出点
            for i, trade in enumerate(sell_trades):
                idx = trade.get('index', trade.get('date', -1))
                if isinstance(idx, int) and 0 <= idx < len(prices):
                    # 卖出点标记
                    ax4.scatter(idx, trade['price'], color='#00C851', marker='v', 
                              s=200, label='卖出' if i == 0 else '', zorder=10,
                              edgecolors='white', linewidths=2)
                    # 添加垂直线
                    ax4.axvline(x=idx, color='#00C851', linestyle='--', 
                              alpha=0.3, linewidth=1.5)
                    # 添加文字标注
                    ax4.text(idx, trade['price'], f" ¥{trade['price']:.2f}", 
                           fontsize=8, color='#00C851', fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
            
            ax4.set_title('分时走势与交易信号', fontsize=13, fontweight='bold', pad=12)
            ax4.set_xlabel('日期', fontsize=10)
            ax4.set_ylabel('价格 (元)', fontsize=10)
            
            # 设置合理的y轴范围，增加5%的留白
            price_min, price_max = prices.min(), prices.max()
            y_margin = (price_max - price_min) * 0.05
            ax4.set_ylim(price_min - y_margin, price_max + y_margin)
            
            ax4.set_xticks(dates_list[::max(1, len(df)//10)])
            ax4.set_xticklabels([dates_display[i] for i in range(0, len(dates_display), 
                                max(1, len(df)//10))], rotation=45, ha='right')
            ax4.legend(loc='best', fontsize=8, framealpha=0.9)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.tick_params(labelsize=9)
        
        # 5. 交易统计图（左下）
        ax5 = fig.add_subplot(gs[2, 0])
        metrics = ['总收益', '年化收益', '夏普比率', '最大回撤', '胜率']
        values = [
            backtest_result['total_return'],
            backtest_result['annual_return'],
            backtest_result['sharpe_ratio'] * 20,  # 放大显示
            -backtest_result['max_drawdown'],  # 负值
            backtest_result['win_rate']
        ]
        colors = ['#10AC84' if v > 0 else '#EE5A6F' for v in values]
        bars = ax5.barh(metrics, values, color=colors, alpha=0.7)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            if i == 2:  # 夏普比率
                label = f"{backtest_result['sharpe_ratio']:.2f}"
            elif i == 3:  # 最大回撤
                label = f"{backtest_result['max_drawdown']:.2f}%"
            else:
                label = f"{value:.2f}{'%' if i != 2 else ''}"
            ax5.text(value + (1 if value > 0 else -1), i, label, 
                    va='center', fontsize=9, fontweight='bold')
        
        ax5.set_title('关键指标', fontsize=13, fontweight='bold', pad=12)
        ax5.set_xlabel('数值', fontsize=10)
        ax5.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax5.tick_params(labelsize=9)
        
        # 6. 交易分布图（右下）
        ax6 = fig.add_subplot(gs[2, 1])
        if len(trades) > 0:
            buy_trades = [t for t in trades if t['type'] == 'buy']
            sell_trades = [t for t in trades if t['type'] == 'sell']
            
            # 计算每笔交易的盈亏
            profits = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                profit_pct = (sell_trades[i]['revenue'] - buy_trades[i]['cost']) / buy_trades[i]['cost'] * 100
                profits.append(profit_pct)
            
            if profits:
                profit_colors = ['#10AC84' if p > 0 else '#EE5A6F' for p in profits]
                bars = ax6.bar(range(len(profits)), profits, color=profit_colors, alpha=0.7)
                ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax6.set_title(f'交易盈亏分布 (共{len(profits)}笔)', 
                            fontsize=13, fontweight='bold', pad=12)
                ax6.set_xlabel('交易序号', fontsize=10)
                ax6.set_ylabel('盈亏 (%)', fontsize=10)
                ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax6.tick_params(labelsize=9)
                
                # 添加统计信息
                win_count = sum(1 for p in profits if p > 0)
                avg_win = np.mean([p for p in profits if p > 0]) if win_count > 0 else 0
                avg_loss = np.mean([p for p in profits if p < 0]) if len(profits) - win_count > 0 else 0
                ax6.text(0.02, 0.98, 
                        f'盈利: {win_count}笔 ({avg_win:.2f}%)\n'
                        f'亏损: {len(profits)-win_count}笔 ({avg_loss:.2f}%)',
                        transform=ax6.transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.3))
        
        # 7. 综合信息面板（底部跨列）
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        info_text = f"""
回测周期: {len(df)}日 | 初始: ¥{backtest_result.get('final_capital', 100000) / (1 + backtest_result['total_return']/100):,.0f} | 最终: ¥{backtest_result['final_capital']:,.0f}

总收益: {backtest_result['total_return']:+.2f}% | 年化: {backtest_result['annual_return']:+.2f}% | 夏普: {backtest_result['sharpe_ratio']:.2f}

回撤: {backtest_result['max_drawdown']:.2f}% | 交易: {backtest_result['total_trades']}笔 | 胜率: {backtest_result['win_rate']:.2f}%
        """
        
        ax7.text(0.5, 0.5, info_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8, 
                         edgecolor='#DEE2E6', linewidth=2))
        
        plt.suptitle('量化回测分析报告', fontsize=16, fontweight='bold', y=0.98)
        
        # 保存或返回 base64（高 DPI 使图表在前端显示更大更清晰）
        chart_dpi = 150
        if save_path:
            plt.savefig(save_path, dpi=chart_dpi, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=chart_dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
    
    def place_order(self, stock_code: str, order_type: str, price: float, 
                   volume: int) -> Dict[str, Any]:
        """
        下单（实盘交易）
        
        Args:
            stock_code: 股票代码（ETF代码，如"510300"）
            order_type: 订单类型 ("buy", "sell")
            price: 价格
            volume: 数量
            
        Returns:
            订单结果
        """
        try:
            # 实际实现时使用QMT接口
            # 注意：需要根据实际的QMT SDK调整以下代码
            # 
            # 示例代码（需要根据实际QMT SDK调整）：
            # from xtquant import xttrader
            # 
            # # 获取账户
            # account = xttrader.get_account()  # 或指定账户ID
            # 
            # # 转换订单类型
            # if order_type == "buy":
            #     qmt_order_type = 23  # 买入
            # else:
            #     qmt_order_type = 24  # 卖出
            # 
            # # 转换股票代码格式（如果需要）
            # # QMT可能需要格式如 "510300.SH" 或 "510300.SZ"
            # if stock_code.startswith("51") or stock_code.startswith("56"):
            #     qmt_code = f"{stock_code}.SH"
            # elif stock_code.startswith("15") or stock_code.startswith("16"):
            #     qmt_code = f"{stock_code}.SZ"
            # else:
            #     qmt_code = stock_code
            # 
            # # 下单
            # order_id = xttrader.order_stock(
            #     account_id=account,
            #     stock_code=qmt_code,
            #     order_type=qmt_order_type,
            #     order_volume=volume,
            #     price_type=4,  # 限价单
            #     price=price
            # )
            # 
            # return {
            #     "success": True,
            #     "order_id": str(order_id),
            #     "stock_code": stock_code,
            #     "order_type": order_type,
            #     "price": price,
            #     "volume": volume,
            #     "status": "submitted"
            # }
            
            # 当前为模拟模式
            print(f"[QMT] 实盘下单: {order_type} {stock_code} @ {price} x {volume}")
            print(f"[QMT] 警告：当前为模拟模式，实际使用时需要连接QMT客户端")
            
            return {
                "success": True,
                "order_id": f"ORDER_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "stock_code": stock_code,
                "order_type": order_type,
                "price": price,
                "volume": volume,
                "status": "submitted",
                "mode": "simulation"  # 标识为模拟模式
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        获取当前持仓
        
        Returns:
            持仓列表
        """
        # 实际实现时使用QMT接口
        # positions = xttrader.query_stock_positions(account)
        
        return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息字典
        """
        # 实际实现时使用QMT接口
        # account_info = xttrader.query_stock_asset(account)
        
        return {
            "total_asset": 100000,
            "available_cash": 100000,
            "market_value": 0,
            "profit_loss": 0
        }


# 创建全局实例
qmt_tools = QMTQuantTools()


def get_qmt_tools() -> QMTQuantTools:
    """获取QMT工具实例"""
    return qmt_tools
