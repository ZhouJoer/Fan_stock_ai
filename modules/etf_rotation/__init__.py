"""
ETF轮动策略模块
通过QMT接口实现ETF轮动策略，支持回测

策略逻辑：
1. 选择多个ETF（如沪深300ETF、中证500ETF、创业板ETF等）
2. 根据技术指标（动量、RSI、均线等）计算每个ETF的得分
3. 选择得分最高的ETF持有
4. 定期轮动（如每周或每月）切换到得分更高的ETF
"""
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# 使用非交互式后端，避免在后台线程中创建图形时触发 GUI 警告
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import io
import base64

from tools.stock_data import get_stock_data
from tools.technical_indicators import add_technical_indicators_to_df
from modules.qmt import get_qmt_tools
from tools.risk_control import audit_risk, RiskAuditInput, get_recent_daily_returns_from_equity, DEFAULT_RISK_CONFIG
from typing import Optional, List, Dict, Any, Callable


# 最小单笔交易金额（元），避免因现金不足产生 1 股等无意义交易
MIN_TRADE_VALUE = 1000

# 常用ETF代码列表
DEFAULT_ETF_LIST = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "512100": "化工50ETF",
    "513130": "恒生科技ETF",
    "513120": "港股创新药ETF",
    "563230": "卫星ETF",
    "512880": "证券ETF",
    "512170": "医疗ETF",
    "513500": "纳指ETF",
    "515050": "5G ETF",
    "515790": "光伏ETF",
    "513030": "中概互联ETF",
    "518880": "黄金ETF",
    "512480": "半导体ETF",
}


class ETFRotationStrategy:
    """ETF轮动策略类"""
    
    def __init__(self, etf_codes: List[str], initial_capital: float = 100000,
                 rotation_interval: int = 5, commission_rate: float = 0.0003,
                 slippage: float = 0.001, top_k: int = 1,
                 score_weights: Optional[Dict[str, float]] = None,
                 rebalance_interval: Optional[int] = None,
                 min_score_threshold: float = 20.0,
                 use_ai: bool = False,
                 position_strategy: str = "equal"):
        """
        初始化ETF轮动策略
        
        参数:
            etf_codes: ETF代码列表
            initial_capital: 初始资金
            rotation_interval: 轮动间隔（交易日）
            commission_rate: 手续费率
            slippage: 滑点
            top_k: 持仓数量（持有得分最高的k只ETF，默认1只）
            score_weights: 得分权重配置，格式：{"momentum": 0.65, "rsi": 0.1, "ma": 0.15, "macd": 0.1}
                          默认值：动量65%，RSI10%，均线15%，MACD10%
            rebalance_interval: 再平衡间隔（交易日），None表示无再平衡，只在轮动时调整仓位
            min_score_threshold: 最低得分阈值，如果所有ETF得分都低于此阈值，则清仓不持仓（默认20.0）
            position_strategy: 仓位策略。"equal"=等权重；"kelly"=按凯利公式（期望收益/方差）分配
        """
        self.etf_codes = etf_codes
        self.initial_capital = initial_capital
        self.rotation_interval = rotation_interval
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.top_k = top_k
        self.rebalance_interval = rebalance_interval
        self.min_score_threshold = min_score_threshold
        self.use_ai = use_ai  # 是否使用AI分析
        self.position_strategy = (position_strategy or "equal").strip().lower()
        if self.position_strategy not in ("equal", "kelly"):
            self.position_strategy = "equal"
        self.qmt = get_qmt_tools()
        
        # 设置得分权重，默认值：动量65%，RSI10%，均线15%，MACD10%
        if score_weights is None:
            self.score_weights = {
                "momentum": 0.65,
                "rsi": 0.1,
                "ma": 0.15,
                "macd": 0.1
            }
        else:
            # 归一化权重，确保总和为1
            total_weight = sum(score_weights.values())
            if total_weight > 0:
                self.score_weights = {k: v / total_weight for k, v in score_weights.items()}
            else:
                self.score_weights = {"momentum": 0.65, "rsi": 0.1, "ma": 0.15, "macd": 0.1}
    
    def calculate_etf_score(self, df: pd.DataFrame, lookback_days: int = 20) -> float:
        """
        计算ETF得分
        
        参数:
            df: ETF的DataFrame数据（包含技术指标）
            lookback_days: 回看天数
            
        返回:
            ETF得分（越高越好）
        """
        if df.empty or len(df) < lookback_days:
            return 0.0
        
        # 取最近的数据
        recent_df = df.tail(lookback_days).copy()
        
        # 1. 动量得分（65%权重）
        # 计算最近N天的收益率
        momentum = (recent_df['close'].iloc[-1] / recent_df['close'].iloc[0] - 1) * 100
        momentum_score = max(0, min(100, momentum * 2))  # 归一化到0-100
        
        # 2. RSI得分（10%权重）
        if 'RSI' in recent_df.columns and not recent_df['RSI'].isna().all():
            rsi = recent_df['RSI'].iloc[-1]
            # RSI在50-70之间较好（上升趋势但未超买）
            if 50 <= rsi <= 70:
                rsi_score = 100
            elif 30 <= rsi < 50:
                rsi_score = 50 + (rsi - 30) * 2.5  # 30-50线性映射到50-100
            elif 70 < rsi <= 80:
                rsi_score = 100 - (rsi - 70) * 2  # 70-80线性映射到100-80
            else:
                rsi_score = max(0, 100 - abs(rsi - 65) * 2)
        else:
            rsi_score = 50  # 默认值
        
        # 3. 均线得分（15%权重）
        if 'MA5' in recent_df.columns and 'MA20' in recent_df.columns:
            ma5 = recent_df['MA5'].iloc[-1]
            ma20 = recent_df['MA20'].iloc[-1]
            current_price = recent_df['close'].iloc[-1]
            
            # 均线多头排列得分
            if ma5 > ma20 and current_price > ma5:
                ma_score = 100
            elif ma5 > ma20:
                ma_score = 70
            elif ma5 < ma20 and current_price < ma5:
                ma_score = 0
            else:
                ma_score = 30
        else:
            ma_score = 50  # 默认值
        
        # 4. MACD得分（10%权重）（与 tools.technical_indicators 列名一致：MACD_Signal, MACD_Hist）
        signal_col = 'MACD_Signal' if 'MACD_Signal' in recent_df.columns else 'Signal'
        hist_col = 'MACD_Hist' if 'MACD_Hist' in recent_df.columns else 'Histogram'
        if 'MACD' in recent_df.columns and signal_col in recent_df.columns:
            macd = recent_df['MACD'].iloc[-1]
            signal = recent_df[signal_col].iloc[-1]
            histogram = recent_df[hist_col].iloc[-1] if hist_col in recent_df.columns else (macd - signal)
            
            # MACD金叉且柱状图为正
            if macd > signal and histogram > 0:
                macd_score = 100
            elif macd > signal:
                macd_score = 70
            elif macd < signal and histogram < 0:
                macd_score = 0
            else:
                macd_score = 30
        else:
            macd_score = 50  # 默认值
        
        # 综合得分（加权平均，使用可配置的权重）
        total_score = (
            momentum_score * self.score_weights.get("momentum", 0.4) +
            rsi_score * self.score_weights.get("rsi", 0.2) +
            ma_score * self.score_weights.get("ma", 0.2) +
            macd_score * self.score_weights.get("macd", 0.2)
        )
        
        return total_score

    def _compute_kelly_weights(self, aligned_dict: Dict[str, pd.DataFrame],
                               target_etfs: List[str], actual_data_idx: int,
                               lookback_days: int,
                               max_single_pct: float = 0.45) -> Dict[str, float]:
        """
        按凯利公式计算各 ETF 目标权重：权重 ∝ max(0, 期望收益/方差)。
        使用 lookback 内的日收益率均值和标准差。
        返回 dict etf_code -> weight，已归一化且单标的不超过 max_single_pct。
        """
        if not target_etfs:
            return {}
        kelly_raw = {}
        for etf_code in target_etfs:
            if etf_code not in aligned_dict or actual_data_idx >= len(aligned_dict[etf_code]):
                continue
            df = aligned_dict[etf_code]
            start_idx = max(0, actual_data_idx - lookback_days)
            window = df.iloc[start_idx:actual_data_idx + 1]
            if 'close' not in window.columns or len(window) < 5:
                continue
            ret = window['close'].pct_change().dropna()
            if len(ret) < 3:
                continue
            mu = float(ret.mean())
            sigma = float(ret.std())
            if sigma <= 0 or np.isnan(sigma):
                sigma = 0.02
            # 对负收益做下限：避免熊市窗口导致 Kelly 全为 0、空仓率过高；用小幅正下限参与分配
            mu = max(0.0002, mu)
            sigma_sq = sigma ** 2
            f = mu / sigma_sq
            kelly_raw[etf_code] = max(0.0, min(f, 2.0))
        total = sum(kelly_raw.values())
        if total <= 0:
            return {c: 1.0 / len(target_etfs) for c in target_etfs}
        weights = {c: kelly_raw[c] / total for c in kelly_raw}
        # 单标的上限
        for c in weights:
            weights[c] = min(weights[c], max_single_pct)
        s = sum(weights.values())
        if s > 0:
            weights = {c: w / s for c, w in weights.items()}
        return weights
    
    def load_etf_data(self, days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        加载所有ETF数据
        
        参数:
            days: 需要的数据天数（回测天数 + 回看天数）
            
        返回:
            ETF代码到DataFrame的映射
        """
        data_dict = {}
        extra_days = 80  # 技术指标需要的额外数据
        # request_days 为交易日数，get_stock_data 内部会换算为日历天；仅加 1.2 倍缓冲，避免请求日历日过多
        request_days = int((days + extra_days) * 1.2)
        
        print(f"[ETF轮动] 开始加载 {len(self.etf_codes)} 只ETF数据（需要{days}交易日，请求{request_days}交易日）...")
        
        for i, code in enumerate(self.etf_codes):
            print(f"[ETF轮动] ({i+1}/{len(self.etf_codes)}) 加载 {code}...")
            try:
                stock_data = get_stock_data(code, request_days, use_cache=True)
                if stock_data and len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    df = add_technical_indicators_to_df(df)
                    df = df.dropna().reset_index(drop=True)
                    data_dict[code] = df
                    print(f"[ETF轮动] {code} 加载成功，{len(df)} 条数据")
                    if len(df) < days:
                        print(f"[ETF轮动] 警告：{code} 只有 {len(df)} 条数据，少于请求的 {days} 天")
                else:
                    print(f"[ETF轮动] {code} 数据不足，跳过")
            except Exception as e:
                print(f"[ETF轮动] {code} 加载失败: {e}")
        
        return data_dict
    
    def align_data_dates(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐所有ETF的日期，确保在同一天有数据
        
        参数:
            data_dict: ETF数据字典
            
        返回:
            对齐后的数据字典
        """
        if not data_dict:
            return {}
        
        # 找到所有ETF的共同日期（按ETF代码顺序处理，确保一致性）
        common_dates = None
        # 按ETF代码排序，确保处理顺序一致
        sorted_codes = sorted(data_dict.keys())
        for code in sorted_codes:
            df = data_dict[code]
            if 'date' in df.columns:
                df_dates = set(pd.to_datetime(df['date']).dt.date)
                if common_dates is None:
                    common_dates = df_dates
                else:
                    common_dates = common_dates & df_dates
        
        if not common_dates:
            return {}
        
        # 对齐数据（按ETF代码排序，确保结果一致）
        aligned_dict = {}
        common_dates_list = sorted(list(common_dates))  # 确保日期排序一致
        
        for code in sorted_codes:
            df = data_dict[code]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
                aligned_df = df[df['date'].isin(common_dates_list)].copy()
                aligned_df = aligned_df.sort_values('date').reset_index(drop=True)
                aligned_dict[code] = aligned_df
        
        print(f"[ETF轮动] 对齐后共有 {len(common_dates_list)} 个交易日")
        return aligned_dict
    
    def backtest(self, days: int = 252, lookback_days: int = 20,
                 progress_callback: Optional[Callable[[str, dict], None]] = None,
                 abort_check: Optional[Callable[[], bool]] = None,
                 risk_control_mode: str = 'off') -> Dict[str, Any]:
        """
        回测ETF轮动策略
        
        参数:
            days: 回测天数
            lookback_days: 计算得分时的回看天数
            progress_callback: 可选，进度回调 (event_type, data)，如 ('progress', {'percent': 50, ...})
            abort_check: 可选，无参可调用，返回 True 时停止回测并返回局部结果
            risk_control_mode: 风控模式。'off'=回测不启用风控；'warn'=模拟盘记录警告但不阻断；'block'=阻断交易
            
        返回:
            回测结果字典（若被停止则含 aborted=True, aborted_message）
        """
        start_time = time.time()
        backtest_aborted = False
        
        # 加载数据：需要加载 days + lookback_days 天的数据，以便在回测开始时就有足够的历史数据计算得分
        # 例如：回测252天，回看20天，需要加载272天的数据
        # 注意：load_etf_data会在这个基础上增加额外的缓冲（技术指标需要的80天 + 1.8倍缓冲）
        data_load_days = days + lookback_days
        print(f"[ETF轮动] 需要 {data_load_days} 天数据（回测{days}天 + 回看{lookback_days}天），实际会加载更多以确保有足够缓冲")
        data_dict = self.load_etf_data(data_load_days)
        if not data_dict:
            return {
                "error": "无法加载ETF数据",
                "total_return": 0,
                "annual_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "total_trades": 0
            }
        
        # 对齐日期
        aligned_dict = self.align_data_dates(data_dict)
        if not aligned_dict:
            return {
                "error": "无法对齐ETF数据日期",
                "total_return": 0,
                "annual_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "total_trades": 0
            }
        
        # 检查对齐后的数据量
        first_etf_aligned = list(aligned_dict.values())[0]
        aligned_days = len(first_etf_aligned)
        if aligned_days < days:
            print(f"[ETF轮动] 警告：对齐后只有 {aligned_days} 个交易日，少于请求的 {days} 天")
            print(f"[ETF轮动] 可能原因：某些ETF的历史数据不足，或ETF上市时间较晚")
            print(f"[ETF轮动] 将使用所有可用的 {aligned_days} 个交易日进行回测")
        
        # 获取所有ETF的共同日期列表（使用第一个ETF，按代码排序确保一致性）
        # 按ETF代码排序，确保每次选择相同的ETF作为参考
        sorted_etf_codes = sorted(aligned_dict.keys())
        first_etf_code = sorted_etf_codes[0]
        first_etf = aligned_dict[first_etf_code]
        dates = first_etf['date'].tolist()
        
        # 获取对齐后的总天数（用于计算actual_data_idx）
        total_aligned_days = len(dates)
        
        # 检查数据是否足够
        if total_aligned_days < data_load_days:
            print(f"[ETF轮动] 警告：对齐后只有 {total_aligned_days} 个交易日，少于请求的 {data_load_days} 天（回测{days}天 + 回看{lookback_days}天）")
            print(f"[ETF轮动] 可能原因：某些ETF的历史数据不足，或ETF上市时间较晚")
            # 使用所有可用数据进行回测
            actual_days = max(0, total_aligned_days - lookback_days)  # 实际回测天数 = 总天数 - 回看天数
            if actual_days <= 0:
                return {
                    "error": f"数据不足：只有 {total_aligned_days} 天数据，无法进行回测（至少需要 {lookback_days + 1} 天）",
                    "total_return": 0,
                    "annual_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "total_trades": 0
                }
            # 保留所有数据用于计算得分，但只对最后actual_days天进行回测
            # 计算回测开始的索引位置
            backtest_start_idx = total_aligned_days - actual_days  # 回测开始位置
            dates = dates[-actual_days:]  # 只保留最后actual_days天用于回测循环
            # 不截断数据，保留完整的历史数据用于计算得分
            print(f"[ETF轮动] 将使用最后 {actual_days} 天进行回测，保留前 {backtest_start_idx} 天作为历史数据（索引0到{backtest_start_idx-1}，回测索引{backtest_start_idx}到{total_aligned_days-1}）")
        else:
            # 保留完整的历史数据用于计算得分，但只对最后days天进行回测
            # 不截断aligned_dict，保留所有数据
            # 计算回测开始的索引位置
            backtest_start_idx = total_aligned_days - days  # 回测开始位置
            dates = dates[-days:]  # 只保留最后days天用于回测循环
            actual_days = days
            print(f"[ETF轮动] 保留前 {backtest_start_idx} 天作为历史数据，使用最后 {days} 天进行回测（索引{backtest_start_idx}到{total_aligned_days-1}）")
        
        print(f"[ETF轮动] 开始回测，共 {len(dates)} 个交易日（请求{days}天），持仓数量：{self.top_k}")
        
        # 回测变量
        capital = self.initial_capital
        # 当前持仓：{etf_code: {'shares': int, 'entry_price': float}}
        current_positions = {}
        
        trades = []  # 交易记录
        equity_curve = []  # 资金曲线
        holdings_history = []  # 持仓历史
        last_rotation_day = -1  # 上次轮动的日期索引（-1表示从未轮动过）
        peak_equity = float(self.initial_capital)  # 风控：历史峰值权益

        # 逐日回测
        total_days = len(dates)
        for day_idx in range(total_days):
            # 进度与停止检查
            if progress_callback:
                progress = (day_idx / total_days) * 100 if total_days else 0
                progress_callback('progress', {
                    'current': day_idx + 1,
                    'total': total_days,
                    'percent': round(progress, 1),
                    'date': str(dates[day_idx]) if day_idx < total_days else '',
                    'elapsed': round(time.time() - start_time, 2)
                })
            if abort_check and abort_check():
                backtest_aborted = True
                print(f"\n[ETF轮动] 用户停止回测 @ Day {day_idx + 1}/{total_days}")
                break

            current_date = dates[day_idx]
            
            # 计算实际数据索引（包含历史数据）
            # day_idx是相对于回测开始日期的索引（0表示回测第一天）
            # actual_data_idx是aligned_dict中的实际索引
            # 例如：如果aligned_dict有272天，回测252天，backtest_start_idx=20
            #   day_idx=0时，actual_data_idx=20+0=20（回测第0日对应aligned_dict索引20）
            #   day_idx=105时，actual_data_idx=20+105=125（回测第105日对应aligned_dict索引125）
            actual_data_idx = backtest_start_idx + day_idx

            # 风控审计（回测不启用；模拟盘可记录警告或阻断）
            risk_no_new_buys = False
            if risk_control_mode != 'off':
                current_equity_for_risk = capital
                positions_for_risk = {}
                for etf_code, pos in current_positions.items():
                    if etf_code in aligned_dict and actual_data_idx < len(aligned_dict[etf_code]):
                        price = float(aligned_dict[etf_code].iloc[actual_data_idx]['close'])
                        mv = pos['shares'] * price
                        current_equity_for_risk += mv
                        positions_for_risk[etf_code] = {'market_value': mv}
                recent_returns = get_recent_daily_returns_from_equity(equity_curve, lookback=10)
                risk_input = RiskAuditInput(
                    cash=capital,
                    total_value=current_equity_for_risk,
                    initial_capital=float(self.initial_capital),
                    positions_value=current_equity_for_risk - capital,
                    positions=positions_for_risk,
                    recent_daily_returns=recent_returns,
                    peak_value=peak_equity,
                    current_date=str(current_date),
                )
                risk_result = audit_risk(risk_input)
                would_block = not risk_result.pass_audit and (risk_result.action in ('stop', 'pause_trading') or (risk_result.action == 'reduce_position' and 'over_weight_code' not in risk_result.details))
                if risk_control_mode == 'block':
                    risk_no_new_buys = would_block
                elif risk_control_mode == 'warn' and would_block:
                    # 模拟盘：记录风控警告，但不阻断，供用户人工确认
                    if day_idx % 10 == 0:
                        print(f"[ETF轮动] {day_idx}日 风控警告（不阻断）: {risk_result.reason}")
                    if progress_callback:
                        progress_callback('risk_warning', {'date': str(current_date), 'reason': risk_result.reason, 'action': risk_result.action})
            # risk_control_mode == 'off' 时 risk_no_new_buys 保持 False，不阻断
            
            # ========== 轮动触发检查 ==========
            # 触发条件1: 每rotation_interval天检查一次（定期轮动）
            # 触发条件2: 首次建仓（从未轮动过且当前无持仓）
            if last_rotation_day < 0:
                is_rotation_day = (day_idx % self.rotation_interval == 0)
                is_first_rotation = (len(current_positions) == 0)
                should_rotate = (is_rotation_day or is_first_rotation) and not risk_no_new_buys
            else:
                days_since_last_rotation = day_idx - last_rotation_day
                should_rotate = (days_since_last_rotation >= self.rotation_interval) and not risk_no_new_buys
                if should_rotate:
                    print(f"[ETF轮动] {day_idx}日 轮动间隔到期（距离上次轮动 {days_since_last_rotation} 天，间隔 {self.rotation_interval} 天）")
            
            # ========== 再平衡触发检查 ==========
            should_rebalance = False
            if self.rebalance_interval is not None and len(current_positions) > 0 and not risk_no_new_buys:
                should_rebalance = (day_idx % self.rebalance_interval == 0) and (day_idx % self.rotation_interval != 0)
            
            if should_rotate:
                # ========== 步骤1: 计算所有ETF的得分 ==========
                # 检查是否有足够的数据进行轮动
                min_data_available = min([len(df) for df in aligned_dict.values()] + [lookback_days])
                if actual_data_idx < lookback_days:
                    # 数据不足，跳过轮动（这种情况理论上不应该发生，因为我们已经加载了足够的数据）
                    print(f"[ETF轮动] {day_idx}日 数据不足（实际索引{actual_data_idx}，需要{lookback_days}天），跳过轮动")
                    # 记录当前权益（即使没有持仓）
                    current_equity = capital
                    for etf_code, pos in current_positions.items():
                        current_df = aligned_dict[etf_code]
                        if actual_data_idx < len(current_df):
                            current_row = current_df.iloc[actual_data_idx]
                            current_price = current_row['close']
                            current_equity += pos['shares'] * current_price
                    equity_curve.append(current_equity)
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    holdings_dict = {}
                    holdings_value_dict = {}
                    holdings_ratio_dict = {}
                    for etf_code, pos in current_positions.items():
                        holdings_dict[etf_code] = pos['shares']
                        current_df = aligned_dict[etf_code]
                        if actual_data_idx < len(current_df):
                            current_row = current_df.iloc[actual_data_idx]
                            current_price = current_row['close']
                            market_value = pos['shares'] * current_price
                            holdings_value_dict[etf_code] = market_value
                            if current_equity > 0:
                                holdings_ratio_dict[etf_code] = (market_value / current_equity) * 100
                            else:
                                holdings_ratio_dict[etf_code] = 0
                    holdings_history.append({
                        'date': str(current_date),
                        'holdings': holdings_dict.copy(),
                        'holdings_value': holdings_value_dict.copy(),
                        'holdings_ratio': holdings_ratio_dict.copy(),
                        'equity': current_equity
                    })
                    continue  # 跳过本次轮动
                
                etf_scores = {}
                for code, df in aligned_dict.items():
                    # 使用实际数据索引（包含历史数据）
                    # 计算得分应该使用当前日期及之前lookback_days-1天的数据（共lookback_days天，包含当前日期）
                    # 因为calculate_etf_score需要当前日期的收盘价来计算技术指标（如RSI、MA等）
                    # 
                    # 索引映射关系：
                    # - aligned_dict包含完整对齐数据（例如272天，索引0-271）
                    # - dates是最后days天的日期列表（例如最后252天，用于回测循环）
                    # - day_idx是dates的索引（0到251）
                    # - backtest_start_idx是回测开始位置（例如272-252=20）
                    # - actual_data_idx = backtest_start_idx + day_idx 是aligned_dict中的实际索引
                    # 
                    # 例如：回看20天，假设aligned_dict有272天数据（索引0-271），回测252天
                    # - backtest_start_idx = 272 - 252 = 20
                    # - 第0天回测时（day_idx=0, actual_data_idx=20），应该取索引1-20的数据（共20天，包含第0日，即索引20）
                    # - 第105天回测时（day_idx=105, actual_data_idx=20+105=125），应该取索引106-125的数据（共20天，包含第105日，即索引125）
                    # 
                    # 注意：索引106-125看起来像是"未来数据"，但实际上：
                    # - actual_data_idx=125对应的是回测第105日的日期（在aligned_dict中的索引）
                    # - 索引125的数据是第105日的收盘价，这是我们在回测第105日时已知的数据
                    # - 我们需要用这个收盘价来计算技术指标，然后决定在第105日收盘时是否买入/卖出
                    # - 这是合理的，因为在实际交易中，我们也是在收盘时看到收盘价后才做决策
                    if actual_data_idx < len(df):
                        # 包含当前日期：使用索引(actual_data_idx - lookback_days + 1)到actual_data_idx（共lookback_days天）
                        # 验证：第105日（actual_data_idx=125），应该使用索引106-125（125-20+1=106, 125+1=126，切片[106:126]即索引106-125）
                        start_idx = max(0, actual_data_idx - lookback_days + 1)
                        end_idx = actual_data_idx + 1  # 包含当前日期
                        df_until_now = df.iloc[start_idx:end_idx].copy()
                        
                        if len(df_until_now) >= lookback_days:
                            score = self.calculate_etf_score(df_until_now, lookback_days)
                            etf_scores[code] = score
                            print(f"[ETF轮动] {day_idx}日 {code} 得分: {score:.2f} (使用数据索引{start_idx}-{end_idx-1}，共{len(df_until_now)}天，当前日期索引{actual_data_idx})")
                        elif len(df_until_now) > 0:
                            # 如果数据不足但接近lookback_days，使用可用数据计算（但会警告）
                            score = self.calculate_etf_score(df_until_now, len(df_until_now))
                            etf_scores[code] = score
                            print(f"[ETF轮动] {day_idx}日 {code} 得分: {score:.2f} (数据不足，仅{len(df_until_now)}天，建议至少{lookback_days}天，当前日期索引{actual_data_idx})")
                
                # ========== 步骤2: 选择Top-K ETF（考虑最低得分阈值和AI分析） ==========
                target_etfs = []
                if etf_scores:
                    # 如果使用AI分析（调用前检查停止，避免长时间 LLM 调用后才检测）
                    if self.use_ai:
                        if abort_check and abort_check():
                            backtest_aborted = True
                            print(f"\n[ETF轮动] 用户停止回测（AI 分析前） @ Day {day_idx + 1}/{total_days}")
                            break
                        print(f"[ETF轮动] {day_idx}日 使用AI分析选择ETF...")
                        # 准备当前日期的数据切片（用于AI分析）
                        current_data_dict = {}
                        for code, df in aligned_dict.items():
                            if actual_data_idx < len(df):
                                # 取当前日期及之前lookback_days-1天的数据（共lookback_days天，包含当前日期）
                                # actual_data_idx是aligned_dict中的实际索引
                                start_idx = max(0, actual_data_idx - lookback_days + 1)
                                end_idx = actual_data_idx + 1  # 包含当前日期
                                current_data_dict[code] = df.iloc[start_idx:end_idx].copy()
                        
                        # 先计算技术指标推荐的ETF
                        sorted_etfs = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)
                        if self.min_score_threshold > 0:
                            technical_recommended = [code for code, score in sorted_etfs[:self.top_k] if score >= self.min_score_threshold]
                        else:
                            technical_recommended = [code for code, score in sorted_etfs[:self.top_k]]
                        if progress_callback:
                            progress_callback('etf_ai_start', {
                                'date': str(current_date),
                                'etf_count': len(etf_scores),
                                'technical_top': technical_recommended,
                            })
                        
                        # 调用AI分析（传入技术指标推荐的ETF）
                        ai_recommended, ai_reasoning = get_ai_rotation_suggestion(
                            etf_codes=list(etf_scores.keys()),
                            etf_scores=etf_scores,
                            etf_data_dict=current_data_dict,
                            lookback_days=lookback_days,
                            top_k=self.top_k,
                            min_score_threshold=self.min_score_threshold,
                            technical_recommended=technical_recommended
                        )
                        
                        # 记录AI的推荐理由
                        if ai_reasoning:
                            for code, reason in ai_reasoning.items():
                                print(f"[ETF AI轮动] {code} 推荐理由: {reason}")
                        
                        if ai_recommended:
                            # 使用AI推荐，但需要检查最低得分阈值
                            if self.min_score_threshold > 0:
                                valid_ai_etfs = [code for code in ai_recommended if etf_scores.get(code, 0) >= self.min_score_threshold]
                                if valid_ai_etfs:
                                    target_etfs = valid_ai_etfs[:self.top_k]
                                    print(f"[ETF轮动] {day_idx}日 AI推荐ETF（得分≥{self.min_score_threshold}）: {target_etfs}")
                                else:
                                    target_etfs = []
                                    print(f"[ETF轮动] {day_idx}日 AI推荐的ETF得分都低于阈值{self.min_score_threshold}，空仓不买入")
                            else:
                                target_etfs = ai_recommended[:self.top_k]
                                print(f"[ETF轮动] {day_idx}日 AI推荐ETF: {target_etfs}")
                        else:
                            # AI分析失败，回退到技术指标
                            print(f"[ETF轮动] {day_idx}日 AI分析失败，回退到技术指标选择")
                            # 注意：不要修改self.use_ai，因为这是实例属性，会影响后续的回测
                            # 如果AI失败，直接使用技术指标即可
                    
                    # 如果不使用AI或AI分析失败，使用技术指标
                    if not target_etfs:
                        # 按得分排序，取前top_k只
                        sorted_etfs = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # 取得分前 top_k 只；仅当这 top_k 只全部低于阈值时才空仓（实现真正的 top-k 持仓）
                        top_k_codes = [code for code, score in sorted_etfs[:self.top_k]]
                        top_k_scores = [score for code, score in sorted_etfs[:self.top_k]]
                        if self.min_score_threshold > 0:
                            if top_k_scores and any(s >= self.min_score_threshold for s in top_k_scores):
                                target_etfs = top_k_codes
                                print(f"[ETF轮动] {day_idx}日 Top-{self.top_k} ETF（前{self.top_k}名，至少1只≥{self.min_score_threshold}）: {target_etfs}, 得分: {top_k_scores}")
                            else:
                                target_etfs = []
                                print(f"[ETF轮动] {day_idx}日 前{self.top_k}名得分均低于阈值{self.min_score_threshold}，空仓不买入")
                                if top_k_scores:
                                    print(f"[ETF轮动] {day_idx}日 ETF得分: {[(code, f'{score:.2f}') for code, score in sorted_etfs[:min(5, len(sorted_etfs))]]}")
                        else:
                            target_etfs = top_k_codes
                            print(f"[ETF轮动] {day_idx}日 Top-{self.top_k} ETF: {target_etfs}, 得分: {top_k_scores}")
                    
                    # ========== 步骤3: 分类ETF（卖出/买入/保持） ==========
                    current_etf_set = set(current_positions.keys())
                    target_etf_set = set(target_etfs)
                    
                    # 需要卖出的ETF：当前持有但不在目标Top-K中
                    etfs_to_sell = current_etf_set - target_etf_set
                    # 需要买入的ETF：在目标Top-K中但当前未持有
                    etfs_to_buy = target_etf_set - current_etf_set
                    # 需要保持的ETF：既在当前持仓又在目标Top-K中（可能需要调整仓位）
                    etfs_to_keep = current_etf_set & target_etf_set
                    
                    print(f"[ETF轮动] {day_idx}日 当前持仓: {list(current_etf_set)}, 目标持仓: {target_etfs}")
                    print(f"[ETF轮动] {day_idx}日 需要卖出: {list(etfs_to_sell)}, 需要买入: {list(etfs_to_buy)}, 需要保持: {list(etfs_to_keep)}")
                    
                    # 推送决策摘要（参考组合回测的 llm_decision）
                    if progress_callback:
                        top_scores = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                        scores_summary = {c: round(s, 1) for c, s in top_scores}
                        progress_callback('etf_decision', {
                            'date': str(current_date),
                            'target_etfs': target_etfs,
                            'priority_buy': list(etfs_to_buy),
                            'priority_sell': list(etfs_to_sell),
                            'etf_scores': scores_summary,
                            'use_ai': self.use_ai,
                            'reason': f'Top-{self.top_k}轮动' + ('（AI分析）' if self.use_ai else '（技术指标）'),
                        })
                    
                    # ========== 步骤4: 执行交易（按顺序：卖出 → 调整 → 买入） ==========
                    
                    # ---------- 4.1 卖出不需要的ETF（全部清仓） ----------
                    for etf_code in etfs_to_sell:
                        if etf_code in current_positions:
                            pos = current_positions[etf_code]
                            current_df = aligned_dict[etf_code]
                            # 使用索引查找当前日期的数据
                            if actual_data_idx < len(current_df):
                                current_row = current_df.iloc[actual_data_idx]
                                sell_price = current_row['close'] * (1 - self.slippage)
                                shares = pos['shares']
                                revenue = shares * sell_price * (1 - self.commission_rate)
                                capital += revenue
                                
                                trades.append({
                                    'date': str(current_date),
                                    'index': day_idx,
                                    'type': 'sell',
                                    'etf_code': etf_code,
                                    'price': sell_price,
                                    'shares': shares,
                                    'revenue': revenue,
                                    'reason': f'轮动卖出（不在top-{self.top_k}）'
                                })
                                if progress_callback:
                                    progress_callback('etf_trade', {
                                        'date': str(current_date), 'stock_code': etf_code,
                                        'action': 'SELL', 'shares': shares, 'price': sell_price,
                                        'reason': f'轮动卖出（不在top-{self.top_k}）',
                                    })
                                print(f"[ETF轮动] {day_idx}日 卖出 {etf_code}: {shares}股 @ ¥{sell_price:.2f}")
                                del current_positions[etf_code]
                    
                    # ---------- 4.2 重新计算总资产（卖出后） ----------
                    # 总资产 = 现金 + 所有保持ETF的当前市值
                    total_target_count = len(target_etfs)
                    if total_target_count > 0:
                        # 计算当前总资产（卖出后，买入前）
                        current_total_value = capital  # 现金余额（包括卖出回收的资金）
                        for etf_code in etfs_to_keep:
                            if etf_code in current_positions:
                                current_df = aligned_dict[etf_code]
                                # 使用索引查找当前日期的数据
                                if actual_data_idx < len(current_df):
                                    current_row = current_df.iloc[actual_data_idx]
                                    current_price = current_row['close']
                                    # 累加保持ETF的当前市值
                                    current_total_value += current_positions[etf_code]['shares'] * current_price
                        
                        # ---------- 4.3 计算目标仓位（等权重或凯利公式，且单标的不超过风控上限） ----------
                        max_single_pct = DEFAULT_RISK_CONFIG.get('max_single_position_pct', 0.45)
                        if self.position_strategy == "kelly":
                            target_weights = self._compute_kelly_weights(
                                aligned_dict, target_etfs, actual_data_idx, lookback_days, max_single_pct
                            )
                            # 补全未在 kelly 中的（如数据不足）
                            for etf_code in target_etfs:
                                if etf_code not in target_weights:
                                    target_weights[etf_code] = min(0.95 / total_target_count, max_single_pct)
                        else:
                            target_weight = min(0.95 / total_target_count, max_single_pct)
                            target_weights = {etf_code: target_weight for etf_code in target_etfs}
                        
                        # ---------- 4.4 调整保持的ETF仓位（加仓/减仓/不变） ----------
                        # 只有在设置了再平衡间隔时才调整保持的ETF仓位
                        if self.rebalance_interval is not None:
                            for etf_code in list(etfs_to_keep):
                                current_df = aligned_dict[etf_code]
                                # 使用索引查找当前日期的数据
                                if actual_data_idx < len(current_df):
                                    current_row = current_df.iloc[actual_data_idx]
                                    current_price = current_row['close']
                                    target_value = current_total_value * target_weights.get(etf_code, 1.0 / total_target_count)
                                    target_shares = int(target_value / current_price)
                                    
                                    if etf_code in current_positions:
                                        current_shares = current_positions[etf_code]['shares']
                                        if target_shares != current_shares:
                                            # 需要调整仓位
                                            diff_shares = target_shares - current_shares
                                            
                                            if diff_shares > 0:
                                                # ---------- 情况A: 加仓（当前持仓 < 目标持仓） ----------
                                                buy_price = current_price * (1 + self.slippage)
                                                cost = diff_shares * buy_price * (1 + self.commission_rate)
                                                if cost <= capital:
                                                    capital -= cost
                                                    current_positions[etf_code]['shares'] = target_shares
                                                    trades.append({
                                                        'date': str(current_date),
                                                        'index': day_idx,
                                                        'type': 'buy',
                                                        'etf_code': etf_code,
                                                        'price': buy_price,
                                                        'shares': diff_shares,
                                                        'cost': cost,
                                                        'score': etf_scores.get(etf_code, 0),
                                                        'reason': f'加仓（得分{etf_scores.get(etf_code, 0):.1f}）'
                                                    })
                                                    if progress_callback:
                                                        progress_callback('etf_trade', {
                                                            'date': str(current_date), 'stock_code': etf_code,
                                                            'action': 'BUY', 'shares': diff_shares, 'price': buy_price,
                                                            'reason': f'加仓（得分{etf_scores.get(etf_code, 0):.1f}）',
                                                        })
                                                    print(f"[ETF轮动] {day_idx}日 加仓 {etf_code}: +{diff_shares}股 @ ¥{buy_price:.2f}")
                                            
                                            elif diff_shares < 0:
                                                # ---------- 情况B: 减仓（当前持仓 > 目标持仓） ----------
                                                sell_price = current_price * (1 - self.slippage)
                                                revenue = abs(diff_shares) * sell_price * (1 - self.commission_rate)
                                                capital += revenue
                                                current_positions[etf_code]['shares'] = target_shares
                                                trades.append({
                                                    'date': str(current_date),
                                                    'index': day_idx,
                                                    'type': 'sell',
                                                    'etf_code': etf_code,
                                                    'price': sell_price,
                                                    'shares': abs(diff_shares),
                                                    'revenue': revenue,
                                                    'reason': f'减仓调整'
                                                })
                                                if progress_callback:
                                                    progress_callback('etf_trade', {
                                                        'date': str(current_date), 'stock_code': etf_code,
                                                        'action': 'SELL', 'shares': abs(diff_shares), 'price': sell_price,
                                                        'reason': '减仓调整',
                                                    })
                                                print(f"[ETF轮动] {day_idx}日 减仓 {etf_code}: -{abs(diff_shares)}股 @ ¥{sell_price:.2f}")
                                            
                                            # 如果 diff_shares == 0，则无需调整（很少见）
                        else:
                            # 无再平衡模式：保持的ETF不调整仓位
                            print(f"[ETF轮动] {day_idx}日 无再平衡模式：保持的ETF不调整仓位 {list(etfs_to_keep)}")
                        
                        # ---------- 4.5 买入新的ETF（新建仓） ----------
                        # 如果有再平衡，使用总资产按等权重分配
                        # 如果无再平衡：默认只使用卖出回收的资金；但当新增ETF且现金不足时，需减持保持的ETF以筹集资金，避免产生 1 股等无意义交易
                        if self.rebalance_interval is not None:
                            # 有再平衡：使用总资产按等权重分配
                            for etf_code in etfs_to_buy:
                                if etf_code not in aligned_dict or actual_data_idx >= len(aligned_dict[etf_code]):
                                    print(f"[ETF轮动] {day_idx}日 跳过买入 {etf_code}: 无数据或索引越界")
                                    continue
                                current_df = aligned_dict[etf_code]
                                current_row = current_df.iloc[actual_data_idx]
                                # 兼容 'close' / '收盘'，并转为 float
                                raw_close = current_row.get('close', current_row.get('收盘', None))
                                if raw_close is None or (hasattr(raw_close, '__float__') and np.isnan(float(raw_close))) or float(raw_close) <= 0:
                                    print(f"[ETF轮动] {day_idx}日 跳过买入 {etf_code}: 收盘价无效 (close={raw_close})")
                                    continue
                                buy_price = float(raw_close) * (1 + self.slippage)
                                target_value = current_total_value * target_weights.get(etf_code, 1.0 / total_target_count)
                                # 按目标市值算股数，再按可用资金上限（含手续费）取整，避免 cost 略大于 capital 导致无法买入
                                price_with_fee = buy_price * (1 + self.commission_rate)
                                shares_by_value = int(target_value / buy_price)
                                shares_by_capital = int(capital / price_with_fee) if price_with_fee > 0 else 0
                                shares = min(shares_by_value, shares_by_capital)
                                if shares <= 0 and target_value >= MIN_TRADE_VALUE and buy_price > 0:
                                    shares = max(100, int(MIN_TRADE_VALUE / price_with_fee))
                                if shares <= 0:
                                    print(f"[ETF轮动] {day_idx}日 跳过买入 {etf_code}: 目标股数=0 (目标市值={target_value:.0f}, 价={buy_price:.4f})")
                                    continue
                                cost = shares * price_with_fee
                                if cost < MIN_TRADE_VALUE:
                                    print(f"[ETF轮动] {day_idx}日 跳过买入 {etf_code}: 金额不足最小交易额 (cost={cost:.0f} < {MIN_TRADE_VALUE})")
                                    continue
                                capital -= cost
                                current_positions[etf_code] = {'shares': shares, 'entry_price': buy_price}
                                trades.append({
                                    'date': str(current_date),
                                    'index': day_idx,
                                    'type': 'buy',
                                    'etf_code': etf_code,
                                    'price': buy_price,
                                    'shares': shares,
                                    'cost': cost,
                                    'score': etf_scores.get(etf_code, 0),
                                    'reason': f'轮动买入（得分{etf_scores.get(etf_code, 0):.1f}，top-{self.top_k}）'
                                })
                                if progress_callback:
                                    progress_callback('etf_trade', {
                                        'date': str(current_date), 'stock_code': etf_code,
                                        'action': 'BUY', 'shares': shares, 'price': buy_price,
                                        'reason': f'轮动买入（得分{etf_scores.get(etf_code, 0):.1f}，top-{self.top_k}）',
                                    })
                                print(f"[ETF轮动] {day_idx}日 买入 {etf_code}: {shares}股 @ ¥{buy_price:.2f} (得分{etf_scores.get(etf_code, 0):.1f})")
                        else:
                            # 无再平衡：买入新ETF
                            # 当新增ETF时，若现金不足，需减持保持的ETF以筹集资金，避免产生 1 股等无意义交易
                            if len(etfs_to_buy) > 0:
                                # 计算新买入ETF总需求（等权重或凯利）
                                total_needed_for_new = sum(
                                    current_total_value * target_weights.get(e, 1.0 / total_target_count) for e in etfs_to_buy
                                )
                                
                                # 若现金不足，减持保持的ETF以筹集资金
                                if capital < total_needed_for_new and len(etfs_to_keep) > 0:
                                    shortfall = total_needed_for_new - capital
                                    # 按市值比例减持各保持的ETF
                                    keep_mv = {}
                                    for etf_code in etfs_to_keep:
                                        if etf_code in current_positions and etf_code in aligned_dict:
                                            if actual_data_idx < len(aligned_dict[etf_code]):
                                                p = float(aligned_dict[etf_code].iloc[actual_data_idx]['close'])
                                                keep_mv[etf_code] = current_positions[etf_code]['shares'] * p
                                    total_keep_mv = sum(keep_mv.values())
                                    if total_keep_mv > 0 and shortfall > 0:
                                        for etf_code in etfs_to_keep:
                                            if etf_code not in current_positions or etf_code not in aligned_dict:
                                                continue
                                            if actual_data_idx >= len(aligned_dict[etf_code]):
                                                continue
                                            sell_ratio = min(1.0, shortfall / total_keep_mv) * (keep_mv.get(etf_code, 0) / total_keep_mv)
                                            if sell_ratio <= 0:
                                                continue
                                            current_df = aligned_dict[etf_code]
                                            current_row = current_df.iloc[actual_data_idx]
                                            sell_price = float(current_row['close']) * (1 - self.slippage)
                                            to_sell_value = shortfall * (keep_mv[etf_code] / total_keep_mv)
                                            shares_to_sell = int(to_sell_value / sell_price)
                                            if shares_to_sell <= 0:
                                                continue
                                            current_shares = current_positions[etf_code]['shares']
                                            # 至少保留 1 股，避免将保持的 ETF 全部清仓
                                            shares_to_sell = min(shares_to_sell, max(0, current_shares - 1))
                                            if shares_to_sell <= 0:
                                                continue
                                            revenue = shares_to_sell * sell_price * (1 - self.commission_rate)
                                            capital += revenue
                                            shortfall -= revenue
                                            current_positions[etf_code]['shares'] -= shares_to_sell
                                            if current_positions[etf_code]['shares'] <= 0:
                                                del current_positions[etf_code]
                                            trades.append({
                                                'date': str(current_date),
                                                'index': day_idx,
                                                'type': 'sell',
                                                'etf_code': etf_code,
                                                'price': sell_price,
                                                'shares': shares_to_sell,
                                                'revenue': revenue,
                                                'reason': '减持以筹集资金买入新ETF（无再平衡模式）'
                                            })
                                            if progress_callback:
                                                progress_callback('etf_trade', {
                                                    'date': str(current_date), 'stock_code': etf_code,
                                                    'action': 'SELL', 'shares': shares_to_sell, 'price': sell_price,
                                                    'reason': '减持以筹集资金买入新ETF（无再平衡模式）',
                                                })
                                            print(f"[ETF轮动] {day_idx}日 减持 {etf_code}: {shares_to_sell}股 @ ¥{sell_price:.2f} （筹集资金买入新ETF）")
                                
                                # 按等权重或凯利分配资金给新买入的ETF
                                total_w_buy = sum(target_weights.get(e, 1.0 / total_target_count) for e in etfs_to_buy) or 1.0
                                for etf_code in etfs_to_buy:
                                    current_df = aligned_dict[etf_code]
                                    if actual_data_idx < len(current_df):
                                        current_row = current_df.iloc[actual_data_idx]
                                        buy_price = current_row['close'] * (1 + self.slippage)
                                        capital_for_etf = capital * (target_weights.get(etf_code, 1.0 / total_target_count) / total_w_buy)
                                        shares = int(capital_for_etf / (buy_price * (1 + self.commission_rate)))
                                        
                                        if shares > 0:
                                            cost = shares * buy_price * (1 + self.commission_rate)
                                            # 最小交易金额检查：避免 1 股等无意义交易
                                            if cost < MIN_TRADE_VALUE:
                                                continue
                                            if cost <= capital:
                                                capital -= cost
                                                current_positions[etf_code] = {'shares': shares, 'entry_price': buy_price}
                                                trades.append({
                                                    'date': str(current_date),
                                                    'index': day_idx,
                                                    'type': 'buy',
                                                    'etf_code': etf_code,
                                                    'price': buy_price,
                                                    'shares': shares,
                                                    'cost': cost,
                                                    'score': etf_scores.get(etf_code, 0),
                                                    'reason': f'轮动买入（得分{etf_scores.get(etf_code, 0):.1f}，top-{self.top_k}，无再平衡）'
                                                })
                                                if progress_callback:
                                                    progress_callback('etf_trade', {
                                                        'date': str(current_date), 'stock_code': etf_code,
                                                        'action': 'BUY', 'shares': shares, 'price': buy_price,
                                                        'reason': f'轮动买入（得分{etf_scores.get(etf_code, 0):.1f}，top-{self.top_k}，无再平衡）',
                                                    })
                                                print(f"[ETF轮动] {day_idx}日 买入 {etf_code}: {shares}股 @ ¥{buy_price:.2f} (得分{etf_scores.get(etf_code, 0):.1f}，无再平衡模式)")
                    
                    # 轮动执行完成，更新上次轮动日期
                    last_rotation_day = day_idx
                    print(f"[ETF轮动] {day_idx}日 轮动完成，下次轮动将在第 {day_idx + self.rotation_interval} 日")
            
            elif should_rebalance:
                # ========== 再平衡逻辑：调整现有持仓的仓位到等权重或凯利（不换ETF） ==========
                if len(current_positions) > 0:
                    # 计算当前总资产
                    current_total_value = capital
                    for etf_code in current_positions.keys():
                        current_df = aligned_dict[etf_code]
                        if actual_data_idx < len(current_df):
                            current_row = current_df.iloc[actual_data_idx]
                            current_price = current_row['close']
                            current_total_value += current_positions[etf_code]['shares'] * current_price
                    
                    # 计算目标权重（等权重或凯利，且单标的不超过风控上限）
                    total_target_count = len(current_positions)
                    max_single_pct = DEFAULT_RISK_CONFIG.get('max_single_position_pct', 0.45)
                    if self.position_strategy == "kelly":
                        rebalance_weights = self._compute_kelly_weights(
                            aligned_dict, list(current_positions.keys()), actual_data_idx, lookback_days, max_single_pct
                        )
                        for etf_code in current_positions:
                            if etf_code not in rebalance_weights:
                                rebalance_weights[etf_code] = min(0.95 / total_target_count, max_single_pct)
                    else:
                        target_weight = min(0.95 / total_target_count, max_single_pct)
                        rebalance_weights = {etf_code: target_weight for etf_code in current_positions}
                    
                    # 调整每个ETF的仓位到目标权重
                    for etf_code in list(current_positions.keys()):
                        current_df = aligned_dict[etf_code]
                        if actual_data_idx < len(current_df):
                            current_row = current_df.iloc[actual_data_idx]
                            current_price = current_row['close']
                            target_value = current_total_value * rebalance_weights.get(etf_code, 1.0 / total_target_count)
                            target_shares = int(target_value / current_price)
                            
                            if etf_code in current_positions:
                                current_shares = current_positions[etf_code]['shares']
                                if target_shares != current_shares:
                                    diff_shares = target_shares - current_shares
                                    
                                    if diff_shares > 0:
                                        # 加仓
                                        buy_price = current_price * (1 + self.slippage)
                                        cost = diff_shares * buy_price * (1 + self.commission_rate)
                                        if cost <= capital:
                                            capital -= cost
                                            current_positions[etf_code]['shares'] = target_shares
                                            trades.append({
                                                'date': str(current_date),
                                                'index': day_idx,
                                                'type': 'buy',
                                                'etf_code': etf_code,
                                                'price': buy_price,
                                                'shares': diff_shares,
                                                'cost': cost,
                                                'reason': f'再平衡加仓'
                                            })
                                            if progress_callback:
                                                progress_callback('etf_trade', {
                                                    'date': str(current_date), 'stock_code': etf_code,
                                                    'action': 'BUY', 'shares': diff_shares, 'price': buy_price,
                                                    'reason': '再平衡加仓',
                                                })
                                            print(f"[ETF轮动] {day_idx}日 再平衡加仓 {etf_code}: +{diff_shares}股 @ ¥{buy_price:.2f}")
                                    
                                    elif diff_shares < 0:
                                        # 减仓
                                        sell_price = current_price * (1 - self.slippage)
                                        revenue = abs(diff_shares) * sell_price * (1 - self.commission_rate)
                                        capital += revenue
                                        current_positions[etf_code]['shares'] = target_shares
                                        trades.append({
                                            'date': str(current_date),
                                            'index': day_idx,
                                            'type': 'sell',
                                            'etf_code': etf_code,
                                            'price': sell_price,
                                            'shares': abs(diff_shares),
                                            'revenue': revenue,
                                            'reason': f'再平衡减仓'
                                        })
                                        if progress_callback:
                                            progress_callback('etf_trade', {
                                                'date': str(current_date), 'stock_code': etf_code,
                                                'action': 'SELL', 'shares': abs(diff_shares), 'price': sell_price,
                                                'reason': '再平衡减仓',
                                            })
                                        print(f"[ETF轮动] {day_idx}日 再平衡减仓 {etf_code}: -{abs(diff_shares)}股 @ ¥{sell_price:.2f}")
            
            # ========== 每日权益计算与记录 ==========
            # 计算当前权益 = 现金 + 所有持仓市值
            current_equity = capital
            for etf_code, pos in current_positions.items():
                current_df = aligned_dict[etf_code]
                # 使用索引查找当前日期的数据
                if actual_data_idx < len(current_df):
                    current_row = current_df.iloc[actual_data_idx]
                    current_price = current_row['close']
                    # 累加持仓市值
                    current_equity += pos['shares'] * current_price
            
            # 记录资金曲线并更新风控峰值
            equity_curve.append(current_equity)
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # 记录持仓历史（用于后续分析和图表展示）
            holdings_dict = {}
            holdings_value_dict = {}  # 持仓市值
            holdings_ratio_dict = {}  # 持仓比例（基于实际市值）
            
            for etf_code, pos in current_positions.items():
                holdings_dict[etf_code] = pos['shares']
                # 计算当前市值
                current_df = aligned_dict[etf_code]
                if actual_data_idx < len(current_df):
                    current_row = current_df.iloc[actual_data_idx]
                    current_price = current_row['close']
                    market_value = pos['shares'] * current_price
                    holdings_value_dict[etf_code] = market_value
                    # 计算持仓比例（相对于总资产）
                    if current_equity > 0:
                        holdings_ratio_dict[etf_code] = (market_value / current_equity) * 100
                    else:
                        holdings_ratio_dict[etf_code] = 0
            
            holdings_history.append({
                'date': str(current_date),
                'holdings': holdings_dict.copy(),  # 股数
                'holdings_value': holdings_value_dict.copy(),  # 市值
                'holdings_ratio': holdings_ratio_dict.copy(),  # 持仓比例（%）
                'equity': current_equity
            })
        
        # 计算回测指标
        equity_curve = np.array(equity_curve)
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital if len(equity_curve) > 0 else 0
        
        # 计算年化收益率
        trading_days = len(dates)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算夏普比率
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 计算胜率（买入到卖出的盈亏）
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        win_trades = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['revenue'] > buy_trades[i]['cost']:
                win_trades += 1
        win_rate = win_trades / len(sell_trades) if len(sell_trades) > 0 else 0
        
        # 计算每只ETF的总盈亏
        etf_total_cost = {}  # etf_code -> 总买入成本
        etf_total_revenue = {}  # etf_code -> 总卖出收入
        for t in trades:
            code = t.get('etf_code')
            if not code:
                continue
            if t['type'] == 'buy':
                cost = t.get('cost', t.get('shares', 0) * t.get('price', 0) * (1 + self.commission_rate))
                etf_total_cost[code] = etf_total_cost.get(code, 0) + cost
            elif t['type'] == 'sell':
                revenue = t.get('revenue', t.get('shares', 0) * t.get('price', 0) * (1 - self.commission_rate))
                etf_total_revenue[code] = etf_total_revenue.get(code, 0) + revenue
        # 期末持仓市值
        final_holdings_value = {}
        if holdings_history:
            last = holdings_history[-1]
            final_holdings_value = last.get('holdings_value', {}) or {}
        # 汇总每只ETF的总盈亏（仅包含有交易的ETF）
        all_etf_codes = set(etf_total_cost.keys()) | set(etf_total_revenue.keys()) | set(final_holdings_value.keys())
        etf_pnl_summary = []
        for code in sorted(all_etf_codes):
            total_cost = etf_total_cost.get(code, 0)
            total_revenue = etf_total_revenue.get(code, 0)
            holdings_value = final_holdings_value.get(code, 0)
            pnl = total_revenue + holdings_value - total_cost
            pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0
            etf_pnl_summary.append({
                'etf_code': code,
                'etf_name': DEFAULT_ETF_LIST.get(code, code),
                'total_cost': round(total_cost, 2),
                'total_revenue': round(total_revenue, 2),
                'holdings_value': round(holdings_value, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            })
        etf_pnl_summary.sort(key=lambda x: -x['pnl'])
        
        backtest_time = time.time() - start_time
        
        # 计算实际回测天数
        actual_trading_days = len(dates)
        requested_days = days
        
        result = {
            "total_return": round(total_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "total_trades": len(sell_trades),
            "final_capital": round(equity_curve[-1], 2) if len(equity_curve) > 0 else self.initial_capital,
            "initial_capital": self.initial_capital,
            "trades": trades,
            "trades_summary": trades[-10:],
            "equity_curve": equity_curve.tolist()[-30:],
            "equity_curve_full": equity_curve.tolist(),
            "holdings_history": holdings_history[-30:],
            "holdings_history_full": holdings_history,
            "dates": [str(d) for d in dates],
            "etf_codes": self.etf_codes,
            "etf_pnl_summary": etf_pnl_summary,
            "rotation_interval": self.rotation_interval,
            "rebalance_interval": self.rebalance_interval,
            "min_score_threshold": self.min_score_threshold,
            "backtest_time": round(backtest_time, 2),
            "trading_days": actual_trading_days,
            "requested_days": requested_days,
            "start_date": str(dates[0]) if dates else "",
            "end_date": str(dates[-1]) if dates else ""
        }
        
        # 如果实际回测天数少于请求天数，添加警告信息
        if actual_trading_days < requested_days:
            result["warning"] = f"实际回测 {actual_trading_days} 天，少于请求的 {requested_days} 天。可能原因：某些ETF历史数据不足或上市时间较晚。"
        
        if backtest_aborted:
            result["aborted"] = True
            result["aborted_message"] = "回测已停止，以下为局部结果"
        
        return result
    
    def generate_backtest_charts(self, backtest_result: Dict[str, Any], 
                                save_path: Optional[str] = None) -> str:
        """
        生成回测图表
        
        参数:
            backtest_result: 回测结果字典
            save_path: 图表保存路径（可选）
            
        返回:
            图表的base64编码字符串或保存路径
        """
        from utils.matplotlib_chinese import setup_chinese_font
        setup_chinese_font()
        plt.rcParams['font.size'] = 9
        
        # 创建图表
        fig = plt.figure(figsize=(18, 13))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3,
                     left=0.08, right=0.95, top=0.94, bottom=0.05)
        
        # 获取数据
        equity_curve = np.array(backtest_result.get('equity_curve_full', []))
        dates = backtest_result.get('dates', [])
        trades = backtest_result.get('trades', [])
        holdings_history = backtest_result.get('holdings_history_full', [])
        
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
            ax3.invert_yaxis()
        
        # 4. 持仓ETF变化图（右中）- 使用堆叠面积图显示持仓占比
        ax4 = fig.add_subplot(gs[1, 1])
        if holdings_history and equity_curve.size > 0:
            # 提取所有出现过的ETF
            all_etfs = set()
            for h in holdings_history:
                if 'holdings' in h:
                    all_etfs.update(h['holdings'].keys())
            
            if all_etfs:
                unique_etfs = sorted(list(all_etfs))
                # 使用更鲜明的颜色
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_etfs), 10)))
                if len(unique_etfs) > 10:
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_etfs)))
                etf_color_map = {etf: colors[i % len(colors)] for i, etf in enumerate(unique_etfs)}
                
                # 计算每个ETF的持仓占比（基于实际市值，反映价格涨跌导致的自然变化）
                time_points = []
                etf_weights = {etf: [] for etf in unique_etfs}
                
                for i, h in enumerate(holdings_history):
                    if 'equity' in h and h['equity'] > 0:
                        time_points.append(i)
                        
                        # 使用实际持仓比例（基于市值），这样可以反映因价格涨跌导致的持仓比例变化
                        for etf_code in unique_etfs:
                            if 'holdings_ratio' in h and etf_code in h['holdings_ratio']:
                                # 使用实际计算的持仓比例
                                etf_weights[etf_code].append(h['holdings_ratio'][etf_code])
                            elif 'holdings' in h and etf_code in h['holdings'] and h['holdings'][etf_code] > 0:
                                # 如果没有持仓比例数据，回退到基于股数的估算（兼容旧数据）
                                # 计算实际持仓的ETF数量
                                active_holdings = [k for k, v in h.get('holdings', {}).items() if v > 0]
                                num_active = len(active_holdings)
                                if num_active > 0:
                                    weight_per_etf = 100.0 / num_active
                                else:
                                    weight_per_etf = 0
                                etf_weights[etf_code].append(weight_per_etf)
                            else:
                                etf_weights[etf_code].append(0)
                
                if time_points:
                    # 准备堆叠面积图数据
                    weights_array = np.array([etf_weights[etf] for etf in unique_etfs])
                    
                    # 绘制堆叠面积图
                    ax4.stackplot(time_points, *weights_array, 
                                 labels=unique_etfs,
                                 colors=[etf_color_map[etf] for etf in unique_etfs],
                                 alpha=0.7, edgecolors='white', linewidth=0.5)
                    
                    # 添加图例（只显示实际持仓过的ETF）
                    handles, labels = ax4.get_legend_handles_labels()
                    # 过滤掉从未持仓的ETF
                    filtered_handles = []
                    filtered_labels = []
                    for etf_code in unique_etfs:
                        if any(etf_weights[etf_code]):
                            idx = unique_etfs.index(etf_code)
                            if idx < len(handles):
                                filtered_handles.append(handles[idx])
                                filtered_labels.append(etf_code)
                    
                    if filtered_handles:
                        ax4.legend(filtered_handles, filtered_labels, 
                                  loc='upper left', fontsize=7, ncol=2,
                                  framealpha=0.9, edgecolor='gray', facecolor='white')
                    
                    ax4.set_title(f'持仓ETF占比变化（Top-{backtest_result.get("top_k", 1)}）', 
                                fontsize=13, fontweight='bold', pad=12)
                    ax4.set_xlabel('交易日', fontsize=10)
                    ax4.set_ylabel('持仓占比 (%)', fontsize=10)
                    ax4.set_ylim(0, 100)
                    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
                    ax4.tick_params(labelsize=9)
                    
                    # 添加y轴刻度
                    ax4.set_yticks([0, 25, 50, 75, 100])
                    ax4.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # 5. 交易统计图（左下）
        ax5 = fig.add_subplot(gs[2, 0])
        metrics = ['总收益', '年化收益', '夏普比率', '最大回撤', '胜率']
        values = [
            backtest_result['total_return'],
            backtest_result['annual_return'],
            backtest_result['sharpe_ratio'] * 20,
            -backtest_result['max_drawdown'],
            backtest_result['win_rate']
        ]
        colors = ['#10AC84' if v > 0 else '#EE5A6F' for v in values]
        bars = ax5.barh(metrics, values, color=colors, alpha=0.7)
        
        for i, (bar, value) in enumerate(zip(bars, values)):
            if i == 2:
                label = f"{backtest_result['sharpe_ratio']:.2f}"
            elif i == 3:
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
        
        # 6. ETF轮动频率统计（右下）
        ax6 = fig.add_subplot(gs[2, 1])
        if trades:
            # 统计每个ETF的交易次数
            etf_trade_count = {}
            for trade in trades:
                etf_code = trade.get('etf_code', 'UNKNOWN')
                etf_trade_count[etf_code] = etf_trade_count.get(etf_code, 0) + 1
            
            if etf_trade_count:
                etfs = list(etf_trade_count.keys())
                counts = list(etf_trade_count.values())
                bars = ax6.bar(range(len(etfs)), counts, color='#2E86DE', alpha=0.7)
                ax6.set_xticks(range(len(etfs)))
                ax6.set_xticklabels(etfs, rotation=45, ha='right')
                ax6.set_title('ETF交易次数统计', fontsize=13, fontweight='bold', pad=12)
                ax6.set_xlabel('ETF代码', fontsize=10)
                ax6.set_ylabel('交易次数', fontsize=10)
                ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax6.tick_params(labelsize=9)
                
                # 添加数值标签
                for bar, count in zip(bars, counts):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           str(count), ha='center', va='bottom', fontsize=9)
        
        # 7. 综合信息面板（底部跨列）
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        etf_pnl_summary = backtest_result.get('etf_pnl_summary', [])
        pnl_lines = ""
        if etf_pnl_summary:
            pnl_lines = "\n各ETF总盈亏: " + " | ".join(
                [f"{s['etf_code']} ¥{s['pnl']:,.0f}({s['pnl_pct']:+.1f}%)" for s in etf_pnl_summary[:8]]
            )
            if len(etf_pnl_summary) > 8:
                pnl_lines += " ..."
        
        info_text = f"""
回测周期: {backtest_result.get('trading_days', 0)}日 | 初始: ¥{backtest_result['initial_capital']:,.0f} | 最终: ¥{backtest_result['final_capital']:,.0f}

总收益: {backtest_result['total_return']:+.2f}% | 年化: {backtest_result['annual_return']:+.2f}% | 夏普: {backtest_result['sharpe_ratio']:.2f}

回撤: {backtest_result['max_drawdown']:.2f}% | 交易: {backtest_result['total_trades']}笔 | 胜率: {backtest_result['win_rate']:.2f}%

ETF池: {', '.join(backtest_result.get('etf_codes', []))} | 轮动间隔: {backtest_result.get('rotation_interval', 0)}天 | 持仓数量: Top-{backtest_result.get('top_k', 1)}
{pnl_lines}
        """
        
        ax7.text(0.5, 0.5, info_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8,
                         edgecolor='#DEE2E6', linewidth=2))
        
        plt.suptitle('ETF轮动策略回测分析报告', fontsize=16, fontweight='bold', y=0.98)
        
        # 保存或返回base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"


def get_ai_rotation_suggestion(etf_codes: List[str], etf_scores: Dict[str, float],
                               etf_data_dict: Dict[str, pd.DataFrame], lookback_days: int = 20,
                               top_k: int = 1, min_score_threshold: float = 20.0,
                               technical_recommended: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """
    使用AI分析获取ETF轮动建议（用于回测和模拟盘）
    
    参数:
        etf_codes: ETF代码列表
        etf_scores: ETF技术指标得分字典 {code: score}
        etf_data_dict: ETF数据字典 {code: DataFrame}
        lookback_days: 回看天数
        top_k: 持仓数量
        min_score_threshold: 最低得分阈值
        technical_recommended: 技术指标推荐的ETF列表（可选）
        
    返回:
        (AI推荐的ETF代码列表, 推荐理由字典 {code: reason})
    """
    try:
        from llm import llm
        import re
        
        # 构建ETF数据摘要
        etf_data_summary = []
        for etf_code in etf_codes:
            if etf_code in etf_data_dict:
                df = etf_data_dict[etf_code]
                if len(df) > 0:
                    latest = df.iloc[-1]
                    prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                    change_pct = ((latest['close'] - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    
                    ma5 = df['close'].tail(5).mean() if len(df) >= 5 else latest['close']
                    ma10 = df['close'].tail(10).mean() if len(df) >= 10 else latest['close']
                    ma20 = df['close'].tail(20).mean() if len(df) >= 20 else latest['close']
                    
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 0
                    
                    etf_name = DEFAULT_ETF_LIST.get(etf_code, etf_code)
                    score = etf_scores.get(etf_code, 0)
                    
                    etf_data_summary.append({
                        "code": etf_code,
                        "name": etf_name,
                        "current_price": round(latest['close'], 3),
                        "change_pct": round(change_pct, 2),
                        "ma5": round(ma5, 3),
                        "ma10": round(ma10, 3),
                        "ma20": round(ma20, 3),
                        "volatility": round(volatility, 2),
                        "technical_score": score
                    })
        
        if not etf_data_summary:
            return [], {}
        
        # 计算技术指标推荐的ETF（如果未提供）
        if technical_recommended is None:
            sorted_etfs = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)
            if min_score_threshold > 0:
                technical_recommended = [code for code, score in sorted_etfs[:top_k] if score >= min_score_threshold]
            else:
                technical_recommended = [code for code, score in sorted_etfs[:top_k]]
        
        # 构建ETF数据摘要（按得分排序）
        sorted_etf_data = sorted(etf_data_summary, key=lambda x: etf_scores.get(x['code'], 0), reverse=True)
        etf_data_text = "\n".join([
            f"ETF代码: {etf['code']} ({etf['name']})\n"
            f"  当前价格: {etf['current_price']}, 涨跌幅: {etf['change_pct']}%\n"
            f"  均线: MA5={etf['ma5']}, MA10={etf['ma10']}, MA20={etf['ma20']}\n"
            f"  年化波动率: {etf['volatility']}%\n"
            f"  技术指标得分: {etf['technical_score']}"
            + (" ⭐技术指标推荐" if etf['code'] in technical_recommended else "")
            + "\n"
            for etf in sorted_etf_data
        ])
        
        technical_recommended_str = ", ".join(technical_recommended) if technical_recommended else "无"
        
        llm_prompt = f"""你是一位专业的ETF轮动策略分析师。请基于以下ETF数据和技术指标分析，给出调仓建议。

回看天数: {lookback_days}天
目标持仓数量: 最多{top_k}只（若符合条件的ETF不足{top_k}只，只推荐符合条件的，不要强凑满额）
最低得分阈值: {min_score_threshold}

ETF数据概览（按技术指标得分排序）:
{etf_data_text}

技术指标推荐的ETF: {technical_recommended_str}

重要说明：
1. 技术指标已推荐上述ETF，但你可以基于更深入的分析（如市场趋势、风险水平、ETF之间的相对强弱等）选择不同的ETF
2. 如果你选择与技术指标不同的ETF，必须给出明确的理由
3. 如果你选择与技术指标相同的ETF，也需要说明为什么这些ETF是最佳选择
4. 推荐最多{top_k}只ETF；若技术指标或你的分析认为符合条件的不足{top_k}只（如仅1只达标），则只推荐符合条件的，不必强凑满额
5. 【资产平衡】黄金ETF(518880)为避险资产，与股票类ETF属性不同。不应过度偏好黄金ETF：仅在明确避险需求或市场调整时考虑纳入，且不宜作为首选或常驻持仓；多标的时优先考虑股票类宽基/行业ETF的轮动，保持资产类型的均衡

请以JSON格式返回，格式如下：
{{
    "recommended_etfs": ["510300", "510500"],
    "reasoning": {{
        "510300": "推荐理由：...",
        "510500": "推荐理由：..."
    }},
    "changes_from_technical": {{
        "added": ["510500"],
        "removed": ["159915"],
        "reason": "添加510500是因为...，移除159915是因为..."
    }}
}}

如果推荐与技术指标相同，changes_from_technical可以为空对象{{}}。"""

        # 调用LLM
        llm_response = llm.invoke(llm_prompt)
        llm_content = llm_response.content.strip()
        
        # 解析响应（使用更健壮的JSON解析）
        ai_recommendations = []
        ai_reasoning = {}
        
        # 尝试多种方法提取JSON
        json_str = None
        
        # 方法1: 尝试提取完整的JSON对象（支持嵌套）
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 简单嵌套
            r'```json\s*(\{.*?\})\s*```',  # 代码块中的JSON
            r'```\s*(\{.*?\})\s*```',  # 代码块中的JSON（无json标记）
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, llm_content, re.DOTALL)
            for match in matches:
                try:
                    ai_data = json.loads(match)
                    if "recommended_etfs" in ai_data:
                        json_str = match
                        break
                except:
                    continue
            if json_str:
                break
        
        # 方法2: 如果没找到，尝试提取第一个看起来像JSON的块
        if not json_str:
            json_match = re.search(r'\{[^}]*"recommended_etfs"[^}]*\}', llm_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
        
        # 方法3: 尝试修复常见的JSON格式问题
        if json_str:
            try:
                # 尝试直接解析
                ai_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # 尝试修复常见问题
                try:
                    # 移除尾随逗号
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    # 修复单引号
                    json_str = json_str.replace("'", '"')
                    ai_data = json.loads(json_str)
                except:
                    # 如果还是失败，尝试提取关键信息
                    print(f"[ETF AI轮动] JSON解析失败，尝试从文本中提取: {e}")
                    # 从文本中提取ETF代码
                    etf_codes_in_text = re.findall(r'"(\d{6})"', llm_content)
                    if etf_codes_in_text:
                        ai_recommendations = list(dict.fromkeys(etf_codes_in_text))[:top_k]
                        print(f"[ETF AI轮动] 从文本中提取到ETF: {ai_recommendations}")
                    ai_data = None
            
            if ai_data:
                ai_recommendations = ai_data.get("recommended_etfs", [])
                ai_reasoning = ai_data.get("reasoning", {})
                
                # 记录AI的修改（如果有）
                changes = ai_data.get("changes_from_technical", {})
                if changes:
                    added = changes.get("added", [])
                    removed = changes.get("removed", [])
                    reason = changes.get("reason", "")
                    if added or removed:
                        print(f"[ETF AI轮动] AI修改了技术指标推荐:")
                        if added:
                            print(f"  添加: {added}")
                        if removed:
                            print(f"  移除: {removed}")
                        if reason:
                            print(f"  理由: {reason}")
        
        # 如果没有JSON，尝试从文本中提取
        if not ai_recommendations:
            for etf in etf_data_summary:
                if etf['code'] in llm_content:
                    ai_recommendations.append(etf['code'])
        
        # 过滤：只返回在etf_codes中的ETF，且数量不超过top_k
        ai_recommendations = [code for code in ai_recommendations if code in etf_codes][:top_k]
        
        # 为没有理由的ETF添加默认理由
        for code in ai_recommendations:
            if code not in ai_reasoning:
                ai_reasoning[code] = "基于综合分析的推荐"
        
        return ai_recommendations, ai_reasoning
        
    except Exception as e:
        print(f"[ETF AI轮动] AI分析失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}  # 失败时返回空列表和空字典，回退到技术指标


def get_current_rotation_suggestion(etf_codes: List[str], lookback_days: int = 20,
                                   top_k: int = 1, score_weights: Optional[Dict[str, float]] = None,
                                   min_score_threshold: float = 20.0, use_ai: bool = False) -> Dict[str, Any]:
    """
    获取当前调仓建议（便捷函数）
    
    参数:
        etf_codes: ETF代码列表
        lookback_days: 计算得分时的回看天数
        top_k: 持仓数量（持有得分最高的k只ETF）
        score_weights: 得分权重配置，格式：{"momentum": 0.65, "rsi": 0.1, "ma": 0.15, "macd": 0.1}
        min_score_threshold: 最低得分阈值，如果所有ETF得分都低于此阈值，则建议不持仓（默认20.0）
        
    返回:
        调仓建议字典
    """
    strategy = ETFRotationStrategy(
        etf_codes=etf_codes,
        initial_capital=100000,  # 不需要实际资金，只是用于初始化
        rotation_interval=5,
        commission_rate=0.0003,
        slippage=0.001,
        top_k=top_k,
        score_weights=score_weights,
        min_score_threshold=min_score_threshold
    )
    
    try:
        # 加载最近的数据（只需要足够计算得分的数据）
        data_dict = {}
        request_days = lookback_days + 100  # 多加载一些数据以确保有足够的历史数据
        
        print(f"[ETF轮动] 开始加载 {len(etf_codes)} 只ETF数据用于调仓建议...")
        
        failed_etfs = []  # 记录加载失败的ETF
        for i, code in enumerate(etf_codes):
            print(f"[ETF轮动] ({i+1}/{len(etf_codes)}) 加载 {code}...")
            try:
                stock_data = get_stock_data(code, request_days, use_cache=True)
                if stock_data and len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    df = add_technical_indicators_to_df(df)
                    df = df.dropna().reset_index(drop=True)
                    if len(df) > 0:
                        data_dict[code] = df
                        print(f"[ETF轮动] {code} 加载成功，{len(df)} 条数据")
                    else:
                        failed_etfs.append(f"{code}(数据为空)")
                        print(f"[ETF轮动] {code} 数据为空，跳过")
                else:
                    failed_etfs.append(f"{code}(数据不足，仅{len(stock_data) if stock_data else 0}条)")
                    print(f"[ETF轮动] {code} 数据不足，跳过")
            except Exception as e:
                failed_etfs.append(f"{code}({str(e)})")
                print(f"[ETF轮动] {code} 加载失败: {e}")
        
        if not data_dict:
            error_msg = f"无法加载ETF数据。"
            if failed_etfs:
                error_msg += f" 失败的ETF: {', '.join(failed_etfs)}。"
            error_msg += "请检查：1) ETF代码是否正确（如：510300, 510500, 159915）；2) 网络连接是否正常；3) 数据源是否可用。"
            return {
                "error": error_msg,
                "suggestions": []
            }
        
        # 对齐日期
        aligned_dict = strategy.align_data_dates(data_dict)
        if not aligned_dict:
            return {
                "error": "无法对齐ETF数据日期",
                "suggestions": []
            }
        
        # 获取最新日期
        first_etf = list(aligned_dict.values())[0]
        if len(first_etf) == 0:
            return {
                "error": "没有可用数据",
                "suggestions": []
            }
        
        # 计算所有ETF的得分（使用最新数据）
        etf_scores = {}
        for code, df in aligned_dict.items():
            if len(df) >= lookback_days:
                # 使用最后lookback_days天的数据计算得分
                df_for_score = df.tail(lookback_days).copy()
                score = strategy.calculate_etf_score(df_for_score, lookback_days)
                etf_scores[code] = score
            elif len(df) > 0:
                # 如果数据不足，使用可用数据
                score = strategy.calculate_etf_score(df, len(df))
                etf_scores[code] = score
        
        if not etf_scores:
            return {
                "error": "无法计算ETF得分",
                "suggestions": []
            }
        
        # 按得分排序
        sorted_etfs = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择Top-K：取前 top_k 名；仅当这 top_k 名全部低于阈值时才空仓（与回测一致）
        top_k_codes = [code for code, score in sorted_etfs[:top_k]]
        top_k_scores = [score for code, score in sorted_etfs[:top_k]]
        if min_score_threshold > 0:
            if top_k_scores and any(s >= min_score_threshold for s in top_k_scores):
                top_k_etfs = top_k_codes
            else:
                top_k_etfs = []
        else:
            top_k_etfs = top_k_codes
        
        # 构建建议列表
        suggestions = []
        for rank, (code, score) in enumerate(sorted_etfs, 1):
            is_recommended = code in top_k_etfs
            suggestions.append({
                "rank": rank,
                "etf_code": code,
                "score": round(score, 2),
                "recommended": is_recommended,
                "action": "持有" if is_recommended else "不持有"
            })
        
        # 获取最新日期
        latest_date = first_etf['date'].iloc[-1]
        
        return {
            "date": str(latest_date),
            "lookback_days": lookback_days,
            "top_k": top_k,
            "min_score_threshold": min_score_threshold,
            "suggestions": suggestions,
            "recommended_etfs": top_k_etfs,
            "etf_scores": {code: round(score, 2) for code, score in etf_scores.items()},
            "all_below_threshold": min_score_threshold > 0 and len(top_k_etfs) == 0
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "suggestions": []
        }


def backtest_etf_rotation(etf_codes: List[str], initial_capital: float = 100000,
                         days: int = 252, rotation_interval: int = 5,
                         lookback_days: int = 20, commission_rate: float = 0.0003,
                         slippage: float = 0.001, top_k: int = 1,
                         score_weights: Optional[Dict[str, float]] = None,
                         rebalance_interval: Optional[int] = None,
                         min_score_threshold: float = 20.0,
                         use_ai: bool = False,
                         position_strategy: str = "equal",
                         progress_callback: Optional[Callable[[str, dict], None]] = None,
                         abort_check: Optional[Callable[[], bool]] = None,
                         risk_control_mode: str = 'off') -> Dict[str, Any]:
    """
    ETF轮动策略回测（便捷函数）
    
    参数:
        etf_codes: ETF代码列表
        initial_capital: 初始资金
        days: 回测天数
        rotation_interval: 轮动间隔（交易日）
        lookback_days: 计算得分时的回看天数
        commission_rate: 手续费率
        slippage: 滑点
        top_k: 持仓数量（持有得分最高的k只ETF，默认1只）
        score_weights: 得分权重配置，格式：{"momentum": 0.65, "rsi": 0.1, "ma": 0.15, "macd": 0.1}
        rebalance_interval: 再平衡间隔（交易日），None表示无再平衡
        min_score_threshold: 最低得分阈值，如果所有ETF得分都低于此阈值，则清仓不持仓（默认20.0）
        position_strategy: 仓位策略。"equal"=等权重；"kelly"=按凯利公式分配
        progress_callback: 可选，进度回调 (event_type, data)
        abort_check: 可选，无参可调用，返回 True 时停止回测
        risk_control_mode: 风控模式。'off'=回测不启用；'warn'=模拟盘记录警告但不阻断；'block'=阻断
        
    返回:
        回测结果字典（包含图表）
    """
    strategy = ETFRotationStrategy(
        etf_codes=etf_codes,
        initial_capital=initial_capital,
        rotation_interval=rotation_interval,
        commission_rate=commission_rate,
        slippage=slippage,
        top_k=top_k,
        score_weights=score_weights,
        rebalance_interval=rebalance_interval,
        min_score_threshold=min_score_threshold,
        use_ai=use_ai,
        position_strategy=position_strategy
    )
    
    # 执行回测
    result = strategy.backtest(days=days, lookback_days=lookback_days,
                               progress_callback=progress_callback,
                               abort_check=abort_check,
                               risk_control_mode=risk_control_mode)
    
    # 添加top_k到结果中
    result['top_k'] = top_k
    
    # 生成图表
    if 'error' not in result:
        try:
            chart_base64 = strategy.generate_backtest_charts(result)
            # 如果返回的已经是完整的data:image格式，直接使用；否则只返回base64部分
            if chart_base64 and chart_base64.startswith('data:image'):
                result['chart'] = chart_base64
            elif chart_base64:
                result['chart'] = chart_base64
            else:
                print("[ETF轮动] 警告：图表生成失败")
        except Exception as e:
            print(f"[ETF轮动] 图表生成错误: {e}")
            import traceback
            traceback.print_exc()
    
    return result


# 子模块：ETF 模拟账户
from .etf_sim_account import get_account_manager, ETFSimAccount, ETFSimAccountManager
