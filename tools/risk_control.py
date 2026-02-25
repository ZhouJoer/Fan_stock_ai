"""
风控模块：根据持仓、资金与近期盈亏进行审计

用于选股池与 ETF 轮动的回测、模拟盘，在每日/每次交易前进行风控审计，
可触发：允许交易、减仓、暂停开新仓、停止交易（清仓或不再买入）。

【如何恢复轮动】
每日/每次调仓前都会重新做一次风控审计，不满足条件即自动恢复，无需人工操作：

1. 回撤超限 (reduce_position)
   - 恢复条件：当前权益相对历史峰值的回撤重新 < 15%
   - 即：市场反弹、净值回升到峰值的 85% 以上后，下一日自动允许轮动

2. 连续亏损 (pause_trading)
   - 恢复条件：出现至少 1 个正收益日，连续亏损天数被打破
   - 即：任意一天收益为正后，下一日自动允许轮动

3. 单日大跌 (pause_trading)
   - 恢复条件：审计用的是「最近一日」收益，自然滚动
   - 即：大跌日的次日，若当日收益不是大跌，即自动允许轮动

4. 累计亏损超限 (stop)
   - 恢复条件（二选一）：
     a) 当前总权益回升到初始资金的 80% 以上；或
     b) 已止损、仓位很轻：持仓市值占总权益 ≤ stop_allow_rotate_when_position_pct（默认 15%）
   - 止损后多为现金时，若不满足 (b) 会一直无法恢复；因此当持仓占比 ≤ 15% 时自动允许恢复轮动
   - 若希望更快恢复，可调大 cumulative_loss_pct_limit（如 0.25）或传入自定义 config
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RiskAuditInput:
    """风控审计输入"""
    cash: float
    total_value: float          # 总权益 = cash + 持仓市值
    initial_capital: float
    positions_value: float     # 持仓市值合计
    positions: Dict[str, Any]  # 各标的持仓，value 需能算市值或直接为 market_value
    recent_daily_returns: List[float]  # 近期每日收益率（小数，如 0.01 表示 1%），最近一天在末尾
    peak_value: Optional[float] = None  # 历史峰值权益（用于回撤）
    current_date: Optional[str] = None


@dataclass
class RiskAuditResult:
    """风控审计结果"""
    pass_audit: bool           # 是否通过（True=允许正常交易）
    action: str                # 'allow' | 'reduce_position' | 'pause_trading' | 'stop'
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# 默认风控参数（可由调用方覆盖）
DEFAULT_RISK_CONFIG = {
    'max_drawdown_pct': 0.15,           # 相对峰值最大回撤 15%，超过则减仓/暂停
    'max_consecutive_loss_days': 5,     # 连续亏损天数上限，超过则暂停开新仓
    'max_single_position_pct': 0.45,   # 单标的最大仓位占比 45%
    'daily_loss_pct_limit': 0.05,       # 单日亏损超 5% 则下一日暂停开新仓
    'cumulative_loss_pct_limit': 0.20,  # 相对初始资金累计亏损超 20% 则停止开新仓
    'stop_allow_rotate_when_position_pct': 0.15,  # 累计亏损触发 stop 时，若持仓占比≤此值（如已止损、多为现金），允许恢复轮动
}


def audit_risk(input_data: RiskAuditInput, config: Optional[Dict[str, Any]] = None) -> RiskAuditResult:
    """
    根据持仓、资金与近期盈亏进行风控审计。

    规则（可配置）：
    1. 回撤：相对历史峰值回撤超过 max_drawdown_pct -> reduce_position 或 pause_trading
    2. 连续亏损：近期连续亏损天数超过 max_consecutive_loss_days -> pause_trading
    3. 单日大跌：最近一日亏损超过 daily_loss_pct_limit -> 下一日 pause_trading（由调用方在下一日传入时体现）
    4. 累计亏损：相对初始资金累计亏损超过 cumulative_loss_pct_limit -> stop 或 pause_trading
    5. 单标的集中度：任一标的仓位超过 max_single_position_pct -> reduce_position（提示）

    返回 RiskAuditResult，调用方根据 action 决定是否允许开新仓、是否强制减仓等。

    恢复：每次调仓/每日都会重新审计；当触发条件不再满足时（回撤缩小、出现正收益日、
    单日大跌滚出窗口、或累计亏损收窄），pass_audit 自动为 True，轮动即恢复，无需额外逻辑。
    """
    cfg = {**DEFAULT_RISK_CONFIG, **(config or {})}
    details: Dict[str, Any] = {}
    total = input_data.total_value
    initial = input_data.initial_capital
    positions_value = input_data.positions_value
    returns = input_data.recent_daily_returns or []
    peak = input_data.peak_value if input_data.peak_value is not None else total

    # 1. 累计亏损
    cumulative_loss_pct = (initial - total) / initial if initial > 0 else 0
    details['cumulative_loss_pct'] = round(cumulative_loss_pct * 100, 2)
    if cumulative_loss_pct >= cfg['cumulative_loss_pct_limit']:
        # 止损后多为现金时允许恢复：持仓占比≤阈值则视为已止损，允许重新开仓/轮动，否则永远无法恢复
        position_pct = (positions_value / total) if total > 0 else 0
        allow_rotate_threshold = cfg.get('stop_allow_rotate_when_position_pct', 0.15)
        if position_pct <= allow_rotate_threshold:
            details['stop_allow_reason'] = f'持仓占比{position_pct*100:.1f}%≤{allow_rotate_threshold*100:.0f}%，视为已止损，允许恢复轮动'
            return RiskAuditResult(
                pass_audit=True,
                action='allow',
                reason=f"累计亏损{cumulative_loss_pct*100:.1f}%超限，但持仓占比{position_pct*100:.1f}%较低，允许恢复轮动",
                details=details,
            )
        return RiskAuditResult(
            pass_audit=False,
            action='stop',
            reason=f"风控：累计亏损 {cumulative_loss_pct*100:.1f}% 超过限制 {cfg['cumulative_loss_pct_limit']*100:.0f}%，停止开新仓",
            details=details,
        )

    # 2. 回撤（相对峰值）
    drawdown = (peak - total) / peak if peak > 0 else 0
    details['drawdown_pct'] = round(drawdown * 100, 2)
    if drawdown >= cfg['max_drawdown_pct']:
        return RiskAuditResult(
            pass_audit=False,
            action='reduce_position',
            reason=f"风控：回撤 {drawdown*100:.1f}% 超过限制 {cfg['max_drawdown_pct']*100:.0f}%，建议减仓/暂停开新仓",
            details=details,
        )

    # 3. 连续亏损天数
    consecutive_loss_days = 0
    for r in reversed(returns):
        if r < 0:
            consecutive_loss_days += 1
        else:
            break
    details['consecutive_loss_days'] = consecutive_loss_days
    if consecutive_loss_days >= cfg['max_consecutive_loss_days']:
        return RiskAuditResult(
            pass_audit=False,
            action='pause_trading',
            reason=f"风控：连续 {consecutive_loss_days} 日亏损，暂停开新仓",
            details=details,
        )

    # 4. 单日大跌（最近一日）
    if returns is not None and len(returns) > 0:
        last_return = returns[-1] if not hasattr(returns, "iloc") else float(returns.iloc[-1])
        details['last_day_return_pct'] = round(last_return * 100, 2)
        if last_return <= -cfg['daily_loss_pct_limit']:
            return RiskAuditResult(
                pass_audit=False,
                action='pause_trading',
                reason=f"风控：单日亏损 {last_return*100:.1f}% 超过限制 {cfg['daily_loss_pct_limit']*100:.0f}%，当日暂停开新仓",
                details=details,
            )

    # 5. 单标的集中度（仅当有持仓且可算占比时）
    if total > 0 and isinstance(input_data.positions, dict) and input_data.positions:
        for code, pos in input_data.positions.items():
            if hasattr(pos, 'get'):
                mv = pos.get('market_value') or pos.get('shares', 0) * pos.get('price', 0) or pos.get('value', 0)
            else:
                mv = getattr(pos, 'market_value', None) or getattr(pos, 'value', 0)
            if isinstance(mv, (int, float)) and mv > 0:
                pct = mv / total
                if pct > cfg['max_single_position_pct']:
                    details['over_weight_code'] = code
                    details['over_weight_pct'] = round(pct * 100, 1)
                    return RiskAuditResult(
                        pass_audit=False,
                        action='reduce_position',
                        reason=f"风控：{code} 仓位 {pct*100:.1f}% 超过限制 {cfg['max_single_position_pct']*100:.0f}%",
                        details=details,
                    )

    details['drawdown_pct'] = details.get('drawdown_pct', round(drawdown * 100, 2))
    return RiskAuditResult(
        pass_audit=True,
        action='allow',
        reason='风控通过',
        details=details,
    )


def get_recent_daily_returns_from_equity(equity_curve: List[float], lookback: int = 10) -> List[float]:
    """
    从权益曲线计算近期每日收益率（小数）。
    equity_curve[i] 为第 i 日结束时的权益，返回最近 lookback 天的日收益率。
    """
    if not equity_curve or len(equity_curve) < 2 or lookback < 1:
        return []
    returns = []
    for i in range(max(1, len(equity_curve) - lookback), len(equity_curve)):
        prev = equity_curve[i - 1]
        curr = equity_curve[i]
        if prev and prev > 0:
            returns.append((curr - prev) / prev)
        else:
            returns.append(0.0)
    return returns[-lookback:] if len(returns) > lookback else returns
