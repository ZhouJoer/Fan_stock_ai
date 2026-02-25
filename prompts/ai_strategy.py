"""
AI 策略决策：单标的信号、组合决策、交易策略封装
依赖 prompts.strategy_prompts 与 tools.strategy_config。
"""
from __future__ import annotations

import json
import re

from prompts.strategy_prompts import (
    STRATEGY_PROMPTS,
    _filter_by_high_win_rate_rules,
    build_portfolio_decision_prompt,
    build_llm_signal_prompt,
)


def llm_portfolio_decision(
    stocks_data: list,
    positions: dict,
    cash: float,
    total_value: float,
    strategy_type: str = 'adaptive',
    risk_preference: str = 'balanced',
    max_positions: int = 5,
    high_win_rate_mode: bool = False,
    market_context: dict = None,
    pool_mode: str = None,
    stock_industry_map: dict = None,
    llm=None
) -> dict:
    """使用 LLM 对选股池进行综合决策。"""
    if llm is None:
        from llm import llm as default_llm
        llm = default_llm
    if high_win_rate_mode:
        stocks_data = _filter_by_high_win_rate_rules(stocks_data, positions)
    prompt = build_portfolio_decision_prompt(
        stocks_data, positions, cash, total_value,
        strategy_type, risk_preference, max_positions,
        high_win_rate_mode=high_win_rate_mode,
        market_context=market_context,
        pool_mode=pool_mode,
        stock_industry_map=stock_industry_map,
    )
    try:
        response = llm.invoke(prompt)
        content = response.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            decisions_dict = {}
            for d in result.get('decisions', []):
                code = d.get('code', '')
                if code:
                    decisions_dict[code] = {
                        'action': d.get('action', 'HOLD').upper(),
                        'confidence': float(d.get('confidence', 0.5)),
                        'reason': d.get('reason', '')
                    }
            return {
                'analysis': result.get('analysis', ''),
                'decisions': decisions_dict,
                'priority_buy': result.get('priority_buy', []),
                'priority_sell': result.get('priority_sell', [])
            }
        return {
            'analysis': 'LLM响应解析失败',
            'decisions': {s['code']: {'action': 'HOLD', 'confidence': 0.5, 'reason': '解析失败'} for s in stocks_data},
            'priority_buy': [],
            'priority_sell': []
        }
    except Exception as e:
        print(f"[LLM组合决策] 失败: {e}")
        return {
            'analysis': f'LLM调用异常: {str(e)[:50]}',
            'decisions': {s['code']: {'action': 'HOLD', 'confidence': 0.5, 'reason': '异常'} for s in stocks_data},
            'priority_buy': [],
            'priority_sell': []
        }


def llm_generate_signal(
    row_or_indicators: dict,
    strategy_type: str = 'trend',
    risk_preference: str = 'balanced',
    has_position: bool = False,
    entry_price: float = None,
    holding_days: int = 0,
    highest_price: float = None,
    llm=None
) -> tuple:
    """使用 LLM 生成交易信号（可用于回测和实时决策）。返回 (action, confidence, reasoning)。"""
    if llm is None:
        from llm import llm as default_llm
        llm = default_llm
    if hasattr(row_or_indicators, 'to_dict'):
        indicators = row_or_indicators.to_dict()
    else:
        indicators = dict(row_or_indicators)
    if 'Current_Price' not in indicators and 'close' in indicators:
        indicators['Current_Price'] = indicators['close']
    prompt = build_llm_signal_prompt(
        indicators, strategy_type, risk_preference,
        has_position, entry_price, holding_days, highest_price
    )
    try:
        response = llm.invoke(prompt)
        content = response.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            decision = json.loads(json_match.group())
            action = decision.get('action', 'HOLD').upper()
            confidence = float(decision.get('confidence', 0.5))
            reasoning = decision.get('reasoning', 'LLM决策')
            if action not in ['BUY', 'SELL', 'HOLD']:
                action = 'HOLD'
            confidence = max(0.0, min(1.0, confidence))
            return action, confidence, reasoning
        return 'HOLD', 0.5, 'LLM响应解析失败，保守观望'
    except Exception as e:
        print(f"[LLM信号] 生成失败: {e}")
        return 'HOLD', 0.5, f'LLM调用异常: {str(e)[:50]}'


def ai_trading_strategy(
    stock_code,
    indicators,
    historical_trades=None,
    strategy_type='trend',
    risk_preference='balanced',
    llm=None
):
    """AI 综合判断交易策略：基于技术指标判断开仓和平仓时机。返回 (决策字典, 分析文本)。"""
    from modules.strategy_config import get_strategy_config
    if llm is None:
        from llm import llm as default_llm
        llm = default_llm
    has_position = False
    entry_price = None
    holding_days = 0
    if historical_trades and len(historical_trades) > 0:
        last_trade = historical_trades[-1]
        if last_trade.get('status') == 'open':
            has_position = True
            entry_price = last_trade.get('entry_price', 0)
            holding_days = last_trade.get('days', 0)
    action, confidence, reasoning = llm_generate_signal(
        indicators, strategy_type, risk_preference,
        has_position, entry_price, holding_days,
        llm=llm
    )
    config = get_strategy_config(strategy_type, risk_preference)
    current_price = indicators.get('Current_Price', indicators.get('close', 0))
    decision = {
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "entry_price": current_price if action == 'BUY' else None,
        "stop_loss": current_price * (1 - config['stop_loss_pct']) if action == 'BUY' else None,
        "take_profit": current_price * (1 + config['take_profit_pct']) if action == 'BUY' else None,
        "position_size": config['position_size']
    }
    analysis = f"""
【LLM交易决策分析】
策略类型：{STRATEGY_PROMPTS.get(strategy_type, STRATEGY_PROMPTS['trend'])['name']}
风险偏好：{risk_preference}

决策：{action}
信心度：{confidence*100:.1f}%

决策理由：
{reasoning}

风险管理建议：
- 止损线：{config['stop_loss_pct']*100:.1f}%
- 止盈线：{config['take_profit_pct']*100:.1f}%
- 建议仓位：{config['position_size']*100:.0f}%
"""
    return decision, analysis
