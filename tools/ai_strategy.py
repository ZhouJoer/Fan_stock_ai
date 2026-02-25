"""
AI 策略决策：re-export 自 prompts 模块，保持向后兼容。
业务实现已迁至 prompts/。
"""
from prompts import (
    llm_generate_signal,
    llm_portfolio_decision,
    ai_trading_strategy,
    STRATEGY_PROMPTS,
)

__all__ = [
    "llm_generate_signal",
    "llm_portfolio_decision",
    "ai_trading_strategy",
    "STRATEGY_PROMPTS",
]
