"""
每日AI决策：re-export 自 modules.strategy_config，保持向后兼容。
业务实现已迁至 modules/strategy_config/daily_decision.py。
"""
from modules.strategy_config import ai_daily_decision

__all__ = ["ai_daily_decision"]
