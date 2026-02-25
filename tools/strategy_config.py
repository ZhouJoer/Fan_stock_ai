"""
策略配置：re-export 自 modules.strategy_config，保持向后兼容。
业务实现已迁至 modules/strategy_config/。
"""
from modules.strategy_config import *  # noqa: F401, F403
