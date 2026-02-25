"""
选股池：re-export 自 modules.stock_pool，保持向后兼容。
业务实现已迁至 modules/stock_pool/。
"""
from modules.stock_pool import *  # noqa: F401, F403
