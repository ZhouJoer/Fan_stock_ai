"""
ETF 轮动：re-export 自 modules.etf_rotation，保持向后兼容。
业务实现已迁至 modules/etf_rotation/。
"""
from modules.etf_rotation import *  # noqa: F401, F403
