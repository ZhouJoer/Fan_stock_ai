"""
ETF 模拟账户：re-export 自 modules.etf_rotation.etf_sim_account，保持向后兼容。
业务实现已迁至 modules/etf_rotation/etf_sim_account.py。
"""
from modules.etf_rotation.etf_sim_account import (
    ETFSimAccount,
    ETFSimAccountManager,
    get_account_manager,
)

__all__ = ["ETFSimAccount", "ETFSimAccountManager", "get_account_manager"]
