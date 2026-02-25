"""
选股池模拟账户：re-export 自 modules.stock_pool.sim_account，保持向后兼容。
业务实现已迁至 modules/stock_pool/sim_account.py。
"""
from modules.stock_pool.sim_account import (
    StockPoolSimAccount,
    StockPoolSimAccountManager,
    get_account_manager,
)

__all__ = ["StockPoolSimAccount", "StockPoolSimAccountManager", "get_account_manager"]
