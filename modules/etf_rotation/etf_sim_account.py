"""
ETF轮动模拟盘账户管理模块
支持模拟盘账户的创建、查询、交易记录等功能
"""
import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path


class ETFSimAccount:
    """ETF轮动模拟盘账户类"""
    
    def __init__(self, account_id: str, initial_capital: float = 100000):
        """
        初始化模拟盘账户
        
        参数:
            account_id: 账户ID（唯一标识）
            initial_capital: 初始资金
        """
        self.account_id = account_id
        self.initial_capital = initial_capital
        self.cash = initial_capital  # 可用资金
        self.positions = {}  # 持仓：{etf_code: {'shares': int, 'entry_price': float, 'entry_date': str}}
        self.trades = []  # 交易记录
        self.etf_pool = []  # ETF池：账户关联的ETF代码列表
        self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self.last_rotation_date = None  # 上次轮动日期（ISO格式字符串）
        self.last_rebalance_date = None  # 上次再平衡日期（ISO格式字符串）
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        计算总资产（现金 + 持仓市值）
        
        参数:
            current_prices: 当前价格字典 {etf_code: price}
            
        返回:
            总资产
        """
        total = self.cash
        for etf_code, pos in self.positions.items():
            if etf_code in current_prices:
                total += pos['shares'] * current_prices[etf_code]
        return total
    
    def get_positions_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算持仓市值
        
        参数:
            current_prices: 当前价格字典
            
        返回:
            持仓市值
        """
        total = 0
        for etf_code, pos in self.positions.items():
            if etf_code in current_prices:
                total += pos['shares'] * current_prices[etf_code]
        return total
    
    def buy(self, etf_code: str, shares: int, price: float, 
            commission_rate: float = 0.0003, slippage: float = 0.001,
            reason: str = "") -> Dict[str, Any]:
        """
        买入ETF
        
        参数:
            etf_code: ETF代码
            shares: 买入股数
            price: 买入价格
            commission_rate: 手续费率
            slippage: 滑点
            reason: 买入原因
            
        返回:
            交易结果
        """
        if shares <= 0:
            return {"success": False, "error": "买入股数必须大于0"}
        
        # 计算实际买入价格（含滑点）
        actual_price = price * (1 + slippage)
        cost = shares * actual_price * (1 + commission_rate)
        
        if cost > self.cash:
            return {"success": False, "error": f"资金不足，需要{cost:.2f}元，可用资金{self.cash:.2f}元"}
        
        # 执行买入
        self.cash -= cost
        
        # 更新持仓
        if etf_code in self.positions:
            # 已有持仓，计算平均成本
            old_pos = self.positions[etf_code]
            total_shares = old_pos['shares'] + shares
            total_cost = old_pos['shares'] * old_pos['entry_price'] + shares * actual_price
            avg_price = total_cost / total_shares
            self.positions[etf_code] = {
                'shares': total_shares,
                'entry_price': avg_price,
                'entry_date': old_pos['entry_date']  # 保持首次买入日期
            }
        else:
            # 新建仓
            self.positions[etf_code] = {
                'shares': shares,
                'entry_price': actual_price,
                'entry_date': datetime.now().date().isoformat()
            }
        
        # 记录交易
        trade = {
            'date': datetime.now().isoformat(),
            'type': 'buy',
            'etf_code': etf_code,
            'shares': shares,
            'price': actual_price,
            'cost': cost,
            'reason': reason
        }
        self.trades.append(trade)
        self.last_updated = datetime.now().isoformat()
        
        return {
            "success": True,
            "trade": trade,
            "cash_after": self.cash,
            "positions": self.positions.copy()
        }
    
    def sell(self, etf_code: str, shares: int, price: float,
             commission_rate: float = 0.0003, slippage: float = 0.001,
             reason: str = "") -> Dict[str, Any]:
        """
        卖出ETF
        
        参数:
            etf_code: ETF代码
            shares: 卖出股数（如果为0或None，则全部卖出）
            price: 卖出价格
            commission_rate: 手续费率
            slippage: 滑点
            reason: 卖出原因
            
        返回:
            交易结果
        """
        if etf_code not in self.positions:
            return {"success": False, "error": f"未持有{etf_code}"}
        
        current_shares = self.positions[etf_code]['shares']
        
        # 如果shares为0或None，全部卖出
        if shares is None or shares == 0 or shares >= current_shares:
            shares = current_shares
        
        if shares > current_shares:
            return {"success": False, "error": f"持仓不足，当前持仓{current_shares}股，尝试卖出{shares}股"}
        
        # 计算实际卖出价格（含滑点）
        actual_price = price * (1 - slippage)
        revenue = shares * actual_price * (1 - commission_rate)
        
        # 计算盈亏
        entry_price = self.positions[etf_code]['entry_price']
        profit_loss = (actual_price - entry_price) * shares
        profit_loss_pct = (actual_price / entry_price - 1) * 100
        
        # 执行卖出
        self.cash += revenue
        
        # 更新持仓
        if shares == current_shares:
            # 全部卖出，删除持仓
            del self.positions[etf_code]
        else:
            # 部分卖出
            self.positions[etf_code]['shares'] -= shares
        
        # 记录交易
        trade = {
            'date': datetime.now().isoformat(),
            'type': 'sell',
            'etf_code': etf_code,
            'shares': shares,
            'price': actual_price,
            'revenue': revenue,
            'entry_price': entry_price,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'reason': reason
        }
        self.trades.append(trade)
        self.last_updated = datetime.now().isoformat()
        
        return {
            "success": True,
            "trade": trade,
            "cash_after": self.cash,
            "positions": self.positions.copy()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "account_id": self.account_id,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "trades": self.trades,
            "etf_pool": self.etf_pool,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "last_rotation_date": self.last_rotation_date,
            "last_rebalance_date": self.last_rebalance_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ETFSimAccount':
        """从字典创建账户"""
        account = cls(data['account_id'], data['initial_capital'])
        account.cash = data['cash']
        account.positions = data.get('positions', {})
        account.trades = data.get('trades', [])
        account.etf_pool = data.get('etf_pool', [])
        account.created_at = data.get('created_at', datetime.now().isoformat())
        account.last_updated = data.get('last_updated', datetime.now().isoformat())
        account.last_rotation_date = data.get('last_rotation_date')
        account.last_rebalance_date = data.get('last_rebalance_date')
        return account


class ETFSimAccountManager:
    """ETF模拟盘账户管理器"""
    
    def __init__(self, data_dir: str = "data/etf_sim_accounts"):
        """
        初始化账户管理器
        
        参数:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.accounts: Dict[str, ETFSimAccount] = {}
        self._load_all_accounts()
    
    def _get_account_file(self, account_id: str) -> Path:
        """获取账户文件路径"""
        return self.data_dir / f"{account_id}.json"
    
    def _load_all_accounts(self):
        """加载所有账户"""
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    account = ETFSimAccount.from_dict(data)
                    self.accounts[account.account_id] = account
            except Exception as e:
                print(f"加载账户文件{file_path}失败: {e}")
    
    def create_account(self, account_id: str, initial_capital: float = 100000) -> ETFSimAccount:
        """
        创建新账户
        
        参数:
            account_id: 账户ID
            initial_capital: 初始资金
            
        返回:
            账户对象
        """
        if account_id in self.accounts:
            raise ValueError(f"账户{account_id}已存在")
        
        account = ETFSimAccount(account_id, initial_capital)
        self.accounts[account_id] = account
        self._save_account(account)
        return account
    
    def get_account(self, account_id: str) -> Optional[ETFSimAccount]:
        """获取账户"""
        if account_id not in self.accounts:
            # 尝试从文件加载
            account_file = self._get_account_file(account_id)
            if account_file.exists():
                try:
                    with open(account_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        account = ETFSimAccount.from_dict(data)
                        self.accounts[account_id] = account
                        return account
                except Exception as e:
                    print(f"加载账户{account_id}失败: {e}")
        return self.accounts.get(account_id)
    
    def _save_account(self, account: ETFSimAccount):
        """保存账户到文件"""
        account_file = self._get_account_file(account.account_id)
        try:
            with open(account_file, 'w', encoding='utf-8') as f:
                json.dump(account.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存账户{account.account_id}失败: {e}")
    
    def update_account(self, account: ETFSimAccount):
        """更新账户"""
        self.accounts[account.account_id] = account
        self._save_account(account)
    
    def list_accounts(self) -> List[str]:
        """列出所有账户ID"""
        return list(self.accounts.keys())
    
    def delete_account(self, account_id: str) -> bool:
        """删除账户"""
        if account_id not in self.accounts:
            return False
        
        del self.accounts[account_id]
        account_file = self._get_account_file(account_id)
        if account_file.exists():
            account_file.unlink()
        return True


# 全局账户管理器实例
_account_manager = None


def get_account_manager() -> ETFSimAccountManager:
    """获取账户管理器实例"""
    global _account_manager
    if _account_manager is None:
        _account_manager = ETFSimAccountManager()
    return _account_manager
