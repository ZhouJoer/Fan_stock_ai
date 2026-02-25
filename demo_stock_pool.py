"""
选股池功能演示脚本
===================================
演示如何使用选股池管理功能：
1. 创建选股池
2. AI自动生成交易信号
3. 根据信号自动配置仓位
4. 回测策略表现
5. 保存/加载配置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.stock_pool import (
    create_stock_pool,
    backtest_stock_pool,
    get_pool_signals,
    get_pool_allocation,
    generate_pool_report,
    save_pool_config,
    load_pool_config,
    list_saved_pools,
    StockPoolManager,
    PoolBacktestEngine,
)


def demo_basic_usage():
    """基础使用演示"""
    print("\n" + "="*60)
    print("  演示1: 基础选股池创建与信号生成")
    print("="*60)
    
    # 定义股票池（可根据需要修改）
    stocks = [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "000333",  # 美的集团
        "601318",  # 中国平安
    ]
    
    # 创建选股池配置
    config = create_stock_pool(
        stocks=stocks,
        initial_capital=1000000,  # 100万初始资金
        strategy_type="adaptive",  # AI自适应策略
        risk_preference="balanced",  # 均衡风险偏好
        allocation_method="signal_strength",  # 按信号强度分配仓位
    )
    
    print(f"\n选股池创建成功!")
    print(f"  名称: {config.name}")
    print(f"  股票: {config.stocks}")
    print(f"  初始资金: ¥{config.initial_capital:,.0f}")
    print(f"  策略类型: {config.strategy_type}")
    print(f"  风险偏好: {config.risk_preference}")
    
    return config


def demo_signal_generation(stocks):
    """信号生成演示"""
    print("\n" + "="*60)
    print("  演示2: AI自动生成交易信号")
    print("="*60)
    
    # 获取每只股票的AI信号
    signals = get_pool_signals(stocks, risk_preference="balanced")
    
    print("\n【AI交易信号】")
    for code, signal in signals.items():
        print(f"\n{code}:")
        print(f"  信号类型: {signal.get('signal_type', 'N/A')}")
        print(f"  信号强度: {signal.get('signal_strength', 0):.4f}")
        print(f"  强度等级: {signal.get('strength_level', 'N/A')}")
        print(f"  市场状态: {signal.get('market_state', 'N/A')}")
        
        risk = signal.get('risk_assessment', {})
        if risk:
            print(f"  风险评估:")
            print(f"    - 最大回撤: {risk.get('最大回撤预测', 'N/A')}")
            print(f"    - 波动率: {risk.get('波动率等级', 'N/A')}")
    
    return signals


def demo_position_allocation(stocks, total_capital=1000000):
    """仓位配置演示"""
    print("\n" + "="*60)
    print("  演示3: 根据信号自动配置仓位")
    print("="*60)
    
    # 获取配置建议
    allocation = get_pool_allocation(
        stocks=stocks,
        total_capital=total_capital,
        allocation_method="signal_strength",
        risk_preference="balanced"
    )
    
    print(f"\n【仓位配置建议】(总资金: ¥{total_capital:,.0f})")
    
    total_weight = 0
    total_investment = 0
    
    for code, alloc in allocation.items():
        weight = alloc.get('target_weight', 0)
        value = alloc.get('target_value', 0)
        shares = alloc.get('suggested_shares', 0)
        
        total_weight += weight
        total_investment += value
        
        print(f"\n{code}:")
        print(f"  信号: {alloc.get('signal_type', 'N/A')} (强度: {alloc.get('signal_strength', 0):.2f})")
        print(f"  目标仓位: {weight:.1f}%")
        print(f"  建议金额: ¥{value:,.0f}")
        print(f"  建议股数: {shares}股")
        if alloc.get('reason'):
            print(f"  备注: {alloc.get('reason')}")
    
    print(f"\n【汇总】")
    print(f"  总仓位: {total_weight:.1f}%")
    print(f"  总投资: ¥{total_investment:,.0f}")
    print(f"  现金保留: ¥{total_capital - total_investment:,.0f}")
    
    return allocation


def demo_backtest(stocks, days=90):
    """回测演示"""
    print("\n" + "="*60)
    print(f"  演示4: 回测策略表现 (过去{days}天)")
    print("="*60)
    
    # 运行回测
    result = backtest_stock_pool(
        stocks=stocks,
        initial_capital=1000000,
        days=days,
        strategy_type="adaptive",
        risk_preference="balanced"
    )
    
    if result.get('status') == 'success':
        stats = result.get('statistics', {})
        
        print("\n【回测结果】")
        print(f"  回测天数: {stats.get('total_days', 0)}天")
        print(f"  交易次数: {stats.get('total_trades', 0)}次")
        print(f"  初始资金: ¥{stats.get('initial_value', 0):,.0f}")
        print(f"  最终资金: ¥{stats.get('final_value', 0):,.0f}")
        print(f"  总收益率: {stats.get('total_return', 0):.2f}%")
        print(f"  年化收益: {stats.get('annualized_return', 0):.2f}%")
        print(f"  最大回撤: {stats.get('max_drawdown', 0):.2f}%")
        print(f"  夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"  胜率: {stats.get('win_rate', 0):.1f}%")
        
        # 保存回测图表
        chart_base64 = result.get('chart_base64')
        if chart_base64:
            print(f"\n  图表已生成 (Base64长度: {len(chart_base64)})")
            
            # 解码并保存图片
            import base64
            img_data = base64.b64decode(chart_base64)
            with open('backtest_result.png', 'wb') as f:
                f.write(img_data)
            print(f"  图表已保存至: backtest_result.png")
    else:
        print(f"\n回测失败: {result.get('message', 'Unknown error')}")
    
    return result


def demo_save_load_config(config):
    """配置保存/加载演示"""
    print("\n" + "="*60)
    print("  演示5: 保存和加载选股池配置")
    print("="*60)
    
    # 保存配置
    filepath = save_pool_config(config)
    
    # 列出所有保存的配置
    print("\n【已保存的选股池】")
    pools = list_saved_pools()
    for pool in pools:
        print(f"  - {pool['name']}: {pool['stocks']} (¥{pool['initial_capital']:,.0f})")
    
    # 重新加载配置
    print("\n【重新加载配置】")
    loaded_config = load_pool_config(filepath)
    print(f"  名称: {loaded_config.name}")
    print(f"  股票: {loaded_config.stocks}")
    
    return loaded_config


def demo_full_report(stocks):
    """完整报告演示"""
    print("\n" + "="*60)
    print("  演示6: 生成AI审计友好的完整报告")
    print("="*60)
    
    report = generate_pool_report(
        stocks=stocks,
        total_capital=1000000,
        risk_preference="balanced"
    )
    
    return report


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("  选股池管理系统演示")
    print("#"*60)
    
    # 定义要分析的股票
    stocks = [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "000333",  # 美的集团
        "601318",  # 中国平安
    ]
    
    # 运行各个演示
    print("\n选择要运行的演示:")
    print("  1. 基础选股池创建")
    print("  2. AI交易信号生成")
    print("  3. 仓位自动配置")
    print("  4. 策略回测")
    print("  5. 配置保存/加载")
    print("  6. 完整分析报告")
    print("  7. 运行所有演示")
    print("  0. 退出")
    
    try:
        choice = input("\n请输入选项 (默认7): ").strip() or "7"
        choice = int(choice)
    except:
        choice = 7
    
    config = None
    
    if choice == 1 or choice == 7:
        config = demo_basic_usage()
    
    if choice == 2 or choice == 7:
        demo_signal_generation(stocks)
    
    if choice == 3 or choice == 7:
        demo_position_allocation(stocks)
    
    if choice == 4 or choice == 7:
        demo_backtest(stocks, days=60)
    
    if choice == 5 or choice == 7:
        if config is None:
            config = demo_basic_usage()
        demo_save_load_config(config)
    
    if choice == 6 or choice == 7:
        demo_full_report(stocks)
    
    print("\n" + "#"*60)
    print("  演示完成!")
    print("#"*60)


if __name__ == "__main__":
    main()
