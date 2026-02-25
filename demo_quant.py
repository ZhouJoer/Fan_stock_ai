"""
量化交易模块快速演示
展示QMT量化工具的核心功能
"""
from tools.qmt_tools import get_qmt_tools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def demo_technical_analysis():
    """演示技术分析功能"""
    print("=" * 80)
    print("1. 技术分析演示")
    print("=" * 80)
    
    # 创建模拟股票数据
    print("\n生成模拟股票数据（100个交易日）...")
    dates = pd.date_range(end=datetime.now(), periods=100)
    np.random.seed(42)
    
    # 模拟一个有趋势的价格序列
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 3
    prices = trend + noise
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100)) * 1.5,
        'low': prices - np.abs(np.random.randn(100)) * 1.5,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    df.set_index('date', inplace=True)
    
    print(f"✓ 数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ 价格范围: ¥{df['close'].min():.2f} - ¥{df['close'].max():.2f}")
    
    # 计算技术指标
    print("\n计算技术指标...")
    qmt = get_qmt_tools()
    df = qmt.calculate_technical_indicators(df)
    
    print("✓ 已计算以下技术指标:")
    print("  - 移动平均线: MA5, MA10, MA20, MA60")
    print("  - RSI (相对强弱指标)")
    print("  - MACD (指数平滑异同移动平均线)")
    print("  - 布林带 (Bollinger Bands)")
    print("  - ATR (平均真实波幅)")
    
    # 显示最新数据
    print("\n最新技术指标值:")
    latest = df.iloc[-1]
    print(f"  当前价格: ¥{latest['close']:.2f}")
    print(f"  MA5:  ¥{latest['MA5']:.2f}")
    print(f"  MA20: ¥{latest['MA20']:.2f}")
    print(f"  RSI:  {latest['RSI']:.2f}")
    print(f"  MACD: {latest['MACD']:.4f}")
    print(f"  布林带上轨: ¥{latest['BB_Upper']:.2f}")
    print(f"  布林带下轨: ¥{latest['BB_Lower']:.2f}")
    
    return df


def demo_trading_strategies(df):
    """演示交易策略"""
    print("\n" + "=" * 80)
    print("2. 交易策略演示")
    print("=" * 80)
    
    qmt = get_qmt_tools()
    strategies = {
        "dual_ma": "双均线策略（MA5/MA20交叉）",
        "macd": "MACD金叉死叉策略",
        "rsi_bb": "RSI+布林带超买超卖策略",
        "multi_factor": "多因子综合策略"
    }
    
    results = {}
    df_signals_dict = {}  # 保存带信号的数据
    
    for strategy_id, strategy_name in strategies.items():
        print(f"\n测试策略: {strategy_name}")
        print("-" * 80)
        
        # 生成信号
        df_signals = qmt.generate_trading_signals(df.copy(), strategy_id)
        df_signals_dict[strategy_id] = df_signals  # 保存
        buy_signals = (df_signals['signal'] == 1).sum()
        sell_signals = (df_signals['signal'] == -1).sum()
        
        print(f"  买入信号: {buy_signals}次")
        print(f"  卖出信号: {sell_signals}次")
        
        # 回测
        backtest_result = qmt.backtest(df_signals, initial_capital=100000)
        results[strategy_id] = backtest_result
        
        print(f"  总收益率: {backtest_result['total_return']:.2f}%")
        print(f"  夏普比率: {backtest_result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {backtest_result['max_drawdown']:.2f}%")
        print(f"  胜率: {backtest_result['win_rate']:.2f}%")
    
    return results, df_signals_dict


def demo_strategy_comparison(results):
    """演示策略对比"""
    print("\n" + "=" * 80)
    print("3. 策略对比分析")
    print("=" * 80)
    
    # 创建对比表格
    print("\n策略表现排名:")
    print("-" * 80)
    print(f"{'策略':<20} {'收益率':<12} {'夏普':<10} {'回撤':<12} {'胜率':<10}")
    print("-" * 80)
    
    strategy_names = {
        "dual_ma": "双均线",
        "macd": "MACD",
        "rsi_bb": "RSI+布林带",
        "multi_factor": "多因子综合"
    }
    
    # 按收益率排序
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['total_return'], 
                           reverse=True)
    
    for strategy_id, result in sorted_results:
        name = strategy_names[strategy_id]
        print(f"{name:<20} {result['total_return']:>10.2f}%  "
              f"{result['sharpe_ratio']:>8.2f}  "
              f"{result['max_drawdown']:>10.2f}%  "
              f"{result['win_rate']:>8.2f}%")
    
    # 推荐策略
    best_strategy = sorted_results[0]
    print("\n" + "=" * 80)
    print("📊 推荐策略")
    print("=" * 80)
    print(f"\n基于回测结果，推荐使用: {strategy_names[best_strategy[0]]}")
    print(f"  ✓ 总收益率: {best_strategy[1]['total_return']:.2f}%")
    print(f"  ✓ 年化收益率: {best_strategy[1]['annual_return']:.2f}%")
    print(f"  ✓ 夏普比率: {best_strategy[1]['sharpe_ratio']:.2f} ", end="")
    
    sharpe = best_strategy[1]['sharpe_ratio']
    if sharpe > 2:
        print("(优秀)")
    elif sharpe > 1:
        print("(良好)")
    else:
        print("(一般)")
    
    print(f"  ✓ 最大回撤: {best_strategy[1]['max_drawdown']:.2f}% ", end="")
    
    drawdown = best_strategy[1]['max_drawdown']
    if drawdown < 10:
        print("(风险低)")
    elif drawdown < 20:
        print("(风险中等)")
    else:
        print("(风险较高)")
    
    print(f"  ✓ 胜率: {best_strategy[1]['win_rate']:.2f}%")
    print(f"  ✓ 交易次数: {best_strategy[1]['total_trades']}笔")
    
    return best_strategy[0]  # 返回最佳策略ID


def demo_risk_assessment(results):
    """演示风险评估"""
    print("\n" + "=" * 80)
    print("4. AI风险评估（模拟）")
    print("=" * 80)
    
    # 选择最佳策略
    best = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    strategy_id, result = best
    
    print("\n评估目标: 多因子综合策略")
    print("-" * 80)
    
    # 收益质量评估
    print("\n✓ 收益质量评估:")
    if result['total_return'] > 50:
        print("  ⚠️ 收益率过高，可能存在过拟合风险")
    elif result['total_return'] > 0:
        print("  ✓ 收益率合理")
    else:
        print("  ✗ 策略亏损，不建议使用")
    
    # 风险控制评估
    print("\n✓ 风险控制评估:")
    if result['max_drawdown'] > 20:
        print("  ✗ 最大回撤过大（>20%），风险较高")
    elif result['max_drawdown'] > 10:
        print("  ⚠️ 最大回撤中等（10-20%），需要注意风险")
    else:
        print("  ✓ 最大回撤较小（<10%），风险可控")
    
    if result['sharpe_ratio'] > 2:
        print("  ✓ 夏普比率优秀（>2），风险调整后收益好")
    elif result['sharpe_ratio'] > 1:
        print("  ✓ 夏普比率良好（>1），风险调整后收益可接受")
    else:
        print("  ⚠️ 夏普比率一般（<1），风险收益比不理想")
    
    # 交易统计评估
    print("\n✓ 交易统计评估:")
    if result['total_trades'] < 5:
        print("  ⚠️ 交易次数过少，统计意义不足")
    elif result['total_trades'] > 50:
        print("  ⚠️ 交易过于频繁，可能导致高额手续费")
    else:
        print(f"  ✓ 交易次数合理（{result['total_trades']}笔）")
    
    if result['win_rate'] > 60:
        print(f"  ✓ 胜率优秀（{result['win_rate']:.1f}%）")
    elif result['win_rate'] > 50:
        print(f"  ✓ 胜率良好（{result['win_rate']:.1f}%）")
    else:
        print(f"  ⚠️ 胜率偏低（{result['win_rate']:.1f}%）")
    
    # 最终审批
    print("\n" + "=" * 80)
    print("📋 AI审计结论")
    print("=" * 80)
    
    score = 0
    if result['total_return'] > 0 and result['total_return'] < 50:
        score += 1
    if result['max_drawdown'] < 20:
        score += 1
    if result['sharpe_ratio'] > 1:
        score += 1
    if 5 <= result['total_trades'] <= 50:
        score += 1
    if result['win_rate'] > 50:
        score += 1
    
    if score >= 4:
        print("\n✅ 审批通过")
        print("该策略表现优秀，风险可控，建议执行。")
    elif score >= 3:
        print("\n⚠️ 有条件通过")
        print("该策略整体可行，但存在一些风险点，建议小仓位试验。")
    else:
        print("\n❌ 审批拒绝")
        print("该策略存在重大问题或风险过高，不建议执行。")
    
    print("\n⚠️ 重要提示:")
    print("  - 历史回测不代表未来收益")
    print("  - 实盘交易存在滑点和手续费")
    print("  - 建议从小资金开始测试")
    print("  - 持续监控策略表现")


def demo_backtest_charts(best_strategy_id, df_signals_dict, results):
    """演示回测图表生成"""
    print("\n" + "=" * 80)
    print("5. 回测图表生成")
    print("=" * 80)
    
    qmt = get_qmt_tools()
    
    strategy_names = {
        "dual_ma": "双均线",
        "macd": "MACD",
        "rsi_bb": "RSI+布林带",
        "multi_factor": "多因子综合"
    }
    
    print(f"\n为最佳策略生成回测图表: {strategy_names[best_strategy_id]}")
    print("-" * 80)
    
    try:
        # 获取策略数据
        df_signals = df_signals_dict[best_strategy_id]
        backtest_result = results[best_strategy_id]
        
        # 生成图表（保存为文件）
        chart_path = f"backtest_chart_{best_strategy_id}.png"
        save_path = qmt.generate_backtest_charts(df_signals, backtest_result, save_path=chart_path)
        
        print(f"✓ 图表已生成: {save_path}")
        print("\n图表包含以下内容:")
        print("  1. 资金曲线图 - 展示账户资金变化")
        print("  2. 累计收益率图 - 展示收益率走势")
        print("  3. 回撤曲线图 - 展示风险控制情况")
        print("  4. 价格与交易信号图 - 展示买卖点位")
        print("  5. 关键指标柱状图 - 展示核心指标")
        print("  6. 交易盈亏分布图 - 展示每笔交易表现")
        print("  7. 综合信息面板 - 展示完整回测数据")
        
        print(f"\n提示: 请使用图片查看器打开 '{chart_path}' 查看图表")
        
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║             QMT量化交易模块 - 核心功能演示                     ║
╚═══════════════════════════════════════════════════════════════╝

本演示将展示：
1. 技术指标计算（MA、RSI、MACD、布林带、ATR）
2. 多种交易策略（双均线、MACD、RSI+布林带、多因子）
3. 策略回测与对比分析
4. AI风险评估与审批
5. 回测图表生成

注意：本演示使用模拟数据，实际使用需连接QMT平台
    """)
    
    input("按Enter开始演示...")
    
    # 1. 技术分析
    df = demo_technical_analysis()
    
    # 2. 交易策略
    results, df_signals_dict = demo_trading_strategies(df)
    
    # 3. 策略对比
    best_strategy_id = demo_strategy_comparison(results)
    
    # 4. 风险评估
    demo_risk_assessment(results)
    
    # 5. 图表生成
    demo_backtest_charts(best_strategy_id, df_signals_dict, results)
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n要使用完整的工作流，请运行:")
    print("  python test_quant.py")
    print("\n或通过Web界面请求量化分析:")
    print('  "请对贵州茅台进行量化分析，回测一下多因子策略"')
    print("\n详细文档请查看: QUANT_README.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n演示中断")
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()
