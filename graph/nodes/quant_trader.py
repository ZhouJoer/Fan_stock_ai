"""
量化交易节点
负责执行量化策略，包括数据处理、信号生成、回测和交易执行

注意：此模块已拆分为多个子模块，主要功能已移至：
- tools/stock_data.py: 股票数据获取
- tools/technical_indicators.py: 技术指标计算
- tools/strategy_config.py: 策略配置
- tools/ai_strategy.py: AI交易策略
- tools/backtest.py: 回测功能
"""
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from graph.state import AgentState
from llm import llm
from tools.qmt_tools import get_qmt_tools

# 从拆分后的模块导入（保持向后兼容）
from tools.stock_data import get_stock_data, get_stock_code_from_name
from tools.technical_indicators import calculate_technical_indicators, add_technical_indicators_to_df
from tools.strategy_config import (
    STRATEGY_TYPES, RISK_PREFERENCES,
    get_strategy_config, analyze_market_state, get_adaptive_strategy_config
)
from tools.ai_strategy import ai_trading_strategy
from tools.backtest import ai_decision_backtest, generate_ai_backtest_chart


# 为向后兼容重新导出这些函数/类
__all__ = [
    # stock_data
    'get_stock_data', 'get_stock_code_from_name',
    # technical_indicators
    'calculate_technical_indicators', 'add_technical_indicators_to_df',
    # strategy_config
    'STRATEGY_TYPES', 'RISK_PREFERENCES', 'get_strategy_config',
    'analyze_market_state', 'get_adaptive_strategy_config',
    # ai_strategy
    'ai_trading_strategy',
    # backtest
    'ai_decision_backtest', 'generate_ai_backtest_chart',
    # node
    'quant_trader_node',
]


def _normalize_stock_code(raw: str) -> str:
    """将 6 位数字或带后缀的代码规范为 6位.SH / 6位.SZ。"""
    raw = str(raw).strip()
    # 已是 600000.SH / 000001.SZ 形式
    if re.match(r"^\d{6}\.(?:SH|SZ)$", raw, re.I):
        return raw.upper()
    # 纯 6 位数字：6 开头上交所，0/3 开头深交所
    m = re.match(r"^(\d{6})$", raw)
    if m:
        code = m.group(1)
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"
    return raw


def quant_trader_node(state: AgentState):
    """
    量化交易节点
    
    功能：
    1. 获取股票历史数据
    2. 计算技术指标
    3. AI综合判断开仓/平仓时机
    4. 生成交易决策报告
    5. （可选）执行回测分析
    """
    messages = state["messages"]
    
    # 获取QMT工具
    qmt = get_qmt_tools()
    
    # 从历史消息中提取股票代码（支持 6位.SH/SZ 或纯 6 位数字）
    stock_code = None
    for msg in reversed(messages):
        content = msg.content
        if not isinstance(content, str):
            continue
        # 先匹配带后缀
        pattern_suffix = r'\d{6}\.(?:SH|SZ)'
        matches = re.findall(pattern_suffix, content)
        if matches:
            stock_code = _normalize_stock_code(matches[0])
            break
        # 再匹配纯 6 位数字（如「量化分析600875」或「代码：600875」）
        if "股票代码" in content or "代码：" in content or "量化" in content or re.search(r"\d{6}", content):
            plain = re.findall(r"\b(\d{6})\b", content)
            if plain:
                stock_code = _normalize_stock_code(plain[0])
                break
    
    if not stock_code:
        user_request = messages[0].content if len(messages) > 0 else ""
        extraction_prompt = f"""
        从以下用户请求中提取股票代码（6位数字，如 600519、000858）：
        "{user_request}"
        
        如果找到股票代码，只返回6位数字本身（如 600519 或 600519.SH），不要其他内容。
        如果没有找到，返回"未找到"。
        """
        extraction_response = llm.invoke(extraction_prompt)
        extracted = extraction_response.content.strip()
        if "未找到" not in extracted:
            # 支持 LLM 返回 600519 或 600519.SH
            code_part = re.sub(r"[^\d.]", "", extracted)
            if re.match(r"^\d{6}(\.(?:SH|SZ))?$", code_part, re.I):
                stock_code = _normalize_stock_code(code_part.split(".")[0])
            elif "." in extracted:
                stock_code = _normalize_stock_code(extracted)
        if not stock_code:
            stock_code = "600000.SH"
    
    # 获取真实股票数据
    try:
        stock_data = get_stock_data(stock_code, days=120, use_cache=True)
    except Exception as e:
        error_msg = f"获取股票 {stock_code} 数据失败: {e}"
        return {
            "messages": messages + [AIMessage(content=error_msg, name="Quant_Trader")]
        }

    try:
        # 计算技术指标
        indicators = calculate_technical_indicators(stock_data)
        # 避免 NaN 导致报告格式化报错
        import math
        for k, v in indicators.items():
            if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                indicators[k] = 0.0 if k != "RSI" else 50.0
            elif isinstance(v, (int, float)) and k == "RSI" and (v < 0 or v > 100):
                indicators[k] = max(0.0, min(100.0, float(v)))
    except Exception as e:
        error_msg = f"计算技术指标失败（{stock_code}）: {e}"
        return {
            "messages": messages + [AIMessage(content=error_msg, name="Quant_Trader")]
        }

    # 历史交易记录
    historical_trades = []
    if "trades" in state:
        historical_trades = state["trades"]

    try:
        # AI 综合判断交易决策
        decision, ai_analysis = ai_trading_strategy(stock_code, indicators, historical_trades, llm=llm)
    except Exception as e:
        error_msg = f"AI 交易决策失败（{stock_code}）: {e}"
        return {
            "messages": messages + [AIMessage(content=error_msg, name="Quant_Trader")]
        }

    # 构建决策报告（防御性取值，避免 KeyError 或类型异常）
    action = decision.get("action") or "HOLD"
    confidence = float(decision.get("confidence", 0.5))
    action_text = {
        "BUY": "🟢 **买入开仓**",
        "SELL": "🔴 **卖出平仓**",
        "HOLD": "🟡 **持有观望**"
    }.get(action, "🟡 **持有观望**")
    confidence_level = "高" if confidence >= 0.8 else "中" if confidence >= 0.6 else "低"
    current_price = indicators.get("Current_Price") or 0.0
    try:
        bb_ratio = ((current_price - indicators["BB_Lower"]) / (indicators["BB_Upper"] - indicators["BB_Lower"]) * 100) if (indicators.get("BB_Upper") or 0) != (indicators.get("BB_Lower") or 0) else 50.0
    except Exception:
        bb_ratio = 50.0

    quant_report = f"""
=== 【量化交易节点】Quant_Trader - AI交易决策 ===
股票代码：{stock_code}
分析时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
策略类型：AI综合技术指标判断策略

📊 **实时技术指标**

**价格信息**：
- 当前价格：¥{current_price:.2f}

**均线系统**：
- MA5: ¥{indicators.get('MA5', 0):.2f}
- MA10: ¥{indicators.get('MA10', 0):.2f}
- MA20: ¥{indicators.get('MA20', 0):.2f}
- MA60: ¥{indicators.get('MA60', 0):.2f}
- 趋势: {'多头排列 📈' if (indicators.get('MA5') or 0) > (indicators.get('MA10') or 0) > (indicators.get('MA20') or 0) else '空头排列 📉' if (indicators.get('MA5') or 0) < (indicators.get('MA10') or 0) < (indicators.get('MA20') or 0) else '震荡整理 ↔️'}

**动量指标**：
- RSI(14): {indicators.get('RSI', 50):.2f} {'(超买)' if (indicators.get('RSI') or 0) > 70 else '(超卖)' if (indicators.get('RSI') or 0) < 30 else '(正常)'}
- MACD: {indicators.get('MACD', 0):.4f}
- MACD信号: {indicators.get('MACD_Signal', 0):.4f}
- MACD柱: {indicators.get('MACD_Hist', 0):.4f} {'✅ 金叉' if (indicators.get('MACD_Hist') or 0) > 0 else '❌ 死叉'}

**布林带**：
- 上轨: ¥{indicators.get('BB_Upper', 0):.2f}
- 中轨: ¥{indicators.get('BB_Middle', 0):.2f}
- 下轨: ¥{indicators.get('BB_Lower', 0):.2f}
- 当前位置: {bb_ratio:.1f}%

**波动率**：
- ATR(14): {indicators.get('ATR', 0):.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 **AI交易决策**

{action_text}
信心度：{confidence:.1%} ({confidence_level})

**决策理由**：
{decision.get('reasoning', '')}

**风险管理参数**：
- 建议止损价：¥{(decision.get('stop_loss') or current_price * 0.95):.2f}
- 建议止盈价：¥{(decision.get('take_profit') or current_price * 1.05):.2f}
- 建议仓位：{(decision.get('position_size') or 0.3):.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **AI详细分析**

{ai_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **风险提示**：
1. AI决策基于技术指标，不构成投资建议
2. 实际交易需考虑市场流动性、滑点等因素
3. 建议结合基本面分析和市场环境
4. 严格执行止损止盈，控制风险敞口
5. 市场有风险，投资需谨慎
"""
    
    return {
        "messages": messages + [AIMessage(content=quant_report, name="Quant_Trader")],
        "trading_decision": decision,
        "technical_indicators": indicators
    }
