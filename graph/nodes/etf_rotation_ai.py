"""
ETF AI轮动节点
使用LLM工作流来分析和决策ETF轮动策略
"""
import json
import re
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from graph.state import AgentState
from llm import llm
from tools.etf_rotation import get_current_rotation_suggestion, DEFAULT_ETF_LIST
from tools.stock_data import get_stock_data
import pandas as pd


def etf_rotation_ai_node(state: AgentState):
    """
    ETF AI轮动节点
    
    功能：
    1. 从用户请求中提取ETF代码列表和参数
    2. 获取ETF数据和技术指标
    3. 使用LLM分析ETF表现和市场情况
    4. 生成AI调仓建议
    5. 返回详细的调仓决策报告
    """
    messages = state.get("messages", [])
    # 确保messages是列表类型
    if not isinstance(messages, list):
        messages = list(messages) if messages else []
    
    # 从历史消息中提取用户请求
    user_request = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            break
    
    # 提取ETF代码列表
    etf_codes = []
    
    # 首先尝试从 "ETF代码: xxx,xxx,xxx" 格式提取
    etf_code_match = re.search(r'ETF代码[:\s]+([0-9,\s]+)', user_request)
    if etf_code_match:
        etf_codes_str = etf_code_match.group(1)
        # 提取所有6位数字代码
        etf_codes = re.findall(r'\d{6}', etf_codes_str)
        print(f"[ETF AI轮动] 从'ETF代码'字段提取到 {len(etf_codes)} 只ETF: {etf_codes}")
    
    # 如果没有找到，尝试其他格式
    if not etf_codes:
        # 尝试从JSON数组格式提取: ["510300", "510500"]
        json_array_match = re.search(r'\["([^"]+)"(?:,\s*"([^"]+)")*\]', user_request)
        if json_array_match:
            etf_codes = [m for match in re.findall(r'"(\d{6})"', user_request) for m in [match] if m]
            print(f"[ETF AI轮动] 从JSON数组格式提取到 {len(etf_codes)} 只ETF: {etf_codes}")
    
    # 如果还是没有找到，尝试提取所有6位数字（可能是逗号或空格分隔）
    if not etf_codes:
        # 查找所有6位数字
        all_codes = re.findall(r'\b\d{6}\b', user_request)
        # 过滤掉明显不是ETF代码的数字（如日期、金额等）
        etf_codes = [code for code in all_codes if len(code) == 6 and code.isdigit()]
        if etf_codes:
            print(f"[ETF AI轮动] 从文本中提取到 {len(etf_codes)} 只ETF: {etf_codes}")
    
    # 如果还是没有找到，尝试从JSON格式提取
    if not etf_codes:
        try:
            if "etf_codes" in user_request or "etfCodes" in user_request:
                json_match = re.search(r'\{[^}]+\}', user_request, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    etf_codes = data.get("etf_codes") or data.get("etfCodes") or []
                    if etf_codes:
                        print(f"[ETF AI轮动] 从JSON对象提取到 {len(etf_codes)} 只ETF: {etf_codes}")
        except Exception as e:
            print(f"[ETF AI轮动] JSON解析失败: {e}")
    
    # 如果还是没有找到，使用默认ETF列表
    if not etf_codes:
        etf_codes = list(DEFAULT_ETF_LIST.keys())[:5]  # 默认使用前5个
        print(f"[ETF AI轮动] 使用默认ETF列表: {etf_codes}")
    
    # 去重并确保都是字符串
    etf_codes = list(dict.fromkeys([str(code).strip() for code in etf_codes if code]))
    print(f"[ETF AI轮动] 最终提取到 {len(etf_codes)} 只ETF: {etf_codes}")
    
    # 提取其他参数
    lookback_days = 20
    top_k = 1
    min_score_threshold = 20.0
    
    # 从请求中提取参数
    if "lookback_days" in user_request or "回看天数" in user_request:
        match = re.search(r'(?:lookback_days|回看天数)[:\s=]+(\d+)', user_request)
        if match:
            lookback_days = int(match.group(1))
    
    if "top_k" in user_request or "持仓数量" in user_request:
        match = re.search(r'(?:top_k|持仓数量)[:\s=]+(\d+)', user_request)
        if match:
            top_k = int(match.group(1))
    
    if "min_score_threshold" in user_request or "最低得分" in user_request:
        match = re.search(r'(?:min_score_threshold|最低得分)[:\s=]+([\d.]+)', user_request)
        if match:
            min_score_threshold = float(match.group(1))
    
    try:
        # 1. 获取传统技术指标建议
        print(f"[ETF AI轮动] 开始分析 {len(etf_codes)} 只ETF...")
        try:
            traditional_suggestion = get_current_rotation_suggestion(
                etf_codes=etf_codes,
                lookback_days=lookback_days,
                top_k=top_k,
                min_score_threshold=min_score_threshold
            )
        except Exception as e:
            error_msg = f"获取ETF调仓建议失败: {str(e)}"
            print(f"[ETF AI轮动] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "messages": list(messages) + [AIMessage(content=error_msg, name="ETF_Rotation_AI")]
            }
        
        if not isinstance(traditional_suggestion, dict):
            error_msg = f"获取ETF调仓建议返回格式错误: {type(traditional_suggestion)}"
            return {
                "messages": list(messages) + [AIMessage(content=error_msg, name="ETF_Rotation_AI")]
            }
        
        if traditional_suggestion.get("error"):
            error_msg = f"获取ETF数据失败: {traditional_suggestion['error']}"
            return {
                "messages": list(messages) + [AIMessage(content=error_msg, name="ETF_Rotation_AI")]
            }
        
        # 2. 获取ETF的详细数据用于LLM分析
        etf_data_summary = []
        failed_etfs = []
        for etf_code in etf_codes:
            try:
                print(f"[ETF AI轮动] 正在获取 {etf_code} 的详细数据...")
                stock_data = get_stock_data(etf_code, days=60, use_cache=True)
                if stock_data and len(stock_data) > 0:
                    df = pd.DataFrame(stock_data)
                    latest = df.iloc[-1]
                    prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                    change_pct = ((latest['close'] - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    
                    # 计算短期趋势（5日、10日、20日均线）
                    ma5 = df['close'].tail(5).mean() if len(df) >= 5 else latest['close']
                    ma10 = df['close'].tail(10).mean() if len(df) >= 10 else latest['close']
                    ma20 = df['close'].tail(20).mean() if len(df) >= 20 else latest['close']
                    
                    # 计算波动率
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5) * 100  # 年化波动率
                    
                    etf_name = DEFAULT_ETF_LIST.get(etf_code, etf_code)
                    score = traditional_suggestion.get("etf_scores", {}).get(etf_code, 0)
                    is_recommended = etf_code in traditional_suggestion.get("recommended_etfs", [])
                    
                    etf_data_summary.append({
                        "code": etf_code,
                        "name": etf_name,
                        "current_price": round(latest['close'], 3),
                        "change_pct": round(change_pct, 2),
                        "ma5": round(ma5, 3),
                        "ma10": round(ma10, 3),
                        "ma20": round(ma20, 3),
                        "volatility": round(volatility, 2),
                        "technical_score": score,
                        "is_recommended_by_technical": is_recommended
                    })
                    print(f"[ETF AI轮动] {etf_code} 数据获取成功")
                else:
                    failed_etfs.append(f"{etf_code}(数据为空)")
                    print(f"[ETF AI轮动] {etf_code} 数据为空")
            except Exception as e:
                failed_etfs.append(f"{etf_code}({str(e)})")
                print(f"[ETF AI轮动] 获取{etf_code}数据失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 记录失败的ETF
        if failed_etfs:
            print(f"[ETF AI轮动] 以下ETF数据获取失败: {', '.join(failed_etfs)}")
        
        print(f"[ETF AI轮动] 成功获取 {len(etf_data_summary)}/{len(etf_codes)} 只ETF的数据")
        
        if not etf_data_summary:
            error_msg = "无法获取ETF数据进行分析"
            return {
                "messages": list(messages) + [AIMessage(content=error_msg, name="ETF_Rotation_AI")]
            }
        
        # 3. 构建LLM分析提示
        etf_data_text = "\n".join([
            f"ETF代码: {etf['code']} ({etf['name']})\n"
            f"  当前价格: {etf['current_price']}, 涨跌幅: {etf['change_pct']}%\n"
            f"  均线: MA5={etf['ma5']}, MA10={etf['ma10']}, MA20={etf['ma20']}\n"
            f"  年化波动率: {etf['volatility']}%\n"
            f"  技术指标得分: {etf['technical_score']}\n"
            f"  技术指标建议: {'推荐持有' if etf['is_recommended_by_technical'] else '不推荐'}\n"
            for etf in etf_data_summary
        ])
        
        current_date = traditional_suggestion.get("date", str(datetime.now().date()))
        
        llm_prompt = f"""你是一位专业的ETF轮动策略分析师。请基于以下ETF数据和技术指标分析，给出AI调仓建议。

当前日期: {current_date}
回看天数: {lookback_days}天
目标持仓数量: 最多{top_k}只（若符合条件的ETF不足{top_k}只，只推荐符合条件的，不要强凑满额）
最低得分阈值: {min_score_threshold}

ETF数据概览:
{etf_data_text}

请综合考虑以下因素：
1. 技术指标得分（动量、RSI、均线、MACD）
2. 价格趋势和均线排列
3. 波动率（风险水平）
4. 市场整体情况
5. ETF之间的相对强弱
6. 【资产平衡】黄金ETF(518880)为避险资产，不宜过度偏好：仅在明确避险需求时考虑，多标的时优先轮动股票类ETF，保持资产类型均衡

请给出：
1. 推荐的ETF代码列表（最多{top_k}只；若符合条件的不足{top_k}只则只推荐符合条件的，不必强凑）
2. 推荐理由（每只ETF的详细分析）
3. 风险提示
4. 调仓建议（买入/卖出/持有）

请以JSON格式返回，格式如下：
{{
    "recommended_etfs": ["510300", "510500"],
    "reasoning": {{
        "510300": "推荐理由...",
        "510500": "推荐理由..."
    }},
    "risk_warning": "风险提示...",
    "trading_advice": "调仓建议..."
}}"""

        # 4. 调用LLM进行分析
        print("[ETF AI轮动] 调用LLM进行AI分析...")
        llm_response = llm.invoke(llm_prompt)
        llm_content = llm_response.content.strip()
        
        # 5. 解析LLM响应
        ai_recommendations = set()  # 初始化为集合，不是字典
        ai_reasoning = {}
        risk_warning = ""
        trading_advice = ""
        
        # 尝试提取JSON
        json_match = re.search(r'\{[^}]+\}', llm_content, re.DOTALL)
        if json_match:
            try:
                ai_data = json.loads(json_match.group())
                ai_recommendations = set(ai_data.get("recommended_etfs", []))
                ai_reasoning = ai_data.get("reasoning", {})
                risk_warning = ai_data.get("risk_warning", "")
                trading_advice = ai_data.get("trading_advice", "")
            except Exception as e:
                print(f"[ETF AI轮动] JSON解析失败: {e}")
                pass
        
        # 如果没有JSON或解析失败，尝试从文本中提取ETF代码
        if not ai_recommendations:
            for etf in etf_data_summary:
                if etf['code'] in llm_content:
                    ai_recommendations.add(etf['code'])
        
        # 6. 对比传统技术指标和AI建议
        technical_recommended = set(traditional_suggestion.get("recommended_etfs", []))
        ai_recommended = ai_recommendations
        
        # 7. 生成最终报告（纯文本格式，不使用Markdown）
        report_lines = [
            f"═══════════════════════════════════════════════════════",
            f"ETF AI轮动分析报告",
            f"═══════════════════════════════════════════════════════",
            f"",
            f"分析日期: {current_date}",
            f"分析ETF数量: {len(etf_codes)}只",
            f"回看天数: {lookback_days}天",
            f"目标持仓数量: {top_k}只",
            f"",
            f"───────────────────────────────────────────────────────",
            f"一、ETF数据概览",
            f"───────────────────────────────────────────────────────",
            f""
        ]
        
        for etf in etf_data_summary:
            report_lines.append(f"{etf['code']} ({etf['name']})")
            report_lines.append(f"  当前价格: {etf['current_price']}")
            report_lines.append(f"  涨跌幅: {etf['change_pct']}%")
            report_lines.append(f"  均线: MA5={etf['ma5']}, MA10={etf['ma10']}, MA20={etf['ma20']}")
            report_lines.append(f"  年化波动率: {etf['volatility']}%")
            report_lines.append(f"  技术指标得分: {etf['technical_score']}")
            report_lines.append("")
        
        report_lines.extend([
            f"───────────────────────────────────────────────────────",
            f"二、技术指标建议",
            f"───────────────────────────────────────────────────────",
            f""
        ])
        
        for suggestion in traditional_suggestion.get("suggestions", []):
            etf_code = suggestion.get("etf_code")
            score = suggestion.get("score", 0)
            action = suggestion.get("action", "")
            report_lines.append(f"  {etf_code}: 得分 {score}, {action}")
        
        report_lines.extend([
            f"",
            f"技术指标推荐ETF: {', '.join(technical_recommended) if technical_recommended else '无'}",
            f"",
            f"───────────────────────────────────────────────────────",
            f"三、AI分析建议",
            f"───────────────────────────────────────────────────────",
            f""
        ])
        
        if ai_recommended:
            report_lines.append(f"AI推荐ETF: {', '.join(ai_recommended)}")
            report_lines.append("")
            for etf_code in ai_recommended:
                reason = ai_reasoning.get(etf_code, "基于综合分析的推荐")
                report_lines.append(f"  {etf_code}: {reason}")
        else:
            report_lines.append("AI未给出明确的ETF推荐")
        
        report_lines.extend([
            f"",
            f"───────────────────────────────────────────────────────",
            f"四、对比分析",
            f"───────────────────────────────────────────────────────",
            f""
        ])
        
        both_recommend = technical_recommended & ai_recommended
        only_technical = technical_recommended - ai_recommended
        only_ai = ai_recommended - technical_recommended
        
        if both_recommend:
            report_lines.append(f"共同推荐: {', '.join(both_recommend)} (技术指标和AI都推荐)")
        if only_technical:
            report_lines.append(f"仅技术指标推荐: {', '.join(only_technical)} (技术指标推荐但AI未推荐)")
        if only_ai:
            report_lines.append(f"仅AI推荐: {', '.join(only_ai)} (AI推荐但技术指标未推荐)")
        
        report_lines.extend([
            f"",
            f"───────────────────────────────────────────────────────",
            f"五、最终调仓建议",
            f"───────────────────────────────────────────────────────",
            f""
        ])
        
        # 最终建议：优先使用AI推荐，如果没有则使用技术指标推荐
        final_recommended = ai_recommended if ai_recommended else technical_recommended
        
        if final_recommended:
            report_lines.append(f"建议持有: {', '.join(final_recommended)}")
        else:
            report_lines.append("建议: 当前所有ETF表现不佳，建议清仓或保持空仓")
        
        if risk_warning:
            report_lines.extend([
                f"",
                f"风险提示: {risk_warning}"
            ])
        
        if trading_advice:
            report_lines.extend([
                f"",
                f"调仓建议: {trading_advice}"
            ])
        
        # 添加LLM原始分析（纯文本格式）
        report_lines.extend([
            f"",
            f"───────────────────────────────────────────────────────",
            f"六、AI详细分析",
            f"───────────────────────────────────────────────────────",
            f"",
            llm_content
        ])
        
        report = "\n".join(report_lines)
        
        # 8. 返回结果（纯文本格式，不包含JSON）
        result_message = report
        
        return {
            "messages": list(messages) + [AIMessage(content=result_message, name="ETF_Rotation_AI")]
        }
        
    except Exception as e:
        import traceback
        error_msg = f"ETF AI轮动分析失败: {str(e)}\n\n{traceback.format_exc()}"
        print(f"[ETF AI轮动] 错误: {error_msg}")
        # 确保messages是列表
        msg_list = list(messages) if messages else []
        return {
            "messages": msg_list + [AIMessage(content=error_msg, name="ETF_Rotation_AI")]
        }
