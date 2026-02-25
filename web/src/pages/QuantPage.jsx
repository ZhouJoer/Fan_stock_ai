import React, { useMemo, useState } from 'react'
import { usePersistedList } from '../hooks/usePersistedList.js'
import { formatTs } from '../utils/format.js'
import * as quantApi from '../api/quant.js'

const strategyNames = {
    dual_ma: '双均线策略',
    macd: 'MACD策略',
    rsi_bb: 'RSI+布林带策略',
    multi_factor: '多因子综合策略',
    trend: '趋势跟踪策略',
    mean_reversion: '均值回归策略',
    chanlun: '缠论策略'
}

const strategyTypeNames = {
    trend: '趋势跟踪',
    mean_reversion: '均值回归',
    chanlun: '缠论',
    adaptive: '自适应策略'
}

const riskNames = {
    aggressive: '激进进取',
    balanced: '均衡稳健',
    conservative: '稳健保守'
}

const actionEmoji = { BUY: '📈', SELL: '📉', HOLD: '⏸' }
const actionText = { BUY: '买入开仓', SELL: '卖出平仓', HOLD: '持有观望' }

export default function QuantPage({ user }) {
    const { items, addItem, clear, count } = usePersistedList('my_stock:quant', 10, user?.user_id)
    const [mode, setMode] = useState('agent')  // agent | backtest | ai-decision

    // Agent 模式
    const [agentQuery, setAgentQuery] = useState('')
    const [agentBusy, setAgentBusy] = useState(false)
    const [agentError, setAgentError] = useState('')

    // 普通回测
    const [stockCode, setStockCode] = useState('')
    const [strategy, setStrategy] = useState('multi_factor')
    const [strategyType, setStrategyType] = useState('trend')
    const [riskPreference, setRiskPreference] = useState('balanced')
    const [initialCapital, setInitialCapital] = useState('100000')
    const [backtestBusy, setBacktestBusy] = useState(false)
    const [backtestError, setBacktestError] = useState('')

    // AI 决策 + AI 回测
    const [aiStockCode, setAiStockCode] = useState('')
    const [aiStrategyType, setAiStrategyType] = useState('adaptive')
    const [aiRiskPreference, setAiRiskPreference] = useState('balanced')
    const [hasPosition, setHasPosition] = useState(false)
    const [entryPrice, setEntryPrice] = useState('')
    const [daysHeld, setDaysHeld] = useState('')
    const [aiDecisionBusy, setAiDecisionBusy] = useState(false)
    const [aiDecisionError, setAiDecisionError] = useState('')
    const [aiBacktestBusy, setAiBacktestBusy] = useState(false)
    const [aiBacktestDays, setAiBacktestDays] = useState('252')
    const [aiInitialCapital, setAiInitialCapital] = useState('100000')
    const [useLlmSignals, setUseLlmSignals] = useState(false)
    const [llmSampleRate, setLlmSampleRate] = useState('10')
    const [backtestProgress, setBacktestProgress] = useState(0)

    const canRunAgent = useMemo(() => agentQuery.trim().length > 0 && !agentBusy, [agentQuery, agentBusy])
    const canRunBacktest = useMemo(() => stockCode.trim().length > 0 && !backtestBusy, [stockCode, backtestBusy])
    const canRunAiDecision = useMemo(() => aiStockCode.trim().length > 0 && !aiDecisionBusy && !aiBacktestBusy, [aiStockCode, aiDecisionBusy, aiBacktestBusy])

    async function runAgent() {
        const q = agentQuery.trim()
        if (!q) return
        setAgentBusy(true)
        setAgentError('')
        try {
            const data = await quantApi.agentQuery({ query: q })
            let resultText = `【AI Agent 量化分析】\n请求：${q}\n\n`
            if (data.full_workflow && Array.isArray(data.full_workflow)) {
                resultText += '【工作流执行过程】\n'
                data.full_workflow.forEach((msg) => {
                    if (msg.name !== 'User') {
                        resultText += `\n[${msg.name}]\n${msg.content}\n`
                    }
                })
            } else {
                resultText += `【分析结果】\n${data.result ?? ''}`
            }
            addItem(resultText)
            setAgentQuery('')
        } catch (e) {
            setAgentError(String(e?.message || e))
        } finally {
            setAgentBusy(false)
        }
    }

    async function runAiDecision() {
        const code = aiStockCode.trim()
        if (!code) return
        setAiDecisionBusy(true)
        setAiDecisionError('')
        try {
            const data = await quantApi.aiDecision({
                stock_code: code,
                has_position: hasPosition,
                entry_price: hasPosition ? parseFloat(entryPrice) || null : null,
                days_held: hasPosition ? parseInt(daysHeld, 10) || null : null,
                strategy_type: aiStrategyType,
                risk_preference: aiRiskPreference
            })
            const result = data.result
            const indicators = result.indicators || {}
            const decision = result.decision || {}
            const text = `【🤖 AI 实时交易决策】📊 分析引擎：LLM大语言模型（DeepSeek）
股票代码：${result.stock_code}
策略配置：${strategyTypeNames[result.strategy_type] || result.strategy_type} + ${riskNames[result.risk_preference] || result.risk_preference}
分析时间：${new Date().toLocaleString('zh-CN')}

📊 技术指标：
  当前价格：¥${indicators.Current_Price?.toFixed(2) ?? '-'}
  MA5: ¥${indicators.MA5?.toFixed(2) ?? '-'}  MA10: ¥${indicators.MA10?.toFixed(2) ?? '-'}  MA20: ¥${indicators.MA20?.toFixed(2) ?? '-'}  MA60: ¥${indicators.MA60?.toFixed(2) ?? '-'}
  RSI(14): ${indicators.RSI?.toFixed(2) ?? '-'}
  MACD: ${indicators.MACD?.toFixed(4) ?? '-'}  MACD柱: ${indicators.MACD_Hist?.toFixed(4) ?? '-'} ${indicators.MACD_Hist > 0 ? '(金叉)' : '(死叉)'}
  布林带上轨 ¥${indicators.BB_Upper?.toFixed(2) ?? '-'}  中轨 ¥${indicators.BB_Middle?.toFixed(2) ?? '-'}  下轨 ¥${indicators.BB_Lower?.toFixed(2) ?? '-'}
  ATR: ${indicators.ATR?.toFixed(2) ?? '-'}

${actionEmoji[decision.action] ?? '⏸'} 决策：${actionText[decision.action] ?? decision.action}
信心度：${((decision.confidence ?? 0) * 100).toFixed(1)}%

💡 LLM决策理由：${decision.reasoning ?? '-'}

🎯 风险管理：  建议止损价：¥${decision.stop_loss?.toFixed(2) ?? '-'}  建议止盈价：¥${decision.take_profit?.toFixed(2) ?? '-'}  建议仓位：${((decision.position_size ?? 0) * 100).toFixed(0)}%

📝 LLM详细分析：${result.ai_analysis ?? '-'}

⚠️ 风险提示：此决策由LLM大模型生成，仅供参考，不构成投资建议。`
            addItem(text)
            setAiStockCode('')
            setEntryPrice('')
            setDaysHeld('')
        } catch (e) {
            setAiDecisionError(String(e?.message || e))
        } finally {
            setAiDecisionBusy(false)
        }
    }

    async function runAiBacktest() {
        const code = aiStockCode.trim()
        if (!code) return
        setAiBacktestBusy(true)
        setAiDecisionError('')
        setBacktestProgress(0)
        const days = parseInt(aiBacktestDays, 10) || 252
        const estimatedTime = Math.max(5, days / 50)
        const progressInterval = setInterval(() => {
            setBacktestProgress(prev => (prev >= 95 ? prev : prev + (95 - prev) * 0.1))
        }, estimatedTime * 10)

        try {
            clearInterval(progressInterval)
            setBacktestProgress(100)
            const data = await quantApi.aiBacktest({
                stock_code: code,
                initial_capital: parseFloat(aiInitialCapital) || 100000,
                days,
                strategy_type: aiStrategyType,
                risk_preference: aiRiskPreference,
                use_llm_signals: useLlmSignals,
                llm_sample_rate: parseInt(llmSampleRate, 10) || 10
            })
            const result = data.result
            const formatTrades = (trades) => {
                if (!trades || trades.length === 0) return '无交易记录'
                return trades.map(t => {
                    const action = t.type === 'BUY' ? '买入' : '卖出'
                    const date = new Date(t.date).toLocaleDateString('zh-CN')
                    const profitInfo = t.profit_pct != null ? ` | 盈亏:${t.profit_pct >= 0 ? '+' : ''}${t.profit_pct.toFixed(1)}%` : ''
                    const holdingInfo = t.holding_days != null ? ` | 持仓${t.holding_days}天` : ''
                    const reason = t.reason ? `\n      💡 ${t.reason}` : ''
                    return `  ${action} ${date} ¥${t.price?.toFixed(2)}${profitInfo}${holdingInfo}${reason}`
                }).join('\n')
            }
            let adaptiveInfo = ''
            if (aiStrategyType === 'adaptive' && result.adaptive_info) {
                const dist = result.adaptive_info.state_distribution || {}
                const distStr = Object.entries(dist).map(([k, v]) => `${k}:${v}%`).join(' | ')
                adaptiveInfo = `
ℹ️ 自适应分析：  市场状态分布：${distStr}
  状态检测次数：${result.adaptive_info.total_state_checks}次（每${result.adaptive_info.check_interval}天）
`
            }
            let llmInfo = ''
            if (result.use_llm_signals && result.llm_info) {
                llmInfo = `
📥 LLM信号引擎：  LLM调用次数：${result.llm_info.call_count}次  采样频率：每${result.llm_info.sample_rate}天  平均调用间隔：${result.llm_info.avg_call_interval}天`
            }
            const signalEngine = result.use_llm_signals
                ? '📥 LLM大语言模型（DeepSeek） 根据策略类型使用不同prompt'
                : '📈 规则算法（MA/RSI/MACD/布林带评分）'
            const sharpeLabel = result.sharpe_ratio > 2 ? '(优秀)' : result.sharpe_ratio > 1 ? '(良好)' : '(一般)'
            const text = `【量化策略回测】📈 股票代码：${result.stock_code}
ℹ️ 信号引擎：${signalEngine}
策略配置：${strategyTypeNames[aiStrategyType]} + ${riskNames[aiRiskPreference]}
策略类型：${result.strategy}
回测周期：${result.start_date} ~ ${result.end_date}（共${result.trading_days ?? '-'}个交易日） 回测耗时：${result.backtest_time ?? '-'}秒  初始资金：¥${result.initial_capital?.toLocaleString()}
${adaptiveInfo}${llmInfo}
📈 回测结果

📋 收益指标：  总收益率：${result.total_return}%
  年化收益率：${result.annual_return}%（基于${result.trading_days ?? '-'}交易日年化）
  最终资金：¥${result.final_capital?.toLocaleString()}

⚠️ 风险指标：  最大回撤：${result.max_drawdown}%
  夏普比率：${result.sharpe_ratio} ${sharpeLabel}

😁 交易统计：  交易次数：${result.total_trades}笔  胜率：${result.win_rate}%

🔝 最近交易记录（含决策原因）：
${formatTrades(result.trades)}

⚠️ 风险提示：  - 历史回测不代表未来收益  - 策略信号引擎：${result.use_llm_signals ? 'LLM大语言模型' : '规则算法'}
  - 实盘存在滑点和手续费  - 建议小资金模拟验证。`
            addItem({ text, chart: result.chart })
            setAiStockCode('')
        } catch (e) {
            clearInterval(progressInterval)
            setAiDecisionError(String(e?.message || e))
        } finally {
            setAiBacktestBusy(false)
            setTimeout(() => setBacktestProgress(0), 500)
        }
    }

    async function runBacktest() {
        const code = stockCode.trim()
        if (!code) return
        setBacktestBusy(true)
        setBacktestError('')
        try {
            const data = await quantApi.backtest({
                stock_code: code,
                strategy,
                strategy_type: strategyType,
                risk_preference: riskPreference,
                initial_capital: parseFloat(initialCapital) || 100000
            })
            const result = data.result
            const configInfo = result.strategy_config
                ? `\n策略参数：${result.strategy_config.name || ''} | 仓位${result.strategy_config.position_size} | 止损${result.strategy_config.stop_loss} | 止盈${result.strategy_config.take_profit}`
                : ''
            const sharpeLabel = result.sharpe_ratio > 2 ? '(优秀)' : result.sharpe_ratio > 1 ? '(良好)' : '(一般)'
            const text = `【普通量化回测】股票代码：${result.stock_code}
策略类型：${strategyNames[result.strategy] || result.strategy}
策略配置：${strategyTypeNames[result.strategy_type] || '趋势跟踪'} + ${riskNames[result.risk_preference] || '均衡稳健'}${configInfo}
回测周期：${result.start_date} - ${result.end_date}
初始资金：¥${result.initial_capital?.toLocaleString()}

📈 回测结果

💰 收益指标：  总收益率：${result.total_return}%
  年化收益率：${result.annual_return}%
  最终资金：¥${result.final_capital?.toLocaleString()}

⚠️ 风险指标：  最大回撤：${result.max_drawdown}%
  夏普比率：${result.sharpe_ratio} ${sharpeLabel}

📋 交易统计：  交易次数：${result.total_trades}笔  胜率：${result.win_rate}%

⚠️ 风险提示：  - 历史回测不代表未来收益  - 实盘存在滑点和手续费  - 建议小资金模拟验证。`
            addItem({ text, chart: result.chart })
            setStockCode('')
        } catch (e) {
            setBacktestError(String(e?.message || e))
        } finally {
            setBacktestBusy(false)
        }
    }

    return (
        <section className="panel poolPanel">
            <header className="panelHeader">
                <div className="panelTitle">📊 量化分析</div>
                <div className="panelMeta">已保存：{count}</div>
            </header>

            <div className="panelInput">
                <div className="modeSwitch">
                    <button className={mode === 'agent' ? 'modeBtnActive' : 'modeBtn'} onClick={() => setMode('agent')}>
                        🤖 AI Agent 量化
                    </button>
                    <button className={mode === 'backtest' ? 'modeBtnActive' : 'modeBtn'} onClick={() => setMode('backtest')}>
                        📈 普通回测
                    </button>
                    <button className={mode === 'ai-decision' ? 'modeBtnActive' : 'modeBtn'} onClick={() => setMode('ai-decision')}>
                        🎯 AI 实时决策
                    </button>
                </div>

                {mode === 'agent' && (
                    <>
                        <div className="infoBox" style={{ marginBottom: '12px' }}>
                            输入量化分析请求，AI 将自动执行：数据收集 → 量化策略 → 审计评估，并返回完整分析报告与风险提示。
                        </div>
                        <div className="formGroup">
                            <label className="label">分析请求</label>
                            <textarea
                                className="textarea"
                                value={agentQuery}
                                onChange={(e) => setAgentQuery(e.target.value)}
                                placeholder="例如：对贵州茅台进行量化分析，回测多因子策略；AI 将自动执行数据收集、量化策略、审计评估，提供完整分析报告和风险提示。"
                                rows={5}
                            />
                        </div>
                        <div className="actions">
                            <button className="buttonPrimary" disabled={!canRunAgent} onClick={runAgent}>
                                {agentBusy ? '分析中...' : '🚀 运行 AI 量化分析'}
                            </button>
                            <button className="button" onClick={clear} disabled={agentBusy}>清空</button>
                        </div>
                        {agentError && <div className="errorText">{agentError}</div>}
                    </>
                )}

                {mode === 'backtest' && (
                    <>
                        <div className="formGroup">
                            <label className="label">股票代码</label>
                            <input
                                type="text"
                                className="input"
                                value={stockCode}
                                onChange={(e) => setStockCode(e.target.value)}
                                placeholder="例如：600519.SH 或 600000.SH"
                            />
                        </div>
                        <div className="formGroup">
                            <label className="label">回测策略</label>
                            <select className="select" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                                <option value="multi_factor">多因子综合策略（推荐）</option>
                                <option value="dual_ma">双均线策略（MA5/MA20）</option>
                                <option value="macd">MACD策略（金叉死叉）</option>
                                <option value="rsi_bb">RSI+布林带策略（超买超卖）</option>
                                <option value="trend">趋势跟踪策略（追涨杀跌）</option>
                                <option value="mean_reversion">均值回归策略（低买高卖）</option>
                            </select>
                        </div>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <div className="formGroup" style={{ flex: 1 }}>
                                <label className="label">策略类型</label>
                                <select className="select" value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
                                    <option value="trend">📈 趋势跟踪（顺势而为）</option>
                                    <option value="mean_reversion">📉 均值回归（逆势操作）</option>
                                    <option value="chanlun">📐 缠论（中枢与买卖点）</option>
                                </select>
                            </div>
                            <div className="formGroup" style={{ flex: 1 }}>
                                <label className="label">风险偏好</label>
                                <select className="select" value={riskPreference} onChange={(e) => setRiskPreference(e.target.value)}>
                                    <option value="aggressive">🔥 激进进取（高收益高风险）</option>
                                    <option value="balanced">⚖️ 均衡稳健（平衡风险收益）</option>
                                    <option value="conservative">🛡️ 稳健保守（保守优先）</option>
                                </select>
                            </div>
                        </div>
                        <div className="formGroup">
                            <label className="label">初始资金（元）</label>
                            <input
                                type="number"
                                className="input"
                                value={initialCapital}
                                onChange={(e) => setInitialCapital(e.target.value)}
                                placeholder="100000"
                            />
                        </div>
                        <div className="actions">
                            <button className="buttonPrimary" disabled={!canRunBacktest} onClick={runBacktest}>
                                {backtestBusy ? '回测中...' : '📈 开始回测'}
                            </button>
                            <button className="button" onClick={clear} disabled={backtestBusy}>清空</button>
                        </div>
                        {backtestError && <div className="errorText">{backtestError}</div>}
                    </>
                )}

                {mode === 'ai-decision' && (
                    <>
                        <div className="formGroup">
                            <label className="label">股票代码</label>
                            <input
                                type="text"
                                className="input"
                                value={aiStockCode}
                                onChange={(e) => setAiStockCode(e.target.value)}
                                placeholder="例如：600519.SH 或东方财富"
                            />
                        </div>
                        <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
                            <div className="formGroup" style={{ flex: 1 }}>
                                <label className="label">策略类型</label>
                                <select className="select" value={aiStrategyType} onChange={(e) => setAiStrategyType(e.target.value)}>
                                    <option value="adaptive">📊 自适应策略（推荐）</option>
                                    <option value="trend">📈 趋势跟踪</option>
                                    <option value="mean_reversion">📉 均值回归</option>
                                </select>
                            </div>
                            <div className="formGroup" style={{ flex: 1 }}>
                                <label className="label">风险偏好</label>
                                <select className="select" value={aiRiskPreference} onChange={(e) => setAiRiskPreference(e.target.value)}>
                                    <option value="aggressive">🔥 激进</option>
                                    <option value="balanced">⚖️ 均衡</option>
                                    <option value="conservative">🛡️ 保守</option>
                                </select>
                            </div>
                        </div>

                        <div className="infoBox" style={{ padding: '12px', marginBottom: '12px', border: '1px solid #E5E7EB' }}>
                            <div style={{ fontWeight: 'bold', marginBottom: '12px', color: '#333' }}>📈 策略回测</div>
                            <div className="formGroup" style={{ marginBottom: '8px' }}>
                                <label className="label">回测天数</label>
                                <select className="select" value={aiBacktestDays} onChange={(e) => setAiBacktestDays(e.target.value)}>
                                    <option value="60">60 个交易日（约1季度）</option>
                                    <option value="126">126 个交易日（约半年）</option>
                                    <option value="252">252 个交易日（约1年）</option>
                                    <option value="504">504 个交易日（约2年）</option>
                                    <option value="756">756 个交易日（约3年）</option>
                                </select>
                            </div>
                            <div className="formGroup" style={{ marginBottom: '8px' }}>
                                <label className="label">初始资金（元）</label>
                                <input
                                    type="number"
                                    className="input"
                                    value={aiInitialCapital}
                                    onChange={(e) => setAiInitialCapital(e.target.value)}
                                    placeholder="100000"
                                />
                            </div>
                            <div style={{
                                marginBottom: '12px',
                                padding: '10px',
                                background: useLlmSignals ? 'linear-gradient(135deg, #6a5acd20, #9370db30)' : '#f5f5f5',
                                borderRadius: '6px',
                                border: useLlmSignals ? '1px solid #9370db' : '1px solid #ddd'
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                    <input type="checkbox" id="useLlmSignals" checked={useLlmSignals} onChange={(e) => setUseLlmSignals(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                                    <label htmlFor="useLlmSignals" style={{ fontWeight: 'bold', cursor: 'pointer' }}>🧠 使用LLM生成交易信号</label>
                                </div>
                                <div style={{ fontSize: '11px', color: '#666', marginBottom: useLlmSignals ? '8px' : '0' }}>
                                    {useLlmSignals ? (
                                        <>
                                            <div>✓ 使用 DeepSeek 大模型分析技术指标</div>
                                            <div>✓ 不同策略类型使用不同的专属 prompt</div>
                                            <div>✓ 决策更智能，但速度较慢</div>
                                        </>
                                    ) : (
                                        <>
                                            <div>当前：规则算法（MA/RSI/MACD 评分）</div>
                                            <div>速度快，结果一致性好</div>
                                        </>
                                    )}
                                </div>
                                {useLlmSignals && (
                                    <div className="formGroup" style={{ margin: 0 }}>
                                        <label className="label" style={{ fontSize: '12px' }}>LLM 采样频率</label>
                                        <select className="select" value={llmSampleRate} onChange={(e) => setLlmSampleRate(e.target.value)} style={{ fontSize: '12px' }}>
                                            <option value="5">每 5 天调用一次（更精确，更慢）</option>
                                            <option value="10">每 10 天调用一次（推荐）</option>
                                            <option value="20">每 20 天调用一次（更快）</option>
                                        </select>
                                    </div>
                                )}
                            </div>
                            <button
                                className="buttonPrimary"
                                style={{ width: '100%', background: useLlmSignals ? '#6a5acd' : '#9C27B0' }}
                                disabled={!canRunAiDecision}
                                onClick={runAiBacktest}
                            >
                                {aiBacktestBusy ? '回测中...' : useLlmSignals ? '🧠 运行 LLM 回测' : '📈 运行策略回测'}
                            </button>
                            {aiBacktestBusy && (
                                <div className="progressContainer" style={{ marginTop: '10px', background: 'rgba(156, 39, 176, 0.1)', padding: '12px', borderRadius: '8px' }}>
                                    <div className="progressTitle" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                        <div className="spinner" style={{ width: '18px', height: '18px' }} />
                                        <span>正在进行策略回测分析...</span>
                                    </div>
                                    <div className="progressBar">
                                        <div className="progressFill" style={{ width: `${backtestProgress}%` }} />
                                    </div>
                                    <div className="progressInfo" style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '12px', color: '#666' }}>
                                        <span>{Math.round(backtestProgress)}% 完成</span>
                                        <span>分析 {aiBacktestDays} 个交易日</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="infoBox" style={{ padding: '12px', marginBottom: '12px', background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)', border: '1px solid #ce93d8' }}>
                            <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#7b1fa2' }}>🤖 LLM 实时决策 <span style={{ fontSize: '11px', color: '#9c27b0', fontWeight: 'normal' }}>（调用大语言模型）</span></div>
                            <div className="formGroup" style={{ marginBottom: '8px' }}>
                                <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                                    <input type="checkbox" checked={hasPosition} onChange={(e) => setHasPosition(e.target.checked)} style={{ margin: 0 }} />
                                    当前有持仓
                                </label>
                            </div>
                            {hasPosition && (
                                <>
                                    <div className="formGroup">
                                        <label className="label">开仓价格（元）</label>
                                        <input type="number" step="0.01" className="input" value={entryPrice} onChange={(e) => setEntryPrice(e.target.value)} placeholder="例如：92.50" />
                                    </div>
                                    <div className="formGroup">
                                        <label className="label">持仓天数</label>
                                        <input type="number" className="input" value={daysHeld} onChange={(e) => setDaysHeld(e.target.value)} placeholder="例如：5" />
                                    </div>
                                </>
                            )}
                            <button className="buttonPrimary" style={{ width: '100%' }} disabled={!canRunAiDecision} onClick={runAiDecision}>
                                {aiDecisionBusy ? '分析中...' : '🤖 获取 AI 实时决策'}
                            </button>
                        </div>

                        <div className="actions">
                            <button className="button" onClick={clear} disabled={aiDecisionBusy || aiBacktestBusy}>清空</button>
                        </div>
                        {aiDecisionError && <div className="errorText">{aiDecisionError}</div>}
                        <div className="infoBox" style={{ marginTop: '12px', fontSize: '13px', color: '#666' }}>
                            💡 <strong>功能说明</strong>：<br />
                            • <strong>策略回测</strong>：规则算法（MA/RSI/MACD/布林带）验证历史表现，确保一致性。<br />
                            • <strong>🤖 实时决策</strong>：<span style={{ color: '#9C27B0', fontWeight: 'bold' }}>调用 LLM 大模型</span>综合分析，给出智能交易建议。
                        </div>
                    </>
                )}
            </div>

            <div className="panelList" aria-label="量化分析 结果列表">
                {items.length === 0 ? (
                    <div className="empty">暂无已保存结果</div>
                ) : (
                    items.map((it) => (
                        <article key={it.id} className="card">
                            <div className="cardMeta">{formatTs(it.ts)}</div>
                            <pre className="cardText">{typeof it === 'string' ? it : (it.text || it)}</pre>
                            {it.chart && (
                                <div className="chartContainer">
                                    <img
                                        src={it.chart.startsWith('data:') ? it.chart : `data:image/png;base64,${it.chart}`}
                                        alt="回测图表"
                                        className="chartImage"
                                        onError={(e) => { e.target.style.display = 'none' }}
                                    />
                                </div>
                            )}
                        </article>
                    ))
                )}
            </div>
        </section>
    )
}
