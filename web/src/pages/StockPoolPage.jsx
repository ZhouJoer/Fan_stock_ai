import React, { useMemo, useState, useEffect, useRef } from 'react'
import { usePersistedList } from '../hooks/usePersistedList.js'
import { formatTs } from '../utils/format.js'
import * as poolApi from '../api/pool.js'
import { TradesHistory } from '../components/TradesHistory.jsx'

export default function StockPoolPage({ user }) {
    const { items, addItem, clear, count } = usePersistedList('my_stock:stock_pool', 10, user?.user_id)
    const [mode, setMode] = useState('manage')  // manage, sim, backtest

    // 选股池配置
    const [poolName, setPoolName] = useState('我的选股池')
    const [stockInput, setStockInput] = useState('')
    const [stocks, setStocks] = useState([])
    const [initialCapital, setInitialCapital] = useState('1000000')
    const [strategyType, setStrategyType] = useState('adaptive')
    const [riskPreference, setRiskPreference] = useState('balanced')
    const [allocationMethod, setAllocationMethod] = useState('signal_strength')

    // 回测配置
    const [backtestDays, setBacktestDays] = useState('252')
    const [rebalanceInterval, setRebalanceInterval] = useState('5')
    const [noLookahead, setNoLookahead] = useState(false)
    const [backtestStartDate, setBacktestStartDate] = useState('')

    // LLM模式配置
    const [usePoolLlmSignals, setUsePoolLlmSignals] = useState(false)
    const [poolLlmSampleRate, setPoolLlmSampleRate] = useState('5')

    // 高胜率模式
    const [poolHighWinRateMode, setPoolHighWinRateMode] = useState(false)

    // 选股来源（手动列表 / 分行业龙头；仅手动需管理选股池）
    const [universeSource, setUniverseSource] = useState('manual')  // manual | industry
    const [industryList, setIndustryList] = useState([])           // 选中的行业名称列表
    const [leadersPerIndustry, setLeadersPerIndustry] = useState(1)
    const [industryNames, setIndustryNames] = useState([])          // 从 API 拉取的行业列表
    const [industryLeadersPreview, setIndustryLeadersPreview] = useState(null)  // 龙头预览 { result, code_to_industry }
    // 状态
    const [busy, setBusy] = useState(false)
    const [poolStopping, setPoolStopping] = useState(false)
    const [error, setError] = useState('')
    const [signals, setSignals] = useState(null)
    const [allocation, setAllocation] = useState(null)
    const [savedPools, setSavedPools] = useState([])
    const [backtestProgress, setBacktestProgress] = useState(0)

    // 实时决策显示
    const [liveDecisions, setLiveDecisions] = useState([])
    const [currentLlmStatus, setCurrentLlmStatus] = useState('')
    // 流式回测 EventSource 引用，用于停止时关闭连接（可选）
    const backtestEventSourceRef = useRef(null)
    const backtestSessionIdRef = useRef(null)

    // 模拟仓状态
    const [simAccounts, setSimAccounts] = useState([])
    const [selectedAccountId, setSelectedAccountId] = useState('')
    const [simAccountInfo, setSimAccountInfo] = useState(null)
    const [useSimLlm, setUseSimLlm] = useState(false)
    const [newAccountId, setNewAccountId] = useState('')

    // 添加股票到池
    function addStock() {
        const code = stockInput.trim()
        if (code && !stocks.includes(code)) {
            setStocks([...stocks, code])
            setStockInput('')
        }
    }

    // 移除股票
    function removeStock(code) {
        setStocks(stocks.filter(s => s !== code))
    }

    // 获取信号
    async function fetchSignals() {
        if (stocks.length === 0) {
            setError('请先添加股票到选股池')
            return
        }

        setBusy(true)
        setError('')
        try {
            const data = await poolApi.signals({ stocks, risk_preference: riskPreference })
            setSignals(data.result)

            // 同时获取配置建议
            const allocData = await poolApi.allocation({
                stocks,
                total_capital: parseFloat(initialCapital) || 1000000,
                allocation_method: allocationMethod,
                risk_preference: riskPreference
            }).catch(() => ({}))
            if (allocData?.result) {
                setAllocation(allocData.result)
            }

            // 生成报告文本
            const signalData = data.result
            let reportText = `【选股池信号分析】\n`
            reportText += `股票池：${stocks.join(', ')}\n`
            reportText += `分析时间：${new Date().toLocaleString('zh-CN')}\n`
            reportText += `风险偏好：${riskPreference}\n\n`

            let buyCount = 0, sellCount = 0, neutralCount = 0

            Object.entries(signalData).forEach(([code, sig]) => {
                const signalType = sig.signal_type || 'Neutral'
                const emoji = signalType === 'LongOpen' ? '🟢' : signalType === 'ShortOpen' ? '🔴' : '🟡'
                const action = signalType === 'LongOpen' ? '买入' : signalType === 'ShortOpen' ? '卖出' : '观望'

                if (signalType === 'LongOpen') buyCount++
                else if (signalType === 'ShortOpen') sellCount++
                else neutralCount++

                reportText += `${emoji} ${code}: ${action}\n`
                reportText += `   信号强度: ${(sig.signal_strength || 0).toFixed(4)} (${sig.strength_level || '无效'})\n`
                reportText += `   市场状态: ${sig.market_state || 'N/A'}\n\n`
            })

            reportText += `\n【信号汇总】\n`
            reportText += `买入信号: ${buyCount}只\n`
            reportText += `卖出信号: ${sellCount}只\n`
            reportText += `中性信号: ${neutralCount}只\n`

            addItem(reportText)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 加载模拟仓账户列表
    async function loadSimAccounts() {
        try {
            const data = await poolApi.simList()
            if (data?.result) setSimAccounts(data.result)
        } catch (e) {
            console.error('[选股池模拟仓] 加载账户列表失败:', e)
        }
    }

    // 创建模拟仓账户
    async function createSimAccount() {
        const accountId = newAccountId.trim()
        if (!accountId) {
            setError('请输入账户ID')
            return
        }

        // 验证账户ID格式（只允许字母、数字、下划线、中文）
        if (!/^[\w\u4e00-\u9fa5]+$/.test(accountId)) {
            setError('账户ID只能包含字母、数字、下划线和中文')
            return
        }

        setBusy(true)
        setError('')
        try {
            const data = await poolApi.simCreate({
                account_id: accountId,
                initial_capital: parseFloat(initialCapital) || 1000000,
                stock_pool: stocks
            })
            if (data?.error) throw new Error(data.error)

            setSelectedAccountId(accountId)
            setNewAccountId('')  // 清空输入框
            setError('')  // 清空错误信息
            await fetchSimAccount(accountId)
            await loadSimAccounts()
            
            // 显示成功消息
            const successMsg = `账户 "${accountId}" 创建成功！初始资金: ¥${(parseFloat(initialCapital) || 1000000).toLocaleString()}`
            console.log('[选股池模拟仓]', successMsg)
        } catch (e) {
            const errorMsg = String(e?.message || e)
            setError(errorMsg)
            console.error('[选股池模拟仓] 创建账户失败:', errorMsg, e)
        } finally {
            setBusy(false)
        }
    }

    // 获取模拟仓账户信息
    async function fetchSimAccount(accountId) {
        if (!accountId) return

        setBusy(true)
        setError('')
        try {
            const data = await poolApi.simGet(accountId)
            setSimAccountInfo(data.result)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 执行调仓
    async function rebalanceSimAccount(accountId) {
        if (!accountId) {
            setError('请先选择或创建账户')
            return
        }

        if (stocks.length === 0) {
            setError('请先添加股票到选股池')
            return
        }

        setBusy(true)
        setError('')
        try {
            const data = await poolApi.simRebalance(accountId, {
                account_id: accountId,
                risk_preference: riskPreference,
                use_llm: useSimLlm
            })

            // 刷新账户信息
            await fetchSimAccount(accountId)

            // 显示调仓结果
            const result = data.result
            let resultText = `【模拟仓调仓结果】\n`
            resultText += `账户: ${accountId}\n`
            resultText += `调仓方式: ${useSimLlm ? 'LLM调仓' : '规则调仓'}\n`
            resultText += `执行交易: ${result.executed_count}笔\n\n`

            if (useSimLlm && result.llm_analysis) {
                resultText += `LLM分析: ${result.llm_analysis}\n\n`
            }

            if (result.trades && result.trades.length > 0) {
                resultText += `交易明细:\n`
                result.trades.forEach(trade => {
                    resultText += `${trade.type === 'buy' ? '买入' : '卖出'} ${trade.stock_code} ${trade.shares}股 @ ¥${trade.price?.toFixed(2)}\n`
                    if (trade.profit_loss) {
                        resultText += `  盈亏: ${trade.profit_loss >= 0 ? '+' : ''}¥${trade.profit_loss.toFixed(2)}\n`
                    }
                })
            } else {
                resultText += `本次调仓无交易执行\n`
            }

            addItem(resultText)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 初始化时加载账户列表；回测且选行业时拉取行业列表
    useEffect(() => {
        if (mode === 'sim') {
            loadSimAccounts()
        }
        if (mode === 'backtest' && universeSource === 'industry' && industryNames.length === 0) {
            poolApi.industryNames()
                .then(data => { if (data?.result?.length) setIndustryNames(data.result) })
                .catch(() => {})
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [mode, universeSource])

    // 停止组合回测（仅对 LLM 流式回测有效）
    function stopPoolBacktest() {
        if (poolStopping || !backtestSessionIdRef.current) return
        setPoolStopping(true)
        poolApi.backtestStop(backtestSessionIdRef.current)
            .finally(() => setPoolStopping(false))
        setCurrentLlmStatus('已发送停止请求，等待后端返回局部结果…')
    }

    // 运行回测（手动需有股票；分行业需选行业）
    const canRunBacktest = useMemo(() => {
        if (universeSource === 'manual') return stocks.length > 0
        if (universeSource === 'industry') return industryList.length > 0
        return false
    }, [universeSource, stocks.length, industryList.length])

    /** 选股池回测/保存时使用的因子相关参数（因子挖掘已移除，固定为 none） */
    function getFactorConfigForApi() {
        return {
            selection_mode: 'none',
            selection_top_n: 10,
            selection_interval: 0,
            factor_set: 'hybrid',
            score_weights: null,
            weight_source: 'manual',
            model_name: ''
        }
    }

    async function runBacktest() {
        if (!canRunBacktest) {
            if (universeSource === 'manual') setError('请先添加股票到选股池')
            else setError('请至少选择一个行业')
            return
        }

        setBusy(true)
        setError('')
        setBacktestProgress(0)
        setLiveDecisions([])
        setCurrentLlmStatus('')
        const factorConfig = getFactorConfigForApi()
        if (noLookahead && !(backtestStartDate || '').trim()) {
            setError('无前视模式下请填写回测起始日期')
            setBusy(false)
            return
        }

        // LLM模式使用SSE流式接收决策
        if (usePoolLlmSignals) {
            try {
                const sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
                    ? crypto.randomUUID() : `pool-${Date.now()}-${Math.random().toString(36).slice(2)}`
                backtestSessionIdRef.current = sessionId
                const params = new URLSearchParams({
                    session_id: sessionId,
                    stocks: (universeSource === 'manual' ? stocks : []).join(','),
                    initial_capital: parseFloat(initialCapital) || 1000000,
                    days: parseInt(backtestDays) || 252,
                    strategy_type: strategyType,
                    risk_preference: riskPreference,
                    allocation_method: allocationMethod,
                    rebalance_interval: parseInt(rebalanceInterval) || 5,
                    use_llm_signals: 'true',
                    llm_sample_rate: parseInt(poolLlmSampleRate) || 5,
                    high_win_rate_mode: poolHighWinRateMode ? 'true' : 'false',
                    universe_source: universeSource,
                    universe_index: '',
                    industry_list: (universeSource === 'industry' ? industryList : []).join(','),
                    leaders_per_industry: String(universeSource === 'industry' ? (leadersPerIndustry || 1) : 1),
                    selection_mode: factorConfig.selection_mode,
                    selection_top_n: String(factorConfig.selection_top_n),
                    selection_interval: String(factorConfig.selection_interval),
                    factor_set: factorConfig.factor_set,
                    score_weights: factorConfig.score_weights ? poolApi.encodeScoreWeights(factorConfig.score_weights) : '',
                    weight_source: factorConfig.weight_source || 'manual',
                    model_name: factorConfig.model_name || '',
                    no_lookahead: noLookahead ? 'true' : 'false',
                    start_date: (backtestStartDate || '').trim()
                })

                const eventSource = new EventSource(`/api/pool/backtest/stream?${params}`)
                backtestEventSourceRef.current = eventSource
                const receivedCompleteRef = { current: false }

                eventSource.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data)

                        if (msg.type === 'progress') {
                            setBacktestProgress(msg.data.percent)
                        } else if (msg.type === 'llm_start') {
                            // 组合决策显示不同的状态
                            if (msg.data.stock_code === '组合决策') {
                                setCurrentLlmStatus(`🧠 组合决策分析中 @ ${msg.data.date}（${msg.data.stocks_count}只股票）...`)
                            } else {
                                setCurrentLlmStatus(`🔄 正在分析 ${msg.data.stock_code} @ ${msg.data.date}...`)
                            }
                        } else if (msg.type === 'llm_decision') {
                            // 组合决策显示买卖建议
                            if (msg.data.stock_code === '组合') {
                                const buyList = msg.data.priority_buy || []
                                const sellList = msg.data.priority_sell || []
                                setCurrentLlmStatus(`✅ 组合决策: 买入${buyList.length}只 卖出${sellList.length}只`)
                                setLiveDecisions(prev => [...prev, {
                                    type: 'portfolio',
                                    date: msg.data.date,
                                    analysis: msg.data.reason,
                                    priority_buy: buyList,
                                    priority_sell: sellList,
                                    call_count: msg.data.call_count
                                }])
                            } else {
                                setCurrentLlmStatus(`✅ ${msg.data.stock_code}: ${msg.data.action} (${msg.data.confidence})`)
                                setLiveDecisions(prev => [...prev, {
                                    type: 'decision',
                                    ...msg.data
                                }])
                            }
                        } else if (msg.type === 'trade') {
                            setLiveDecisions(prev => [...prev, {
                                type: 'trade',
                                ...msg.data
                            }])
                        } else if (msg.type === 'complete') {
                            receivedCompleteRef.current = true
                            backtestEventSourceRef.current = null
                            eventSource.close()
                            setPoolStopping(false)
                            setBacktestProgress(100)
                            setCurrentLlmStatus('')

                            // 处理完整结果（含用户停止时的局部结果）
                            const data = msg.data
                            const result = data.result || {}
                            if (result.aborted) {
                                setError(result.aborted_message || '回测已停止，以下为局部结果')
                            }
                            processBacktestResult(result, data.chart)
                            setBusy(false)
                        } else if (msg.type === 'error') {
                            receivedCompleteRef.current = true
                            eventSource.close()
                            setPoolStopping(false)
                            setError(msg.data.message)
                            setBusy(false)
                        }
                    } catch (e) {
                        console.error('SSE解析错误:', e)
                    }
                }

                eventSource.onerror = () => {
                    backtestEventSourceRef.current = null
                    eventSource.close()
                    setPoolStopping(false)
                    if (!receivedCompleteRef.current) {
                        setError('连接中断，请重试')
                    }
                    setBusy(false)
                }

                return // SSE模式不继续执行下面的代码
            } catch (e) {
                setError(String(e?.message || e))
                setBusy(false)
                return
            }
        }

        // 非LLM模式使用普通POST请求
        let progressInterval = setInterval(() => {
            setBacktestProgress(prev => prev >= 95 ? prev : prev + (95 - prev) * 0.08)
        }, 500)

        try {
            clearInterval(progressInterval)
            setBacktestProgress(100)

            const data = await poolApi.backtest({
                stocks: universeSource === 'manual' ? stocks : [],
                initial_capital: parseFloat(initialCapital) || 1000000,
                days: parseInt(backtestDays) || 252,
                strategy_type: strategyType,
                risk_preference: riskPreference,
                allocation_method: allocationMethod,
                rebalance_interval: parseInt(rebalanceInterval) || 5,
                use_llm_signals: false,
                llm_sample_rate: parseInt(poolLlmSampleRate) || 10,
                universe_source: universeSource,
                universe_index: '',
                industry_list: universeSource === 'industry' ? industryList : null,
                leaders_per_industry: universeSource === 'industry' ? (leadersPerIndustry || 1) : 1,
                no_lookahead: noLookahead,
                start_date: (backtestStartDate || '').trim(),
                ...factorConfig
            })
            processBacktestResult(data.result, data.chart)
        } catch (e) {
            clearInterval(progressInterval)
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
            setTimeout(() => setBacktestProgress(0), 500)
        }
    }

    // 处理回测结果
    function processBacktestResult(result, chart) {
        const strategyNames = {
            'trend': '趋势跟踪',
            'mean_reversion': '均值回归',
            'adaptive': '自适应策略',
            'chanlun': '缠论'
        }
        const riskNames = {
            'aggressive': '激进进取',
            'balanced': '均衡稳健',
            'conservative': '稳健保守'
        }
        const allocNames = {
            'equal_weight': '等权重',
            'signal_strength': '信号强度',
            'risk_parity': '风险平价',
            'kelly': '凯利公式'
        }

        // 信号引擎显示
        const signalEngine = result.use_llm_signals
            ? '🧠 LLM大语言模型（DeepSeek）- 根据策略类型使用不同prompt'
            : '📊 规则算法（MA/RSI/MACD/布林带评分）'

        // LLM信息
        let llmInfo = ''
        if (result.use_llm_signals && result.llm_info) {
            llmInfo = `
🧠 LLM信号引擎：
  LLM调用次数：${result.llm_info.call_count}次
  采样频率：每${result.llm_info.sample_rate}天
  股票数量：${result.llm_info.stocks_count}只`
        }
        const poolDesc = universeSource === 'industry' && industryList.length > 0
            ? `分行业龙头（${industryList.join('、')}，每行业${leadersPerIndustry}只）`
            : stocks.join(', ')
        const allocationLabel = allocNames[result.allocation_method] || result.allocation_method || allocationMethod
        const reportText = `【选股池动态回测】
📊 回测模式：动态选股（每日信号判断买卖时机）
⚙️ 信号引擎：${signalEngine}
📐 仓位分配：${allocationLabel}
股票池：${poolDesc}
策略配置：${strategyNames[strategyType]} + ${riskNames[riskPreference]}
回测周期：${result.start_date} ~ ${result.end_date}（${result.trading_days}个交易日）
回测耗时：${result.backtest_time}秒
${llmInfo}
💰 收益指标：
  初始资金：¥${result.initial_capital?.toLocaleString()}
  最终资金：¥${result.final_capital?.toLocaleString()}
  总收益率：${result.total_return}%
  年化收益率：${result.annual_return}%

⚠️ 风险指标：
  最大回撤：${result.max_drawdown}%
  夏普比率：${result.sharpe_ratio} ${result.sharpe_ratio > 2 ? '(优秀)' : result.sharpe_ratio > 1 ? '(良好)' : '(一般)'}

📈 交易统计：
  交易次数：${result.total_trades}笔
  胜率：${result.win_rate}%

� 各股票历史盈亏：
${(result.stock_summary || []).map(s => {
            const profitSign = s.total_profit >= 0 ? '+' : ''
            const pctSign = s.total_profit_pct >= 0 ? '+' : ''
            const statusIcon = s.status === '持仓中' ? '📌' : '✅'
            const maxDdStr = s.max_drawdown != null ? `最大回撤${s.max_drawdown}%` : ''
            const periodStr = s.stock_period_return != null ? `期间涨跌${s.stock_period_return >= 0 ? '+' : ''}${s.stock_period_return}%` : ''
            const extraInfo = [maxDdStr, periodStr].filter(Boolean).join(' | ')
            return `  ${statusIcon} ${s.stock_code}: ${profitSign}¥${s.total_profit.toLocaleString()} (${pctSign}${s.total_profit_pct}%) | 买${s.buy_count}次/卖${s.sell_count}次${extraInfo ? ` | ${extraInfo}` : ''}${s.status === '持仓中' ? ` | 当前持仓${s.current_shares}股` : ''}`
        }).join('\n') || '  暂无交易记录'}

📦 当前持仓：
${(result.final_positions || []).map(p =>
            `  ${p.stock_code}: ${p.shares}股 | 市值¥${p.market_value.toLocaleString()} | 权重${p.weight}% | 盈亏${p.profit_loss_pct > 0 ? '+' : ''}${p.profit_loss_pct}%`
        ).join('\n') || '  空仓（已全部卖出）'}

⚠️ 风险提示：
  - 历史回测不代表未来收益
  - 策略信号引擎：${result.use_llm_signals ? 'LLM大语言模型' : '规则算法'}
  - 组合策略存在相关性风险`

        // 保存完整结果用于展示决策和持仓变动
        addItem({
            text: reportText,
            chart: chart,
            trades: result.all_trades || [],
            positions: result.position_snapshots || [],
            finalPositions: result.final_positions || [],
            stockSummary: result.stock_summary || [],
            liveDecisions: liveDecisions
        })

        // 清空实时决策
        setTimeout(() => {
            setLiveDecisions([])
            setBacktestProgress(0)
        }, 500)
    }

    // 保存选股池
    async function savePool() {
        if (stocks.length === 0) {
            setError('请先添加股票到选股池')
            return
        }

        setBusy(true)
        setError('')
        try {
            const factorConfig = getFactorConfigForApi()
            await poolApi.savePool({
                name: poolName,
                stocks,
                initial_capital: parseFloat(initialCapital) || 1000000,
                strategy_type: strategyType,
                risk_preference: riskPreference,
                allocation_method: allocationMethod,
                ...factorConfig,
                strategy_meta: {
                    updated_at: new Date().toISOString(),
                    no_lookahead: noLookahead,
                    suggested_backtest_start_date: (backtestStartDate || '').trim()
                }
            })

            // 刷新已保存列表
            loadSavedPools()
            addItem(`✅ 选股池 "${poolName}" 已保存`)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 加载已保存的选股池列表
    async function loadSavedPools() {
        try {
            const data = await poolApi.poolList()
            if (data?.result) setSavedPools(data.result)
        } catch (e) {
            console.error('加载选股池列表失败:', e)
        }
    }

    async function deletePool(name) {
        const ok = window.confirm(`确认删除选股池 "${name}" 吗？`)
        if (!ok) return
        setBusy(true)
        setError('')
        try {
            await poolApi.poolDelete(name)
            await loadSavedPools()
            addItem(`🗑️ 已删除选股池 "${name}"`)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 加载指定选股池
    async function loadPool(name) {
        setBusy(true)
        setError('')
        try {
            const data = await poolApi.poolLoad(name)
            const config = data.result
            setPoolName(config.name)
            setStocks(config.stocks)
            setInitialCapital(String(config.initial_capital))
            setStrategyType(config.strategy_type)
            setRiskPreference(config.risk_preference)
            setAllocationMethod(config.allocation_method)
            setNoLookahead(!!config.strategy_meta?.no_lookahead)
            setBacktestStartDate(config.strategy_meta?.suggested_backtest_start_date || '')

            addItem(`📂 已加载选股池 "${config.name}": ${config.stocks.join(', ')}`)
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    // 支持从“因子挖掘”页一键跳转并加载策略
    React.useEffect(() => {
        const pending = localStorage.getItem('stock:pending_pool_name')
        if (pending) {
            localStorage.removeItem('stock:pending_pool_name')
            setMode('manage')
            loadPool(pending)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // 组件加载时获取已保存的选股池
    React.useEffect(() => {
        loadSavedPools()
    }, [])

    const canRunAction = stocks.length > 0 && !busy

    // ─── 通用卡片样式 ─────────────────────────────────────────────
    const card = {
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '10px',
        padding: '16px',
        marginBottom: '14px'
    }
    const cardTitle = {
        fontWeight: '600',
        fontSize: '14px',
        color: 'rgba(255,255,255,0.85)',
        marginBottom: '12px'
    }

    return (
        <section className="panel poolPanel">
            {/* ── 顶部：标题 + 流程指示 ── */}
            <header className="panelHeader">
                <div>
                    <div className="panelTitle">📊 选股池策略管理</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginTop: '5px' }}>
                        {[
                            { key: 'backtest', label: '① 回测验证' },
                            { key: 'sim',      label: '② 模拟盘'  },
                            { key: 'live',     label: '③ 实盘'    },
                        ].map((step, i) => (
                            <React.Fragment key={step.key}>
                                {i > 0 && <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: '10px' }}>→</span>}
                                <span style={{
                                    padding: '2px 9px', borderRadius: '10px', fontSize: '11px', fontWeight: '500',
                                    background: mode === step.key ? 'rgba(86,185,255,0.25)' : 'rgba(255,255,255,0.08)',
                                    color: mode === step.key ? 'rgba(86,185,255,0.95)' : 'rgba(255,255,255,0.45)',
                                    border: mode === step.key ? '1px solid rgba(86,185,255,0.4)' : '1px solid rgba(255,255,255,0.1)',
                                }}>
                                    {step.label}
                                </span>
                            </React.Fragment>
                        ))}
                    </div>
                </div>
                <div className="panelMeta">已保存：{count}</div>
            </header>

            <div className="panelInput">
                {/* ── Tab 导航 ── */}
                <div style={{
                    display: 'flex', gap: '2px', marginBottom: '20px',
                    background: 'rgba(0,0,0,0.25)', borderRadius: '10px', padding: '4px'
                }}>
                    {[
                        { key: 'manage',   icon: '📋', label: '管理股票池', hint: `${stocks.length} 只股票` },
                        { key: 'backtest', icon: '📈', label: '组合回测',   hint: '历史验证策略' },
                        { key: 'sim',      icon: '💼', label: '模拟仓',     hint: '模拟实盘交易' },
                    ].map(tab => (
                        <button key={tab.key} onClick={() => setMode(tab.key)} style={{
                            flex: 1, padding: '10px 6px', border: 'none', borderRadius: '8px', cursor: 'pointer',
                            transition: 'all 0.2s',
                            background: mode === tab.key ? 'rgba(86,185,255,0.18)' : 'transparent',
                            color: mode === tab.key ? 'rgba(86,185,255,0.95)' : 'rgba(255,255,255,0.55)',
                            borderBottom: mode === tab.key ? '2px solid rgba(86,185,255,0.6)' : '2px solid transparent',
                        }}>
                            <div style={{ fontSize: '17px', marginBottom: '2px' }}>{tab.icon}</div>
                            <div style={{ fontSize: '13px', fontWeight: mode === tab.key ? '600' : '400' }}>{tab.label}</div>
                            <div style={{ fontSize: '10px', opacity: 0.65, marginTop: '1px' }}>{tab.hint}</div>
                        </button>
                    ))}
                </div>

                {/* ════════════════════════════════════════
                    管理股票池
                ════════════════════════════════════════ */}
                {mode === 'manage' && (
                    <div>
                        {/* 两列：当前股票池 + 已保存股票池 */}
                        <div style={{ display: 'flex', gap: '14px', marginBottom: '14px', flexWrap: 'wrap' }}>
                            {/* 左：当前股票池 */}
                            <div style={{ flex: '1 1 280px', minWidth: '260px' }}>
                                <div style={{
                                    background: 'rgba(86,185,255,0.07)',
                                    border: '1px solid rgba(86,185,255,0.22)',
                                    borderRadius: '10px', padding: '14px'
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                                        <span style={{ fontWeight: '600', fontSize: '14px', color: 'rgba(86,185,255,0.9)' }}>
                                            当前股票池
                                            {stocks.length > 0 && (
                                                <span style={{
                                                    marginLeft: '6px', background: 'rgba(86,185,255,0.3)',
                                                    borderRadius: '10px', padding: '1px 7px', fontSize: '12px'
                                                }}>{stocks.length}</span>
                                            )}
                                        </span>
                                        {stocks.length > 0 && (
                                            <button className="buttonSmall" onClick={() => setStocks([])}>清空</button>
                                        )}
                                    </div>

                                    {/* 添加输入 */}
                                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                                        <input
                                            type="text" className="input"
                                            value={stockInput}
                                            onChange={(e) => setStockInput(e.target.value)}
                                            onKeyPress={(e) => e.key === 'Enter' && addStock()}
                                            placeholder="输入股票代码（如 600519）"
                                            style={{ flex: 1, fontSize: '13px' }}
                                        />
                                        <button className="buttonPrimary" onClick={addStock} style={{ padding: '8px 12px', fontSize: '13px' }}>添加</button>
                                    </div>

                                    {/* 快速预设 */}
                                    <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '10px', alignItems: 'center' }}>
                                        <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)' }}>快速：</span>
                                        {[
                                            { label: '白马股', codes: ['600519','000858','000333','601318'] },
                                            { label: '新能源', codes: ['300750','002475','300059','002594'] },
                                            { label: '银行股', codes: ['600036','601166','600000','601398'] },
                                        ].map(p => (
                                            <button key={p.label} className="buttonSmall"
                                                onClick={() => setStocks([...new Set([...stocks, ...p.codes])])}>
                                                {p.label}
                                            </button>
                                        ))}
                                    </div>

                                    {/* 股票标签 */}
                                    <div className="stockTags" style={{ minHeight: '38px' }}>
                                        {stocks.length === 0
                                            ? <span className="emptyHint">尚未添加股票</span>
                                            : stocks.map(code => (
                                                <span key={code} className="stockTag">
                                                    {code}
                                                    <button className="tagRemove" onClick={() => removeStock(code)}>×</button>
                                                </span>
                                            ))
                                        }
                                    </div>
                                </div>
                            </div>

                            {/* 右：已保存股票池 */}
                            <div style={{ flex: '1 1 220px', minWidth: '200px' }}>
                                <div style={{
                                    ...card, marginBottom: 0, height: '100%',
                                    display: 'flex', flexDirection: 'column'
                                }}>
                                    <div style={cardTitle}>
                                        已保存的股票池
                                        {savedPools.length > 0 && (
                                            <span style={{ fontWeight: '400', color: 'rgba(255,255,255,0.45)', marginLeft: '6px', fontSize: '12px' }}>
                                                ({savedPools.length})
                                            </span>
                                        )}
                                    </div>
                                    {savedPools.length === 0
                                        ? <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.35)', textAlign: 'center', padding: '18px 0', flex: 1 }}>
                                            暂无保存的股票池
                                          </div>
                                        : <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', overflowY: 'auto', maxHeight: '200px' }}>
                                            {savedPools.map(pool => (
                                                <div key={pool.filepath} style={{
                                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                    padding: '8px 10px', background: 'rgba(255,255,255,0.07)',
                                                    borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)',
                                                    cursor: 'pointer',
                                                }} onClick={() => loadPool(pool.name)}>
                                                    <div style={{ flex: 1, minWidth: 0 }}>
                                                        <div style={{ fontWeight: '500', fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                            {pool.name}
                                                        </div>
                                                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', marginTop: '2px' }}>
                                                            {pool.stocks?.length || 0}只 · ¥{(pool.initial_capital || 0).toLocaleString()}
                                                        </div>
                                                    </div>
                                                    <button className="buttonSmall" style={{ marginLeft: '8px', flexShrink: 0 }}
                                                        type="button"
                                                        onClick={(e) => { e.stopPropagation(); deletePool(pool.name) }}>
                                                        删除
                                                    </button>
                                                </div>
                                            ))}
                                          </div>
                                    }
                                </div>
                            </div>
                        </div>

                        {/* 策略配置 & 保存 */}
                        <div style={card}>
                            <div style={cardTitle}>策略配置 & 保存</div>
                            <div className="formGroup">
                                <label className="label">股票池名称</label>
                                <input type="text" className="input" value={poolName}
                                    onChange={(e) => setPoolName(e.target.value)} placeholder="我的选股池" />
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                <div className="formGroup">
                                    <label className="label">策略类型</label>
                                    <select className="select" value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
                                        <option value="adaptive">⚙️ 自适应策略</option>
                                        <option value="trend">📈 趋势跟踪</option>
                                        <option value="mean_reversion">📉 均值回归</option>
                                        <option value="chanlun">📐 缠论</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">风险偏好</label>
                                    <select className="select" value={riskPreference} onChange={(e) => setRiskPreference(e.target.value)}>
                                        <option value="aggressive">🔥 激进</option>
                                        <option value="balanced">⚖️ 均衡</option>
                                        <option value="conservative">🛡️ 保守</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">仓位分配</label>
                                    <select className="select" value={allocationMethod} onChange={(e) => setAllocationMethod(e.target.value)}>
                                        <option value="signal_strength">📊 按信号强度</option>
                                        <option value="equal_weight">⚖️ 等权重</option>
                                        <option value="risk_parity">🛡️ 风险平价</option>
                                        <option value="kelly">📈 凯利公式</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">初始资金</label>
                                    <input type="number" className="input" value={initialCapital}
                                        onChange={(e) => setInitialCapital(e.target.value)} placeholder="1000000" />
                                </div>
                            </div>
                            <div className="actions" style={{ marginTop: '12px' }}>
                                <button className="buttonPrimary" disabled={!canRunAction} onClick={savePool}>
                                    💾 保存股票池
                                </button>
                                <button className="button" onClick={clear} disabled={busy}>清空记录</button>
                            </div>
                        </div>
                    </div>
                )}

                {/* ════════════════════════════════════════
                    模拟仓
                ════════════════════════════════════════ */}
                {mode === 'sim' && (
                    <div>
                        {/* 股票池为空时提示 */}
                        {stocks.length === 0 && (
                            <div style={{
                                background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)',
                                borderRadius: '8px', padding: '10px 14px', marginBottom: '16px',
                                fontSize: '13px', color: 'rgba(251,191,36,0.9)'
                            }}>
                                ⚠️ 当前股票池为空，请先在「管理股票池」中添加股票后再执行调仓
                            </div>
                        )}

                        {/* 账户管理区 */}
                        <div style={{ ...card }}>
                            <div style={cardTitle}>账户管理</div>
                            <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                                <div className="formGroup" style={{ flex: '1 1 160px', margin: 0 }}>
                                    <label className="label" style={{ marginBottom: '6px' }}>选择已有账户</label>
                                    <select className="select" value={selectedAccountId}
                                        onChange={(e) => {
                                            setSelectedAccountId(e.target.value)
                                            if (e.target.value) fetchSimAccount(e.target.value)
                                        }}>
                                        <option value="">-- 选择账户 --</option>
                                        {simAccounts.map(acc => (
                                            <option key={acc.account_id} value={acc.account_id}>
                                                {acc.account_id} (¥{acc.initial_capital.toLocaleString()})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div style={{
                                    flex: '2 1 260px', background: 'rgba(86,185,255,0.07)',
                                    border: '1px solid rgba(86,185,255,0.2)', borderRadius: '8px', padding: '12px'
                                }}>
                                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.55)', marginBottom: '8px' }}>➕ 创建新账户</div>
                                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                                        <input type="text" className="input" value={newAccountId}
                                            onChange={(e) => setNewAccountId(e.target.value)}
                                            placeholder="账户名称" style={{ flex: '2 1 120px' }} />
                                        <input type="number" className="input" value={initialCapital}
                                            onChange={(e) => setInitialCapital(e.target.value)}
                                            placeholder="初始资金" style={{ flex: '1 1 90px' }} />
                                        <button className="buttonPrimary" onClick={createSimAccount}
                                            disabled={busy || !newAccountId.trim()} style={{ flexShrink: 0 }}>
                                            {busy ? '创建中...' : '✨ 创建'}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* 账户面板 */}
                        {simAccountInfo && (
                            <>
                                {/* 资产概览 */}
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px,1fr))', gap: '10px', marginBottom: '14px' }}>
                                    {[
                                        { label: '总资产',   value: `¥${simAccountInfo.statistics.total_equity.toLocaleString()}`,   color: 'rgba(255,255,255,0.95)' },
                                        { label: '现金',     value: `¥${simAccountInfo.statistics.cash.toLocaleString()}`,            color: 'rgba(255,255,255,0.8)'  },
                                        { label: '持仓市值', value: `¥${simAccountInfo.statistics.positions_value.toLocaleString()}`, color: 'rgba(86,185,255,0.9)'   },
                                        {
                                            label: '总盈亏',
                                            value: `${simAccountInfo.statistics.profit_loss >= 0 ? '+' : ''}¥${simAccountInfo.statistics.profit_loss.toLocaleString()} (${simAccountInfo.statistics.profit_loss_pct >= 0 ? '+' : ''}${simAccountInfo.statistics.profit_loss_pct.toFixed(2)}%)`,
                                            color: simAccountInfo.statistics.profit_loss >= 0 ? '#4ade80' : '#f87171'
                                        },
                                    ].map(stat => (
                                        <div key={stat.label} style={{
                                            background: 'rgba(255,255,255,0.07)', borderRadius: '8px',
                                            padding: '10px 12px', border: '1px solid rgba(255,255,255,0.1)'
                                        }}>
                                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', marginBottom: '4px' }}>{stat.label}</div>
                                            <div style={{ fontSize: '13px', fontWeight: '600', color: stat.color }}>{stat.value}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* 持仓列表 */}
                                {simAccountInfo.positions_detail.length > 0 && (
                                    <div style={{ marginBottom: '14px' }}>
                                        <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginBottom: '8px' }}>
                                            当前持仓（{simAccountInfo.positions_detail.length}只）
                                        </div>
                                        <div style={{ display: 'grid', gap: '6px' }}>
                                            {simAccountInfo.positions_detail.map(pos => (
                                                <div key={pos.stock_code} style={{
                                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                    padding: '10px 12px', background: 'rgba(255,255,255,0.07)',
                                                    borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)'
                                                }}>
                                                    <div>
                                                        <span style={{ fontWeight: '600', fontSize: '14px' }}>{pos.stock_code}</span>
                                                        <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.55)', marginLeft: '8px' }}>
                                                            {pos.shares}股 · 成本¥{pos.avg_cost.toFixed(2)} · 现价¥{pos.current_price.toFixed(2)}
                                                        </span>
                                                    </div>
                                                    <div style={{ textAlign: 'right' }}>
                                                        <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.8)' }}>¥{pos.market_value.toLocaleString()}</div>
                                                        <div style={{ fontSize: '12px', fontWeight: '600', color: pos.profit_loss >= 0 ? '#4ade80' : '#f87171' }}>
                                                            {pos.profit_loss >= 0 ? '+' : ''}¥{pos.profit_loss.toFixed(2)} ({pos.profit_loss_pct >= 0 ? '+' : ''}{pos.profit_loss_pct.toFixed(2)}%)
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* 调仓操作栏 */}
                                <div style={{
                                    display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap',
                                    background: 'rgba(255,255,255,0.05)', borderRadius: '10px',
                                    padding: '12px 14px', marginBottom: '14px'
                                }}>
                                    <div className="formGroup" style={{ flex: '1 1 120px', margin: 0 }}>
                                        <label className="label" style={{ marginBottom: '5px' }}>风险偏好</label>
                                        <select className="select" value={riskPreference} onChange={(e) => setRiskPreference(e.target.value)}>
                                            <option value="aggressive">🔥 激进</option>
                                            <option value="balanced">⚖️ 均衡</option>
                                            <option value="conservative">🛡️ 保守</option>
                                        </select>
                                    </div>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '13px', flexShrink: 0 }}>
                                        <input type="checkbox" checked={useSimLlm} onChange={(e) => setUseSimLlm(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                                        🧠 LLM调仓
                                    </label>
                                    <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto', flexWrap: 'wrap' }}>
                                        <button className="buttonPrimary"
                                            disabled={!selectedAccountId || busy || stocks.length === 0}
                                            onClick={() => rebalanceSimAccount(selectedAccountId)}>
                                            {busy ? '调仓中...' : useSimLlm ? '🧠 LLM调仓' : '📊 规则调仓'}
                                        </button>
                                        {selectedAccountId && (
                                            <button className="button" onClick={() => fetchSimAccount(selectedAccountId)} disabled={busy}>刷新</button>
                                        )}
                                    </div>
                                </div>

                                {/* 调仓股票池 */}
                                {stocks.length > 0 && (
                                    <div style={{ marginBottom: '10px' }}>
                                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.45)', marginBottom: '5px' }}>调仓股票池：</div>
                                        <div className="stockTags">
                                            {stocks.map(code => (
                                                <span key={code} className="stockTag">
                                                    {code}
                                                    <button className="tagRemove" onClick={() => removeStock(code)}>×</button>
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* 最近交易记录 */}
                                {simAccountInfo.account.trades?.length > 0 && (
                                    <div>
                                        <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginBottom: '8px' }}>
                                            最近交易（最新 {Math.min(10, simAccountInfo.account.trades.length)} 条）
                                        </div>
                                        <div style={{
                                            background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                                            border: '1px solid rgba(255,255,255,0.1)', overflow: 'hidden'
                                        }}>
                                            {simAccountInfo.account.trades.slice(-10).reverse().map((trade, idx, arr) => (
                                                <div key={idx} style={{
                                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                    padding: '8px 12px', fontSize: '12px',
                                                    borderBottom: idx < arr.length - 1 ? '1px solid rgba(255,255,255,0.07)' : 'none'
                                                }}>
                                                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                                        <span style={{ fontWeight: '600' }}>{trade.stock_code || trade.etf_code}</span>
                                                        <span style={{ color: trade.type === 'buy' ? '#4ade80' : '#f87171', fontWeight: '500' }}>
                                                            {trade.type === 'buy' ? '买入' : '卖出'} {trade.shares}股
                                                        </span>
                                                    </div>
                                                    <div style={{ color: 'rgba(255,255,255,0.65)' }}>
                                                        ¥{trade.price?.toFixed(2) || trade.cost?.toFixed(2) || '0.00'}
                                                        {trade.profit_loss && (
                                                            <span style={{ marginLeft: '6px', fontWeight: '600', color: trade.profit_loss >= 0 ? '#4ade80' : '#f87171' }}>
                                                                {trade.profit_loss >= 0 ? '+' : ''}¥{trade.profit_loss.toFixed(2)}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        {error && <div className="errorText" style={{ marginTop: '12px' }}>{error}</div>}
                    </div>
                )}

                {/* ════════════════════════════════════════
                    组合回测
                ════════════════════════════════════════ */}
                {mode === 'backtest' && (
                    <div>
                        {/* 股票来源 */}
                        <div style={card}>
                            <div style={cardTitle}>股票来源</div>
                            <div style={{ display: 'flex', gap: '10px', marginBottom: '12px', flexWrap: 'wrap' }}>
                                {[
                                    { key: 'manual',   icon: '📋', label: '手动列表',   desc: '使用「管理股票池」中添加的股票' },
                                    { key: 'industry', icon: '🏭', label: '分行业龙头', desc: '按行业自动选取龙头股，无需管理股票池' },
                                ].map(opt => (
                                    <label key={opt.key} style={{
                                        flex: '1 1 160px', cursor: 'pointer', padding: '10px 12px', borderRadius: '8px',
                                        border: universeSource === opt.key ? '2px solid rgba(86,185,255,0.5)' : '1px solid rgba(255,255,255,0.15)',
                                        background: universeSource === opt.key ? 'rgba(86,185,255,0.1)' : 'transparent',
                                        display: 'flex', alignItems: 'flex-start', gap: '8px'
                                    }}>
                                        <input type="radio" name="universeSource" checked={universeSource === opt.key}
                                            onChange={() => setUniverseSource(opt.key)} style={{ marginTop: '3px' }} />
                                        <div>
                                            <div style={{ fontSize: '13px', fontWeight: '600' }}>{opt.icon} {opt.label}</div>
                                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)', marginTop: '3px' }}>{opt.desc}</div>
                                        </div>
                                    </label>
                                ))}
                            </div>

                            {universeSource === 'manual' && (
                                <div className="stockTags" style={{ minHeight: '32px' }}>
                                    {stocks.length === 0
                                        ? <span className="emptyHint">股票池为空 — 请先在「管理股票池」中添加股票</span>
                                        : stocks.map(code => (
                                            <span key={code} className="stockTag">
                                                {code}
                                                <button className="tagRemove" onClick={() => removeStock(code)}>×</button>
                                            </span>
                                        ))
                                    }
                                </div>
                            )}

                            {universeSource === 'industry' && (
                                <div>
                                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
                                        <select className="select" value=""
                                            onChange={(e) => {
                                                const v = e.target.value
                                                if (v && !industryList.includes(v)) setIndustryList([...industryList, v])
                                                e.target.value = ''
                                            }} style={{ minWidth: '140px' }}>
                                            <option value="">选择行业...</option>
                                            {industryNames.map(name => <option key={name} value={name}>{name}</option>)}
                                        </select>
                                        <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.65)' }}>每行业龙头</span>
                                        <input type="number" className="input" min={1} max={10} value={leadersPerIndustry}
                                            onChange={(e) => setLeadersPerIndustry(Math.max(1, parseInt(e.target.value, 10) || 1))}
                                            style={{ width: '52px' }} />
                                        <button type="button" className="buttonSmall"
                                            onClick={() => {
                                                if (industryList.length === 0) return
                                                poolApi.industryLeaders({ industries: industryList.join(','), per: leadersPerIndustry })
                                                    .then(data => { if (data?.result) setIndustryLeadersPreview({ result: data.result, code_to_industry: data.code_to_industry || {} }) })
                                                    .catch(() => {})
                                            }} disabled={industryList.length === 0}>
                                            预览龙头
                                        </button>
                                    </div>
                                    <div className="stockTags" style={{ minHeight: '28px' }}>
                                        {industryList.length === 0
                                            ? <span className="emptyHint">从下拉框选择行业（可多选）</span>
                                            : industryList.map(name => (
                                                <span key={name} className="stockTag">
                                                    {name}
                                                    <button className="tagRemove" onClick={() => setIndustryList(industryList.filter(x => x !== name))}>×</button>
                                                </span>
                                            ))
                                        }
                                    </div>
                                    {industryLeadersPreview && (
                                        <div style={{ marginTop: '6px', fontSize: '12px', color: 'rgba(255,255,255,0.65)' }}>
                                            龙头预览：{industryLeadersPreview.result?.join(', ') || '-'}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* 策略参数 */}
                        <div style={card}>
                            <div style={cardTitle}>策略参数</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                <div className="formGroup">
                                    <label className="label">策略类型</label>
                                    <select className="select" value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
                                        <option value="adaptive">⚙️ 自适应策略（推荐）</option>
                                        <option value="trend">📈 趋势跟踪</option>
                                        <option value="mean_reversion">📉 均值回归</option>
                                        <option value="chanlun">📐 缠论</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">风险偏好</label>
                                    <select className="select" value={riskPreference} onChange={(e) => setRiskPreference(e.target.value)}>
                                        <option value="aggressive">🔥 激进</option>
                                        <option value="balanced">⚖️ 均衡</option>
                                        <option value="conservative">🛡️ 保守</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">初始资金</label>
                                    <input type="number" className="input" value={initialCapital} onChange={(e) => setInitialCapital(e.target.value)} />
                                </div>
                                <div className="formGroup">
                                    <label className="label">回测天数</label>
                                    <select className="select" value={backtestDays} onChange={(e) => setBacktestDays(e.target.value)}>
                                        <option value="60">60天（约1季度）</option>
                                        <option value="126">126天（约半年）</option>
                                        <option value="252">252天（约1年）</option>
                                        <option value="504">504天（约2年）</option>
                                        <option value="756">756天（约3年）</option>
                                    </select>
                                </div>
                                <div className="formGroup">
                                    <label className="label">调仓周期（天）</label>
                                    <input type="number" className="input" min={1} max={30} value={rebalanceInterval}
                                        onChange={(e) => setRebalanceInterval(e.target.value)} />
                                </div>
                                <div className="formGroup">
                                    <label className="label">无前视模式</label>
                                    <select className="select" value={noLookahead ? '1' : '0'} onChange={(e) => setNoLookahead(e.target.value === '1')}>
                                        <option value="0">关闭</option>
                                        <option value="1">开启（推荐）</option>
                                    </select>
                                </div>
                            </div>
                            {noLookahead && (
                                <div className="formGroup" style={{ marginTop: '8px' }}>
                                    <label className="label">回测起始日期（无前视模式必填）</label>
                                    <input type="date" className="input" value={backtestStartDate}
                                        onChange={(e) => setBacktestStartDate(e.target.value)} />
                                </div>
                            )}
                        </div>

                        {/* LLM 信号设置 */}
                        <div style={{
                            ...card,
                            background: usePoolLlmSignals
                                ? 'linear-gradient(135deg, rgba(139,92,246,0.18), rgba(168,85,247,0.26))'
                                : card.background,
                            border: usePoolLlmSignals
                                ? '2px solid rgba(168,85,247,0.45)'
                                : card.border,
                        }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', marginBottom: usePoolLlmSignals ? '12px' : '0' }}>
                                <input type="checkbox" checked={usePoolLlmSignals}
                                    onChange={(e) => setUsePoolLlmSignals(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                                <span style={{ fontWeight: '600', fontSize: '14px' }}>🧠 使用LLM生成交易信号</span>
                                {!usePoolLlmSignals && (
                                    <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.45)', marginLeft: '2px' }}>（当前：规则算法，速度快）</span>
                                )}
                            </label>
                            {usePoolLlmSignals && (
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                    <div>
                                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.65)', lineHeight: '1.7', marginBottom: '10px' }}>
                                            <div>✓ DeepSeek大模型分析技术指标</div>
                                            <div>✓ 不同策略使用专属 Prompt</div>
                                            <div style={{ color: '#fca5a5' }}>⚠️ 速度较慢（多次LLM调用）</div>
                                        </div>
                                        <div className="formGroup" style={{ margin: 0 }}>
                                            <label className="label" style={{ fontSize: '12px' }}>采样频率</label>
                                            <select className="select" value={poolLlmSampleRate}
                                                onChange={(e) => setPoolLlmSampleRate(e.target.value)} style={{ fontSize: '12px' }}>
                                                <option value="1">每天</option>
                                                <option value="3">每3天</option>
                                                <option value="5">每5天（推荐）</option>
                                                <option value="10">每10天</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div>
                                        <label style={{ display: 'flex', alignItems: 'flex-start', gap: '6px', cursor: 'pointer' }}>
                                            <input type="checkbox" checked={poolHighWinRateMode}
                                                onChange={(e) => setPoolHighWinRateMode(e.target.checked)}
                                                style={{ width: '16px', height: '16px', marginTop: '2px' }} />
                                            <div>
                                                <div style={{ fontSize: '13px', fontWeight: '600', color: poolHighWinRateMode ? '#f87171' : 'rgba(255,255,255,0.9)' }}>
                                                    🎯 高胜率模式
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.55)', marginTop: '4px', lineHeight: '1.6' }}>
                                                    {poolHighWinRateMode ? (
                                                        <>
                                                            <div style={{ color: '#fca5a5' }}>✓ 多指标共振确认</div>
                                                            <div style={{ color: '#fca5a5' }}>✓ 严格趋势对齐</div>
                                                            <div style={{ color: '#fca5a5' }}>✓ 2:1 盈亏比</div>
                                                        </>
                                                    ) : '严格入场条件，提高胜率但减少交易次数'}
                                                </div>
                                            </div>
                                        </label>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 执行按钮 */}
                        <div className="actions">
                            <button className="buttonPrimary"
                                style={{ background: usePoolLlmSignals ? '#6a5acd' : undefined }}
                                disabled={!canRunBacktest || busy}
                                onClick={runBacktest}>
                                {busy ? '回测中...' : usePoolLlmSignals ? '🧠 运行LLM组合回测' : '📈 运行动态选股回测'}
                            </button>
                            <button className="button" onClick={clear} disabled={busy}>清空记录</button>
                        </div>

                        {/* 进度 */}
                        {busy && (
                            <div className="progressContainer">
                                <div className="progressTitle">
                                    <div className="spinner"></div>
                                    <span>正在进行组合回测{usePoolLlmSignals ? '（LLM模式）' : ''}...</span>
                                    {usePoolLlmSignals && (
                                        <button type="button" className="button" onClick={stopPoolBacktest} disabled={poolStopping}
                                            style={{ marginLeft: '12px', padding: '6px 12px', fontSize: '12px' }}>
                                            {poolStopping ? '正在停止...' : '停止回测'}
                                        </button>
                                    )}
                                </div>
                                <div className="progressBar">
                                    <div className="progressFill" style={{ width: `${backtestProgress}%` }}></div>
                                </div>
                                <div className="progressInfo">
                                    <span>{Math.round(backtestProgress)}% 完成</span>
                                    <span>
                                        {universeSource === 'manual' ? `${stocks.length}只股票` : `分行业龙头(${industryList.length}行业)`} × {backtestDays}天
                                    </span>
                                </div>
                                {usePoolLlmSignals && (
                                    <div className="liveDecisionsPanel">
                                        {currentLlmStatus && <div className="llmStatus">{currentLlmStatus}</div>}
                                        {liveDecisions.length > 0 && (
                                            <div className="liveDecisionsList">
                                                <div className="liveDecisionsTitle">🧠 LLM组合决策 ({liveDecisions.length})</div>
                                                <div className="liveDecisionsScroll">
                                                    {[...liveDecisions].slice(-10).reverse().map((d, idx) => (
                                                        d.type === 'portfolio' ? (
                                                            <div key={idx} className="liveDecisionItem portfolioDecision">
                                                                <span className="ldDate">{d.date}</span>
                                                                <span className="ldPortfolio">📊 组合决策#{d.call_count}</span>
                                                                <div className="ldPortfolioDetail">
                                                                    <span className="ldAnalysis">{d.analysis}</span>
                                                                    {d.priority_buy?.length > 0 && <span className="ldBuyList">📈买: {d.priority_buy.join(', ')}</span>}
                                                                    {d.priority_sell?.length > 0 && <span className="ldSellList">📉卖: {d.priority_sell.join(', ')}</span>}
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div key={idx} className={`liveDecisionItem ${d.type === 'trade' ? (d.action === 'BUY' ? 'tradeBuy' : 'tradeSell') : ''}`}>
                                                                <span className="ldDate">{d.date}</span>
                                                                <span className="ldCode">{d.stock_code}</span>
                                                                <span className={`ldAction ${d.action === 'BUY' ? 'actionBuy' : d.action === 'SELL' ? 'actionSell' : ''}`}>
                                                                    {d.type === 'trade' ? (d.action === 'BUY' ? '🟢买入' : '🔴卖出') : d.action}
                                                                </span>
                                                                {d.type === 'trade' && <span className="ldShares">{d.shares}股</span>}
                                                                {d.confidence && <span className="ldConf">置信度:{d.confidence}</span>}
                                                                <span className="ldReason" title={d.reason}>{d.reason?.substring(0, 30)}...</span>
                                                            </div>
                                                        )
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {error && mode !== 'sim' && <div className="errorText">{error}</div>}
            </div>

            <div className="panelList" aria-label="选股池 结果列表">
                {items.length === 0 ? (
                    <div className="empty">暂无已保存结果</div>
                ) : (
                    items.map((it) => (
                        <article key={it.id} className="card backtestResult">
                            <div className="cardMeta">{formatTs(it.ts)}</div>
                            <pre className="cardText">{typeof it === 'string' ? it : (it.text || it)}</pre>

                            {/* 回测图表 */}
                            {it.chart && (
                                <div className="chartContainer">
                                    <img src={it.chart.startsWith('data:') ? it.chart : `data:image/png;base64,${it.chart}`} alt="回测图表" className="chartImage" />
                                </div>
                            )}

                            {/* 交易决策记录 */}
                            {it.trades && it.trades.length > 0 && (
                                <div className="tradesSection">
                                    <div className="sectionTitle">📋 交易决策记录 ({it.trades.length}笔)</div>
                                    <div className="tradesTable">
                                        <div className="tradesHeader">
                                            <span>日期</span>
                                            <span>股票</span>
                                            <span>操作</span>
                                            <span>数量</span>
                                            <span>价格</span>
                                            <span>金额</span>
                                            <span>决策原因</span>
                                        </div>
                                        {it.trades.slice(0, 50).map((trade, idx) => (
                                            <div key={idx} className={`tradeRow ${trade.action === 'BUY' ? 'tradeBuy' : 'tradeSell'}`}>
                                                <span>{trade.date}</span>
                                                <span className="tradeCode">{trade.stock_code}</span>
                                                <span className={trade.action === 'BUY' ? 'actionBuy' : 'actionSell'}>
                                                    {trade.action === 'BUY' ? '🟢买入' : '🔴卖出'}
                                                </span>
                                                <span>{trade.shares}股</span>
                                                <span>¥{trade.price}</span>
                                                <span>¥{trade.amount?.toLocaleString()}</span>
                                                <span className="tradeReason" title={trade.reason}>{trade.reason}</span>
                                            </div>
                                        ))}
                                        {it.trades.length > 50 && (
                                            <div className="tradesMore">... 还有 {it.trades.length - 50} 条交易记录</div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* 持仓变动快照 */}
                            {it.positions && it.positions.length > 0 && (
                                <div className="positionsSection">
                                    <div className="sectionTitle">📈 持仓变动时间线 ({it.positions.length}个采样点)</div>
                                    <div className="positionsTimeline">
                                        {it.positions.map((snap, idx) => {
                                            const prevSnap = idx > 0 ? it.positions[idx - 1] : null
                                            const positionCodes = Object.keys(snap.positions || {})
                                            const prevCodes = prevSnap ? Object.keys(prevSnap.positions || {}) : []
                                            const newCodes = positionCodes.filter(c => !prevCodes.includes(c))
                                            const exitCodes = prevCodes.filter(c => !positionCodes.includes(c))
                                            const cashRatio = snap.total_value > 0 ? ((snap.cash / snap.total_value) * 100).toFixed(1) : 100

                                            return (
                                                <div key={idx} className="positionSnap">
                                                    <div className="snapHeader">
                                                        <span className="snapDate">{snap.date}</span>
                                                        <span className={`snapReturn ${snap.cumulative_return >= 0 ? 'positive' : 'negative'}`}>
                                                            累计: {snap.cumulative_return >= 0 ? '+' : ''}{snap.cumulative_return}%
                                                        </span>
                                                        <span className={`snapDailyReturn ${snap.daily_return >= 0 ? 'positive' : 'negative'}`}>
                                                            日收益: {snap.daily_return >= 0 ? '+' : ''}{snap.daily_return}%
                                                        </span>
                                                        <span className="snapValue">总值: ¥{snap.total_value?.toLocaleString()}</span>
                                                        <span className="snapCash">现金: {cashRatio}%</span>
                                                    </div>

                                                    {/* 持仓变动提示 */}
                                                    {(newCodes.length > 0 || exitCodes.length > 0) && (
                                                        <div className="snapChanges">
                                                            {newCodes.length > 0 && (
                                                                <span className="changeNew">🟢 新建仓: {newCodes.join(', ')}</span>
                                                            )}
                                                            {exitCodes.length > 0 && (
                                                                <span className="changeExit">🔴 已清仓: {exitCodes.join(', ')}</span>
                                                            )}
                                                        </div>
                                                    )}

                                                    <div className="snapPositions">
                                                        {positionCodes.length === 0 ? (
                                                            <span className="emptyPosition">💵 空仓观望中</span>
                                                        ) : (
                                                            positionCodes.map(code => {
                                                                const pos = snap.positions[code]
                                                                const isNew = newCodes.includes(code)
                                                                return (
                                                                    <span key={code} className={`positionChip ${pos.profit_loss_pct >= 0 ? 'profitUp' : 'profitDown'} ${isNew ? 'newPosition' : ''}`}>
                                                                        {isNew && '🆕'}{code}: {pos.weight}%
                                                                        <small>({pos.profit_loss_pct >= 0 ? '+' : ''}{pos.profit_loss_pct}%)</small>
                                                                    </span>
                                                                )
                                                            })
                                                        )}
                                                    </div>
                                                </div>
                                            )
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* 最终持仓 */}
                            {it.finalPositions && it.finalPositions.length > 0 && (
                                <div className="finalPositionsSection">
                                    <div className="sectionTitle">📦 最终持仓明细</div>
                                    <div className="finalPositionsList">
                                        {it.finalPositions.map((pos, idx) => (
                                            <div key={idx} className={`finalPositionItem ${pos.profit_loss_pct >= 0 ? 'profitUp' : 'profitDown'}`}>
                                                <span className="posCode">{pos.stock_code}</span>
                                                <span className="posShares">{pos.shares}股</span>
                                                <span className="posValue">¥{pos.market_value?.toLocaleString()}</span>
                                                <span className="posWeight">权重{pos.weight}%</span>
                                                <span className={`posPnl ${pos.profit_loss_pct >= 0 ? 'positive' : 'negative'}`}>
                                                    {pos.profit_loss_pct >= 0 ? '+' : ''}{pos.profit_loss_pct}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </article>
                    ))
                )}
            </div>
        </section>
    )
}
