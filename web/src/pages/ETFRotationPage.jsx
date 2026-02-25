import React, { useMemo, useState, useEffect, useRef } from 'react'
import { usePersistedList } from '../hooks/usePersistedList.js'
import { formatTs } from '../utils/format.js'
import * as etfRotationApi from '../api/etfRotation.js'
import * as etfSimApi from '../api/etfSim.js'
import { TradesHistory } from '../components/TradesHistory.jsx'

export default function ETFRotationPage({ user }) {
    const { items, addItem, clear, count } = usePersistedList('my_stock:etf_rotation', 10, user?.user_id)
    const [mode, setMode] = useState('backtest')  // backtest | sim | ai

    // ETF 池与输入
    const [etfCodes, setEtfCodes] = useState(['510300', '510500', '159915'])
    const [etfInput, setEtfInput] = useState('')

    // 回测/策略参数
    const [initialCapital, setInitialCapital] = useState('100000')
    const [days, setDays] = useState('252')
    const [rotationInterval, setRotationInterval] = useState('5')
    const [rebalanceInterval, setRebalanceInterval] = useState('')  // 空表示无再平衡
    const [lookbackDays, setLookbackDays] = useState('20')
    const [topK, setTopK] = useState('1')
    const [minScoreThreshold, setMinScoreThreshold] = useState('20')
    const [useAi, setUseAi] = useState(false)
    const [positionStrategy, setPositionStrategy] = useState('equal')  // equal=等权重, kelly=凯利公式

    // 得分权重（百分比，会归一化）
    const [scoreWeights, setScoreWeights] = useState({
        momentum: 65,
        rsi: 10,
        ma: 15,
        macd: 10
    })
    const weightsSum = scoreWeights.momentum + scoreWeights.rsi + scoreWeights.ma + scoreWeights.macd

    function updateWeight(key, value) {
        const numValue = parseFloat(value) || 0
        setScoreWeights(prev => ({
            ...prev,
            [key]: Math.max(0, Math.min(100, numValue))
        }))
    }

    function resetWeights() {
        setScoreWeights({ momentum: 65, rsi: 10, ma: 15, macd: 10 })
    }

    /** 归一化权重（转换为 0–1 比例） */
    function getNormalizedWeights() {
        if (weightsSum <= 0) {
            return { momentum: 0.65, rsi: 0.1, ma: 0.15, macd: 0.1 }
        }
        return {
            momentum: scoreWeights.momentum / 100,
            rsi: scoreWeights.rsi / 100,
            ma: scoreWeights.ma / 100,
            macd: scoreWeights.macd / 100
        }
    }

    // 状态
    const [busy, setBusy] = useState(false)
    const [stopping, setStopping] = useState(false)
    const [error, setError] = useState('')
    const [backtestProgress, setBacktestProgress] = useState(0)
    const [backtestStreamInfo, setBacktestStreamInfo] = useState({ current: 0, total: 0, date: '', elapsed: 0 })
    const [liveDecisions, setLiveDecisions] = useState([])
    const [defaultEtfs, setDefaultEtfs] = useState([])
    const backtestEventSourceRef = useRef(null)
    const backtestSessionIdRef = useRef(null)

    // 调仓建议
    const [suggestion, setSuggestion] = useState(null)
    const [suggestionLoading, setSuggestionLoading] = useState(false)

    // AI 分析
    const [aiResult, setAiResult] = useState(null)
    const [aiLoading, setAiLoading] = useState(false)

    // 模拟盘
    const [simAccountId, setSimAccountId] = useState('')
    const [simAccountIdInput, setSimAccountIdInput] = useState('')
    const [simInitialCapital, setSimInitialCapital] = useState('100000')
    const [simAccount, setSimAccount] = useState(null)
    const [simAccountLoading, setSimAccountLoading] = useState(false)
    const [simAccounts, setSimAccounts] = useState([])

    // 加载默认 ETF 列表
    useEffect(() => {
        etfRotationApi.getDefaultEtfs()
            .then(data => { if (data?.result) setDefaultEtfs(data.result) })
            .catch(e => console.error('加载默认ETF列表失败:', e))
    }, [])

    function addEtf() {
        const code = etfInput.trim().toUpperCase()
        if (!code) return
        if (mode === 'sim' && simAccountId) {
            setBusy(true)
            setError('')
            etfSimApi.etfPoolAdd(simAccountId, code)
                .then(data => {
                    if (data?.result?.etf_pool) {
                        setEtfCodes(data.result.etf_pool)
                        setEtfInput('')
                    } else {
                        setError(data?.result?.message || data?.detail || '添加ETF失败')
                    }
                })
                .then(() => loadSimAccount(simAccountId))
                .catch(e => setError(String(e?.message || e)))
                .finally(() => setBusy(false))
        } else {
            if (!etfCodes.includes(code)) {
                setEtfCodes([...etfCodes, code])
                setEtfInput('')
            }
        }
    }

    function removeEtf(code) {
        if (mode === 'sim' && simAccountId) {
            if (!confirm(`确定要从ETF池移除 ${code} 吗？\n如果账户持有该ETF，将自动清仓。`)) return
            setBusy(true)
            setError('')
            etfSimApi.etfPoolRemove(simAccountId, code, true)
                .then(data => {
                    if (data?.result?.success && data?.result?.etf_pool) {
                        setEtfCodes(data.result.etf_pool)
                    } else {
                        setError(data?.result?.message || '移除ETF失败')
                    }
                })
                .then(() => loadSimAccount(simAccountId))
                .catch(e => setError(String(e?.message || e)))
                .finally(() => setBusy(false))
        } else {
            setEtfCodes(etfCodes.filter(c => c !== code))
        }
    }

    const canRun = etfCodes.length > 0 && !busy

    // ---------- 调仓建议（回测/通用） ----------
    async function getSuggestion() {
        if (etfCodes.length === 0) {
            setError('请先添加ETF代码')
            return
        }
        setSuggestionLoading(true)
        setError('')
        try {
            const data = await etfRotationApi.suggestion({
                etf_codes: etfCodes,
                lookback_days: parseInt(lookbackDays) || 20,
                top_k: parseInt(topK) || 1,
                score_weights: getNormalizedWeights(),
                min_score_threshold: parseFloat(minScoreThreshold) || 20
            })
            const result = data?.result
            if (result?.error) throw new Error(result.error)
            setSuggestion(result)
        } catch (e) {
            console.error('[ETF轮动] 获取调仓建议失败:', e)
            setError(String(e?.message || e))
        } finally {
            setSuggestionLoading(false)
        }
    }

    // ---------- AI 轮动分析 ----------
    async function runAIRotation() {
        if (etfCodes.length === 0) {
            setError('请先添加ETF代码')
            return
        }
        setAiLoading(true)
        setError('')
        setAiResult(null)
        try {
            const data = await etfRotationApi.ai({
                etf_codes: etfCodes,
                lookback_days: parseInt(lookbackDays) || 20,
                top_k: parseInt(topK) || 1,
                min_score_threshold: parseFloat(minScoreThreshold) || 20,
                score_weights: getNormalizedWeights()
            })
            setAiResult(data?.result)
        } catch (e) {
            console.error('[ETF AI轮动] 错误:', e)
            setError(String(e?.message || e))
        } finally {
            setAiLoading(false)
        }
    }

    // ---------- 回测：停止 ----------
    function stopBacktest() {
        if (stopping || !backtestSessionIdRef.current) return
        setStopping(true)
        etfRotationApi.backtestStop(backtestSessionIdRef.current)
            .finally(() => setStopping(false))
        // 不关闭 EventSource：等待后端检测停止后发送 complete（含局部结果），否则前端会一直 busy
    }

    // ---------- 回测：处理结果并写入历史 ----------
    function processBacktestResult(result, chartUrl) {
        const formatTrades = (trades) => {
            if (!trades || trades.length === 0) return '无交易记录'
            return trades.map(t => {
                const action = t.type === 'buy' ? '买入' : '卖出'
                const date = typeof t.date === 'string' ? t.date : new Date(t.date).toLocaleDateString('zh-CN')
                const reason = t.reason ? ` | ${t.reason}` : ''
                const score = t.score != null ? ` | 得分:${t.score.toFixed(1)}` : ''
                return `  ${action} ${date} ${t.etf_code} @¥${t.price?.toFixed(2)}${reason}${score}`
            }).join('\n')
        }
        const warningText = result.warning ? `\n⚠️ 注意：${result.warning}\n` : ''
        const abortedText = result.aborted ? `\n⚠️ ${result.aborted_message || '回测已停止'}\n` : ''
        const requestedDaysText = result.requested_days != null && result.requested_days !== result.trading_days
            ? `（请求${result.requested_days}天，实际${result.trading_days}天）`
            : ''
        const text = `【ETF轮动策略回测】${warningText}${abortedText}
📊 ETF池：${etfCodes.join(', ')}
回测周期：${result.start_date || '-'} ~ ${result.end_date || '-'}（共${result.trading_days || '-'}个交易日${requestedDaysText}）
轮动间隔：${result.rotation_interval ?? rotationInterval}天 | 回看天数：${lookbackDays}天 | 持仓数量：Top-${result.top_k ?? topK}
${result.rebalance_interval != null ? `再平衡间隔：${result.rebalance_interval}天 | ` : '无再平衡 | '}最低得分阈值：${result.min_score_threshold ?? minScoreThreshold}
初始资金：¥${(result.initial_capital ?? initialCapital).toLocaleString()}
回测耗时：${result.backtest_time ?? '-'}秒

📈 回测结果：
💰 收益指标：  总收益率：${result.total_return ?? 0}%
  年化收益率：${result.annual_return ?? 0}%（基于${result.trading_days ?? '-'}交易日年化）
  最终资金：¥${(result.final_capital ?? result.initial_capital ?? initialCapital).toLocaleString()}

⚠️ 风险指标：  最大回撤：${result.max_drawdown ?? 0}%
  夏普比率：${result.sharpe_ratio ?? 0} ${result.sharpe_ratio > 2 ? '(优秀)' : result.sharpe_ratio > 1 ? '(良好)' : '(一般)'}

📋 交易统计：  交易次数：${result.total_trades ?? 0}笔  胜率：${result.win_rate ?? 0}%

${(result.etf_pnl_summary && result.etf_pnl_summary.length > 0)
    ? `📊 各ETF总盈亏：\n${result.etf_pnl_summary.map(s =>
        `  ${s.etf_code}(${s.etf_name}) 盈亏: ¥${s.pnl.toLocaleString()} (${s.pnl_pct >= 0 ? '+' : ''}${s.pnl_pct}%) | 买入成本: ¥${s.total_cost.toLocaleString()} | 卖出收入: ¥${s.total_revenue.toLocaleString()} | 期末持仓市值: ¥${s.holdings_value.toLocaleString()}`
    ).join('\n')}\n\n`
    : ''}📌 最近交易记录：
${formatTrades(result.trades_summary || result.trades || [])}

⚠️ 风险提示： 历史回测不代表未来收益；ETF轮动策略基于技术指标得分选择ETF；实盘存在滑点和手续费；建议小资金测试验证。`
        addItem({ text, chart: chartUrl || null })
    }

    // ---------- 回测：运行（SSE 流式） ----------
    async function runBacktest() {
        if (!canRun) return
        setBusy(true)
        setError('')
        setBacktestProgress(0)
        setBacktestStreamInfo({ current: 0, total: 0, date: '', elapsed: 0 })
        setLiveDecisions([])
        const sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID() : `etf-${Date.now()}-${Math.random().toString(36).slice(2)}`
        backtestSessionIdRef.current = sessionId
        const params = new URLSearchParams({
            session_id: sessionId,
            etf_codes: etfCodes.join(','),
            initial_capital: String(parseFloat(initialCapital) || 100000),
            days: String(parseInt(days) || 252),
            rotation_interval: String(parseInt(rotationInterval) || 5),
            lookback_days: String(parseInt(lookbackDays) || 20),
            commission_rate: '0.0003',
            slippage: '0.001',
            top_k: String(parseInt(topK) || 1),
            min_score_threshold: String(parseFloat(minScoreThreshold) || 20),
            use_ai: useAi ? 'true' : 'false',
            position_strategy: positionStrategy || 'equal'
        })
        if (rebalanceInterval !== '') {
            const ri = parseInt(rebalanceInterval, 10)
            if (!isNaN(ri)) params.set('rebalance_interval', String(ri))
        }
        const weights = getNormalizedWeights()
        if (weights && Object.keys(weights).length) {
            params.set('score_weights', JSON.stringify(weights))
        }
        const streamUrl = `/api/etf-rotation/backtest/stream?${params.toString()}`
        const eventSource = new EventSource(streamUrl)
        backtestEventSourceRef.current = eventSource
        const receivedCompleteRef = { current: false }

        eventSource.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data)
                if (msg.type === 'progress') {
                    setBacktestProgress(msg.data?.percent ?? 0)
                    setBacktestStreamInfo({
                        current: msg.data?.current ?? 0,
                        total: msg.data?.total ?? 0,
                        date: msg.data?.date ?? '',
                        elapsed: msg.data?.elapsed ?? 0
                    })
                } else if (msg.type === 'etf_ai_start') {
                    setLiveDecisions(prev => [...prev, {
                        type: 'etf_ai_start',
                        date: msg.data?.date ?? '',
                        etf_count: msg.data?.etf_count ?? 0,
                        technical_top: msg.data?.technical_top ?? [],
                    }])
                } else if (msg.type === 'etf_decision') {
                    setLiveDecisions(prev => [...prev, {
                        type: 'portfolio',
                        date: msg.data?.date ?? '',
                        priority_buy: msg.data?.priority_buy ?? [],
                        priority_sell: msg.data?.priority_sell ?? [],
                        target_etfs: msg.data?.target_etfs ?? [],
                        etf_scores: msg.data?.etf_scores ?? {},
                        use_ai: msg.data?.use_ai ?? false,
                        analysis: msg.data?.reason ?? '',
                    }])
                } else if (msg.type === 'etf_trade') {
                    setLiveDecisions(prev => [...prev, {
                        type: 'trade',
                        date: msg.data?.date ?? '',
                        stock_code: msg.data?.stock_code ?? '',
                        action: msg.data?.action ?? '',
                        shares: msg.data?.shares ?? 0,
                        price: msg.data?.price ?? 0,
                        reason: msg.data?.reason ?? '',
                    }])
                } else if (msg.type === 'complete') {
                    receivedCompleteRef.current = true
                    backtestEventSourceRef.current = null
                    eventSource.close()
                    setStopping(false)
                    setBacktestProgress(100)
                    const data = msg.data || {}
                    const result = data.result
                    if (result?.error) {
                        setError(result.error)
                    } else if (result) {
                        if (result.aborted) {
                            setError(result.aborted_message || '回测已停止，以下为局部结果')
                        }
                        processBacktestResult(result, data.chart)
                    } else {
                        setError('回测结果为空')
                    }
                    setBusy(false)
                } else if (msg.type === 'error') {
                    receivedCompleteRef.current = true
                    eventSource.close()
                    backtestEventSourceRef.current = null
                    setStopping(false)
                    setError(msg.data?.message || '回测出错')
                    setBusy(false)
                }
            } catch (e) {
                console.error('[ETF轮动] SSE 解析错误:', e)
            }
        }

        eventSource.onerror = () => {
            backtestEventSourceRef.current = null
            eventSource.close()
            setStopping(false)
            if (!receivedCompleteRef.current) {
                setError('连接中断，请重试')
            }
            setBusy(false)
        }
    }

    // ---------- 模拟盘：账户 ----------
    async function loadSimAccounts() {
        try {
            const data = await etfSimApi.getAccounts()
            if (data?.result) setSimAccounts(data.result)
        } catch (e) {
            console.error('加载账户列表失败:', e)
        }
    }

    async function createSimAccount() {
        const accountId = simAccountIdInput.trim()
        if (!accountId) {
            setError('请输入账户ID')
            return
        }
        setBusy(true)
        setError('')
        try {
            const data = await etfSimApi.createAccount({
                account_id: accountId,
                initial_capital: parseFloat(simInitialCapital) || 100000
            })
            if (data?.result) {
                setSimAccountId(accountId)
                setSimAccountIdInput('')
                await loadSimAccount(accountId)
                await loadSimAccounts()
            } else {
                setError('创建账户失败')
            }
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    async function loadSimAccount(accountId) {
        if (!accountId) return
        setSimAccountLoading(true)
        setError('')
        try {
            const data = await etfSimApi.getAccount(accountId)
            if (data?.result) {
                setSimAccount(data.result)
                const pool = data.result?.account?.etf_pool
                setEtfCodes(Array.isArray(pool) ? pool : [])
            }
        } catch (e) {
            console.error('加载账户失败:', e)
            setError(String(e?.message || e))
        } finally {
            setSimAccountLoading(false)
        }
    }

    async function deleteSimAccount() {
        if (!simAccountId) {
            setError('请先选择要删除的账户')
            return
        }
        if (!confirm(`确定要删除账户「${simAccountId}」吗？此操作不可恢复！`)) return
        setBusy(true)
        setError('')
        try {
            const data = await etfSimApi.deleteAccount(simAccountId)
            if (data?.result?.success) {
                setSimAccountId('')
                setSimAccount(null)
                setEtfCodes([])
                await loadSimAccounts()
            } else {
                setError('删除账户失败')
            }
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    async function executeSimAutoTrade() {
        if (!simAccountId) {
            setError('请先创建或选择账户')
            return
        }
        setBusy(true)
        setError('')
        try {
            const data = await etfSimApi.autoTrade({
                account_id: simAccountId,
                etf_codes: etfCodes.length > 0 ? etfCodes : (simAccount?.account?.etf_pool || []),
                lookback_days: parseInt(lookbackDays, 10) || 20,
                top_k: parseInt(topK, 10) || 1,
                score_weights: getNormalizedWeights(),
                min_score_threshold: parseFloat(minScoreThreshold) || 20,
                rotation_interval: rotationInterval === '' ? null : parseInt(rotationInterval, 10),
                rebalance_interval: rebalanceInterval === '' ? null : parseInt(rebalanceInterval, 10),
                use_ai: useAi
            })
            if (data?.result) {
                await loadSimAccount(simAccountId)
                const executed = data.result.trades_executed?.length ?? 0
                const errs = data.result.errors && data.result.errors.length > 0 ? `\n\n错误：\n${data.result.errors.join('\n')}` : ''
                const warns = data.result.warnings?.length ? `\n\n警告：\n${data.result.warnings.join('\n')}` : ''
                alert(`自动交易完成。\n执行交易：${executed}笔${errs}${warns}`)
            } else {
                const errMsg = data?.detail || data?.error || '自动交易失败'
                setError(errMsg)
                alert(`自动调仓失败：\n${errMsg}`)
            }
        } catch (e) {
            const errMsg = String(e?.message || e)
            setError(errMsg)
            alert(`自动调仓失败：\n${errMsg}`)
        } finally {
            setBusy(false)
        }
    }

    async function getSimSuggestion() {
        if (!simAccountId) {
            setError('请先选择账户')
            return
        }
        const etfCodesToUse = etfCodes.length > 0 ? etfCodes : (simAccount?.account?.etf_pool || [])
        if (etfCodesToUse.length === 0) {
            setError('ETF池为空，请先添加ETF到账户池中')
            setSuggestionLoading(false)
            return
        }
        setSuggestionLoading(true)
        setError('')
        try {
            const q = new URLSearchParams({
                etf_codes: etfCodesToUse.join(','),
                lookback_days: parseInt(lookbackDays, 10) || 20,
                top_k: parseInt(topK, 10) || 1,
                min_score_threshold: parseFloat(minScoreThreshold) || 20,
                rebalance_interval: rebalanceInterval === '' ? '' : String(parseInt(rebalanceInterval, 10)),
                score_weights: JSON.stringify(getNormalizedWeights())
            }).toString()
            const data = await etfSimApi.getAccountSuggestion(simAccountId, q)
            const result = data?.result
            if (result?.suggestion?.error) throw new Error(result.suggestion.error)
            setSuggestion({
                ...result.suggestion,
                trading_plan: result.trading_plan,
                account_info: result.account_info
            })
        } catch (e) {
            console.error('[ETF轮动] 获取调仓建议失败:', e)
            setError(String(e?.message || e))
        } finally {
            setSuggestionLoading(false)
        }
    }

    useEffect(() => {
        if (mode === 'sim') loadSimAccounts()
    }, [mode])


    // ─── 通用卡片样式 ────────────────────────────────────────────
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

    // ─── 得分权重卡片（回测 & 模拟盘共用） ───────────────────────
    const scoreWeightsBlock = (
        <div style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={cardTitle}>📊 得分权重配置</span>
                <button className="buttonSmall" onClick={resetWeights} style={{ fontSize: '11px' }}>重置</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '8px' }}>
                {['momentum', 'rsi', 'ma', 'macd'].map(key => (
                    <div key={key}>
                        <label style={{ fontSize: '11px', color: 'rgba(255,255,255,0.55)', marginBottom: '4px', display: 'block' }}>
                            {key === 'momentum' ? '动量' : key === 'rsi' ? 'RSI' : key === 'ma' ? '均线' : 'MACD'} (%)
                        </label>
                        <input type="number" className="input" value={scoreWeights[key]}
                            onChange={e => updateWeight(key, e.target.value)} min="0" max="100" step="1" />
                    </div>
                ))}
            </div>
            <div style={{
                fontSize: '11px', fontWeight: '500', padding: '6px 10px', borderRadius: '4px',
                color: weightsSum === 100 ? '#4ade80' : '#fbbf24',
                background: weightsSum === 100 ? 'rgba(74,222,128,0.1)' : 'rgba(251,191,36,0.1)',
                border: `1px solid ${weightsSum === 100 ? 'rgba(74,222,128,0.3)' : 'rgba(251,191,36,0.3)'}`
            }}>
                {weightsSum === 100 ? `✓ 权重总和：100%` : `⚠ 权重总和：${weightsSum}%（将自动归一化）`}
            </div>
        </div>
    )

    // ─── 策略参数卡片（回测 & 模拟盘共用） ───────────────────────
    const strategyParamsBlock = (
        <div style={card}>
            <div style={cardTitle}>策略参数</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="formGroup">
                    <label className="label">初始资金</label>
                    <input type="number" className="input" value={initialCapital} onChange={e => setInitialCapital(e.target.value)} placeholder="100000" />
                </div>
                <div className="formGroup">
                    <label className="label">回测天数</label>
                    <select className="select" value={days} onChange={e => setDays(e.target.value)}>
                        <option value="60">60天（约一季）</option>
                        <option value="126">126天（约半年）</option>
                        <option value="252">252天（约一年）</option>
                        <option value="504">504天（约两年）</option>
                        <option value="756">756天（约三年）</option>
                    </select>
                </div>
                <div className="formGroup">
                    <label className="label">轮动间隔（交易日）</label>
                    <select className="select" value={rotationInterval} onChange={e => setRotationInterval(e.target.value)}>
                        <option value="1">1天（每日）</option>
                        <option value="5">5天（每周）</option>
                        <option value="10">10天（每两周）</option>
                        <option value="20">20天（每月）</option>
                    </select>
                </div>
                <div className="formGroup">
                    <label className="label">回看天数（计算得分）</label>
                    <select className="select" value={lookbackDays} onChange={e => setLookbackDays(e.target.value)}>
                        <option value="10">10天</option>
                        <option value="20">20天（推荐）</option>
                        <option value="30">30天</option>
                        <option value="60">60天</option>
                    </select>
                </div>
                <div className="formGroup">
                    <label className="label">持仓数量（Top-K）</label>
                    <select className="select" value={topK} onChange={e => setTopK(e.target.value)}>
                        {[1,2,3,4,5].map(n => <option key={n} value={String(n)}>{n}只</option>)}
                    </select>
                </div>
                <div className="formGroup">
                    <label className="label">仓位策略</label>
                    <select className="select" value={positionStrategy} onChange={e => setPositionStrategy(e.target.value)}>
                        <option value="equal">等权重</option>
                        <option value="kelly">凯利公式</option>
                    </select>
                </div>
                <div className="formGroup">
                    <label className="label">最低得分阈值</label>
                    <input type="number" className="input" value={minScoreThreshold}
                        onChange={e => setMinScoreThreshold(e.target.value)} placeholder="0（不限制）" step="0.1" min="0" />
                </div>
                <div className="formGroup">
                    <label className="label">再平衡间隔</label>
                    <select className="select" value={rebalanceInterval} onChange={e => setRebalanceInterval(e.target.value)}>
                        <option value="">无再平衡</option>
                        <option value="1">每天</option>
                        <option value="5">每周</option>
                        <option value="10">每两周</option>
                        <option value="20">每月</option>
                    </select>
                </div>
            </div>
        </div>
    )

    // ─── 调仓建议展示（回测 & 模拟盘共用） ───────────────────────
    const suggestionPanel = suggestion && !suggestion.error && (
        <div style={{ ...card, border: '1px solid rgba(86,185,255,0.35)', background: 'rgba(86,185,255,0.06)', marginTop: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontWeight: '600', fontSize: '14px', color: 'rgba(86,185,255,0.9)' }}>📋 当前调仓建议</span>
                <button className="buttonSmall" onClick={() => setSuggestion(null)}>关闭</button>
            </div>
            <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.55)', marginBottom: '10px' }}>
                数据日期：{suggestion.date || '-'} · 回看{suggestion.lookback_days}天 · Top-{suggestion.top_k}
                {suggestion.min_score_threshold > 0 && ` · 阈值 ${suggestion.min_score_threshold}`}
                {suggestion.account_info && ` · 总资产 ¥${suggestion.account_info.total_equity?.toLocaleString()}`}
            </div>
            {suggestion.all_below_threshold && (
                <div style={{ padding: '8px 12px', background: 'rgba(248,113,113,0.12)', border: '1px solid rgba(248,113,113,0.35)', borderRadius: '6px', color: '#f87171', fontWeight: '600', marginBottom: '10px', fontSize: '13px' }}>
                    ⚠️ 所有ETF得分均低于阈值，建议不持仓
                </div>
            )}
            {suggestion.recommended_etfs?.length > 0 && (
                <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '12px', color: '#4ade80', fontWeight: '600', marginBottom: '6px' }}>✓ 建议持有（Top-{suggestion.top_k}）</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {suggestion.recommended_etfs.map(code => {
                            const etfInfo = suggestion.suggestions?.find(s => s.etf_code === code)
                            return (
                                <span key={code} style={{ padding: '5px 12px', background: 'rgba(74,222,128,0.2)', border: '1px solid rgba(74,222,128,0.4)', borderRadius: '6px', fontSize: '13px', fontWeight: '600', color: '#4ade80' }}>
                                    {code} ({etfInfo?.score?.toFixed(1) ?? suggestion.etf_scores?.[code] ?? '-'})
                                </span>
                            )
                        })}
                    </div>
                </div>
            )}
            {suggestion.suggestions?.length > 0 && (
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
                                {['排名','ETF','得分','建议'].map(h => (
                                    <th key={h} style={{ padding: '7px 10px', textAlign: h === '得分' ? 'right' : h === '建议' ? 'center' : 'left', color: 'rgba(255,255,255,0.55)', fontWeight: '500' }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {suggestion.suggestions.map((item, idx) => (
                                <tr key={item.etf_code} style={{ borderBottom: idx < suggestion.suggestions.length - 1 ? '1px solid rgba(255,255,255,0.07)' : 'none', background: item.recommended ? 'rgba(74,222,128,0.06)' : 'transparent' }}>
                                    <td style={{ padding: '7px 10px', color: 'rgba(255,255,255,0.5)' }}>{item.rank}</td>
                                    <td style={{ padding: '7px 10px', fontWeight: item.recommended ? '600' : '400', color: item.recommended ? '#4ade80' : 'rgba(255,255,255,0.85)' }}>{item.etf_code}</td>
                                    <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: '600', color: item.recommended ? '#4ade80' : 'rgba(255,255,255,0.8)' }}>{item.score.toFixed(2)}</td>
                                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>
                                        {item.recommended
                                            ? <span style={{ fontSize: '11px', padding: '2px 8px', background: 'rgba(74,222,128,0.2)', color: '#4ade80', borderRadius: '4px', fontWeight: '600' }}>✓ 持有</span>
                                            : <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.35)' }}>—</span>}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
            {suggestion.trading_plan && (
                <div style={{ marginTop: '14px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '14px' }}>
                    <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginBottom: '10px' }}>调仓执行计划</div>
                    {[
                        { key: 'to_sell',   label: '🔴 卖出', color: '#f87171',  cols: ['ETF','当前持仓','卖出股数','现价','预估回收'],   fields: ['etf_code','current_shares','sell_shares','current_price','estimated_revenue'] },
                        { key: 'to_buy',    label: '🟢 买入', color: '#4ade80',  cols: ['ETF','买入金额','买入股数','现价','目标仓位'],   fields: ['etf_code','target_value','target_shares','current_price','target_weight'] },
                        { key: 'to_adjust', label: '🟡 调整', color: '#fbbf24',  cols: ['ETF','当前持仓','目标持仓','调整股数','操作'],   fields: ['etf_code','current_shares','target_shares','adjust_shares','action'] },
                    ].map(section => suggestion.trading_plan[section.key]?.length > 0 && (
                        <div key={section.key} style={{ marginBottom: '10px' }}>
                            <div style={{ fontSize: '12px', fontWeight: '600', color: section.color, marginBottom: '6px' }}>{section.label}</div>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
                                            {section.cols.map(c => <th key={c} style={{ padding: '5px 8px', textAlign: 'left', color: 'rgba(255,255,255,0.45)', fontWeight: '500' }}>{c}</th>)}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {suggestion.trading_plan[section.key].map((item, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                                {section.fields.map(f => (
                                                    <td key={f} style={{ padding: '5px 8px', color: 'rgba(255,255,255,0.8)', fontWeight: f === 'etf_code' ? '600' : '400' }}>
                                                        {f === 'current_price' || f === 'estimated_revenue' || f === 'target_value' || f === 'adjust_value'
                                                            ? `¥${item[f]?.toFixed(2) ?? '-'}`
                                                            : f === 'target_weight' ? `${item[f]?.toFixed(1)}%`
                                                            : item[f] ?? '-'}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ))}
                    <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', marginTop: '6px' }}>
                        卖出{suggestion.trading_plan.to_sell?.length||0}笔 · 买入{suggestion.trading_plan.to_buy?.length||0}笔 · 调整{suggestion.trading_plan.to_adjust?.length||0}笔
                    </div>
                </div>
            )}
            <div style={{ marginTop: '10px', fontSize: '11px', color: 'rgba(255,255,255,0.35)', fontStyle: 'italic' }}>
                基于技术指标得分（动量{scoreWeights.momentum}% / RSI{scoreWeights.rsi}% / 均线{scoreWeights.ma}% / MACD{scoreWeights.macd}%），仅供参考
            </div>
        </div>
    )

    return (
        <section className="panel poolPanel">
            <header className="panelHeader">
                <div>
                    <div className="panelTitle">📊 ETF轮动策略</div>
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
                {/* Tab 导航 */}
                <div style={{ display: 'flex', gap: '2px', marginBottom: '20px', background: 'rgba(0,0,0,0.25)', borderRadius: '10px', padding: '4px' }}>
                    {[
                        { key: 'backtest', icon: '📈', label: '回测分析',   hint: '历史验证策略' },
                        { key: 'sim',      icon: '💼', label: '模拟盘',     hint: '模拟实盘交易' },
                        { key: 'ai',       icon: '🤖', label: 'AI轮动',     hint: '智能调仓建议' },
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

                {/* ETF 池（所有 Tab 共用） */}
                <div style={{ background: 'rgba(86,185,255,0.07)', border: '1px solid rgba(86,185,255,0.22)', borderRadius: '10px', padding: '14px', marginBottom: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                        <span style={{ fontWeight: '600', fontSize: '14px', color: 'rgba(86,185,255,0.9)' }}>
                            当前ETF池
                            {etfCodes.length > 0 && (
                                <span style={{ marginLeft: '6px', background: 'rgba(86,185,255,0.3)', borderRadius: '10px', padding: '1px 7px', fontSize: '12px' }}>{etfCodes.length}</span>
                            )}
                        </span>
                        {etfCodes.length > 0 && <button className="buttonSmall" onClick={() => setEtfCodes([])}>清空</button>}
                    </div>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                        <input type="text" className="input" value={etfInput}
                            onChange={e => setEtfInput(e.target.value)}
                            onKeyPress={e => e.key === 'Enter' && addEtf()}
                            placeholder="输入ETF代码（如 510300）"
                            disabled={mode === 'sim' && !simAccountId}
                            style={{ flex: 1, fontSize: '13px' }} />
                        <button className="buttonPrimary" onClick={addEtf}
                            disabled={busy || (mode === 'sim' && !simAccountId)}
                            style={{ padding: '8px 12px', fontSize: '13px' }}>添加</button>
                    </div>
                    {mode === 'sim' && !simAccountId && (
                        <div style={{ fontSize: '11px', color: '#fbbf24', marginBottom: '6px' }}>⚠️ 请先创建或选择账户</div>
                    )}
                    {defaultEtfs.length > 0 && (
                        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '8px', alignItems: 'center' }}>
                            <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)' }}>快速：</span>
                            {defaultEtfs.map(etf => (
                                <button key={etf.code} className="buttonSmall"
                                    onClick={() => { if (!etfCodes.includes(etf.code)) setEtfCodes([...etfCodes, etf.code]) }}
                                    disabled={etfCodes.includes(etf.code)}>
                                    {etf.name}
                                </button>
                            ))}
                        </div>
                    )}
                    <div className="stockTags" style={{ minHeight: '32px' }}>
                        {etfCodes.length === 0
                            ? <span className="emptyHint">请添加ETF...</span>
                            : etfCodes.map(code => (
                                <span key={code} className="stockTag">
                                    {code}
                                    {mode === 'sim' && simAccount?.account?.positions?.[code] && (
                                        <span style={{ fontSize: '10px', color: '#4ade80', marginLeft: '3px', fontWeight: 'bold' }}>持仓</span>
                                    )}
                                    <button className="tagRemove" onClick={() => removeEtf(code)} disabled={busy}>×</button>
                                </span>
                            ))
                        }
                    </div>
                </div>

                {/* ════ 回测分析 ════ */}
                {mode === 'backtest' && (
                    <div>
                        <div style={{
                            ...card,
                            background: useAi ? 'linear-gradient(135deg,rgba(139,92,246,0.18),rgba(168,85,247,0.26))' : card.background,
                            border: useAi ? '2px solid rgba(168,85,247,0.45)' : card.border,
                        }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                                <input type="checkbox" checked={useAi} onChange={e => setUseAi(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                                <span style={{ fontWeight: '600', fontSize: '14px' }}>🤖 使用AI分析</span>
                                {!useAi && <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.45)' }}>（当前：纯技术指标得分）</span>}
                            </label>
                            {useAi && (
                                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.65)', marginTop: '8px', lineHeight: '1.6' }}>
                                    <div>✓ 每次轮动时调用LLM分析ETF表现</div>
                                    <div style={{ color: '#fca5a5' }}>⚠️ 速度较慢</div>
                                </div>
                            )}
                        </div>
                        {strategyParamsBlock}
                        {scoreWeightsBlock}
                        <div className="actions">
                            <button className="buttonPrimary" disabled={!canRun || busy} onClick={runBacktest}>
                                {busy ? '回测中...' : '📈 运行ETF轮动回测'}
                            </button>
                            {busy && (
                                <button type="button" className="button" onClick={stopBacktest} disabled={stopping} style={{ padding: '6px 12px', fontSize: '12px' }}>
                                    {stopping ? '正在停止...' : '停止回测'}
                                </button>
                            )}
                            <button className="button" onClick={getSuggestion}
                                disabled={etfCodes.length === 0 || suggestionLoading || busy}
                                style={{ background: 'rgba(74,222,128,0.2)', border: '1px solid rgba(74,222,128,0.4)', color: '#4ade80' }}>
                                {suggestionLoading ? '计算中...' : '📋 当前调仓建议'}
                            </button>
                            <button className="button" onClick={clear} disabled={busy}>清空记录</button>
                        </div>
                        {busy && (
                            <div className="progressContainer" style={{ marginTop: '12px' }}>
                                <div className="progressTitle">
                                    <div className="spinner" />
                                    <span>{stopping ? '正在停止，等待后端...' : '正在进行ETF轮动回测...'}</span>
                                </div>
                                <div className="progressBar">
                                    <div className="progressFill" style={{ width: `${backtestProgress}%` }} />
                                </div>
                                <div className="progressInfo">
                                    <span>{Math.round(backtestProgress)}%</span>
                                    {backtestStreamInfo.total > 0 && (
                                        <span>第{backtestStreamInfo.current}/{backtestStreamInfo.total}天{backtestStreamInfo.date && ` · ${backtestStreamInfo.date}`}{backtestStreamInfo.elapsed > 0 && ` · ${backtestStreamInfo.elapsed}秒`}</span>
                                    )}
                                    <span>{etfCodes.length}只ETF × {days}天</span>
                                </div>
                                {liveDecisions.length > 0 && (
                                    <div className="liveDecisionsPanel" style={{ marginTop: '10px' }}>
                                        <div className="liveDecisionsList">
                                            <div className="liveDecisionsTitle">📊 ETF轮动决策 ({liveDecisions.length})</div>
                                            <div className="liveDecisionsScroll">
                                                {[...liveDecisions].slice(-10).reverse().map((d, idx) => (
                                                    d.type === 'portfolio' ? (
                                                        <div key={idx} className="liveDecisionItem portfolioDecision">
                                                            <span className="ldDate">{d.date}</span>
                                                            <span className="ldPortfolio">📊 Top-{topK}{d.use_ai ? '（AI）' : ''}</span>
                                                            <div className="ldPortfolioDetail">
                                                                <span className="ldAnalysis">{d.analysis}</span>
                                                                {d.target_etfs?.length > 0 && <span style={{ color: 'rgba(255,255,255,0.85)' }}>目标: {d.target_etfs.join(', ')}</span>}
                                                                {d.priority_buy?.length > 0 && <span className="ldBuyList">📈 {d.priority_buy.join(', ')}</span>}
                                                                {d.priority_sell?.length > 0 && <span className="ldSellList">📉 {d.priority_sell.join(', ')}</span>}
                                                            </div>
                                                        </div>
                                                    ) : d.type === 'etf_ai_start' ? (
                                                        <div key={idx} className="liveDecisionItem" style={{ borderLeftColor: '#fbbf24' }}>
                                                            <span className="ldDate">{d.date}</span>
                                                            <span style={{ color: '#fbbf24' }}>🤖 AI分析中（{d.etf_count}只）</span>
                                                        </div>
                                                    ) : (
                                                        <div key={idx} className={`liveDecisionItem ${d.action === 'BUY' ? 'tradeBuy' : 'tradeSell'}`}>
                                                            <span className="ldDate">{d.date}</span>
                                                            <span className="ldCode">{d.stock_code}</span>
                                                            <span className={d.action === 'BUY' ? 'actionBuy' : 'actionSell'}>
                                                                {d.action === 'BUY' ? '🟢买入' : '🔴卖出'}
                                                            </span>
                                                            <span className="ldShares">{d.shares}股</span>
                                                        </div>
                                                    )
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                        {error && <div className="errorText" style={{ marginTop: '8px' }}>{error}</div>}
                        {suggestionPanel}
                    </div>
                )}

                {/* ════ 模拟盘 ════ */}
                {mode === 'sim' && (
                    <div>
                        <div style={card}>
                            <div style={cardTitle}>账户管理</div>
                            <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                                <div className="formGroup" style={{ flex: '1 1 160px', margin: 0 }}>
                                    <label className="label" style={{ marginBottom: '6px' }}>选择已有账户</label>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                                        <select className="select" value={simAccountId} style={{ flex: 1 }}
                                            onChange={e => { setSimAccountId(e.target.value); if (e.target.value) loadSimAccount(e.target.value); else setSimAccount(null) }}>
                                            <option value="">-- 选择账户 --</option>
                                            {simAccounts.map(id => <option key={id} value={id}>{id}</option>)}
                                        </select>
                                        {simAccountId && (
                                            <>
                                                <button className="buttonSmall" onClick={() => loadSimAccount(simAccountId)} disabled={simAccountLoading}>
                                                    {simAccountLoading ? '加载中...' : '刷新'}
                                                </button>
                                                <button className="buttonSmall" onClick={deleteSimAccount} disabled={busy}
                                                    style={{ background: 'rgba(248,113,113,0.2)', border: '1px solid rgba(248,113,113,0.4)', color: '#f87171' }}>
                                                    删除
                                                </button>
                                            </>
                                        )}
                                    </div>
                                </div>
                                <div style={{ flex: '2 1 260px', background: 'rgba(86,185,255,0.07)', border: '1px solid rgba(86,185,255,0.2)', borderRadius: '8px', padding: '12px' }}>
                                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.55)', marginBottom: '8px' }}>➕ 创建新账户</div>
                                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                                        <input type="text" className="input" value={simAccountIdInput}
                                            onChange={e => setSimAccountIdInput(e.target.value)}
                                            placeholder="账户名称" style={{ flex: '2 1 120px' }} />
                                        <input type="number" className="input" value={simInitialCapital}
                                            onChange={e => setSimInitialCapital(e.target.value)}
                                            placeholder="初始资金" style={{ flex: '1 1 90px' }} />
                                        <button className="buttonPrimary" onClick={createSimAccount} disabled={busy} style={{ flexShrink: 0 }}>
                                            {busy ? '创建中...' : '✨ 创建'}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {simAccount && (
                            <>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px,1fr))', gap: '10px', marginBottom: '14px' }}>
                                    {[
                                        { label: '总资产',   value: `¥${simAccount.statistics?.total_equity?.toLocaleString()}`,   color: 'rgba(255,255,255,0.95)' },
                                        { label: '可用资金', value: `¥${simAccount.statistics?.cash?.toLocaleString()}`,            color: 'rgba(255,255,255,0.8)'  },
                                        { label: '持仓市值', value: `¥${simAccount.statistics?.positions_value?.toLocaleString()}`, color: 'rgba(86,185,255,0.9)'   },
                                        {
                                            label: '总盈亏',
                                            value: `${(simAccount.statistics?.profit_loss ?? 0) >= 0 ? '+' : ''}¥${simAccount.statistics?.profit_loss?.toLocaleString()} (${(simAccount.statistics?.profit_loss_pct ?? 0) >= 0 ? '+' : ''}${simAccount.statistics?.profit_loss_pct?.toFixed(2)}%)`,
                                            color: (simAccount.statistics?.profit_loss ?? 0) >= 0 ? '#4ade80' : '#f87171'
                                        },
                                    ].map(stat => (
                                        <div key={stat.label} style={{ background: 'rgba(255,255,255,0.07)', borderRadius: '8px', padding: '10px 12px', border: '1px solid rgba(255,255,255,0.1)' }}>
                                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', marginBottom: '4px' }}>{stat.label}</div>
                                            <div style={{ fontSize: '13px', fontWeight: '600', color: stat.color }}>{stat.value}</div>
                                        </div>
                                    ))}
                                </div>
                                {(simAccount.account?.last_rotation_date || simAccount.account?.last_rebalance_date) && (
                                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '12px' }}>
                                        {simAccount.account.last_rotation_date && `上次轮动：${new Date(simAccount.account.last_rotation_date).toLocaleDateString('zh-CN')}`}
                                        {simAccount.account.last_rebalance_date && `　上次再平衡：${new Date(simAccount.account.last_rebalance_date).toLocaleDateString('zh-CN')}`}
                                    </div>
                                )}
                                {simAccount.positions_detail?.length > 0 && (
                                    <div style={{ marginBottom: '14px' }}>
                                        <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginBottom: '8px' }}>
                                            当前持仓（{simAccount.positions_detail.length}只）
                                        </div>
                                        <div style={{ overflowX: 'auto' }}>
                                            <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                                                <thead>
                                                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
                                                        {['ETF','持仓','成本价','现价','市值','盈亏','开仓日'].map(h => (
                                                            <th key={h} style={{ padding: '7px 10px', textAlign: h === 'ETF' || h === '开仓日' ? 'left' : 'right', color: 'rgba(255,255,255,0.5)', fontWeight: '500' }}>{h}</th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {simAccount.positions_detail.map((pos, idx) => {
                                                        const entryDate = simAccount.account?.positions?.[pos.etf_code]?.entry_date || pos.entry_date
                                                        return (
                                                            <tr key={pos.etf_code} style={{ borderBottom: idx < simAccount.positions_detail.length - 1 ? '1px solid rgba(255,255,255,0.07)' : 'none' }}>
                                                                <td style={{ padding: '7px 10px', fontWeight: '600' }}>{pos.etf_code}</td>
                                                                <td style={{ padding: '7px 10px', textAlign: 'right', color: 'rgba(255,255,255,0.8)' }}>{pos.shares}股</td>
                                                                <td style={{ padding: '7px 10px', textAlign: 'right', color: 'rgba(255,255,255,0.6)' }}>¥{pos.entry_price?.toFixed(2)}</td>
                                                                <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: '600' }}>¥{pos.current_price?.toFixed(2)}</td>
                                                                <td style={{ padding: '7px 10px', textAlign: 'right' }}>¥{pos.market_value?.toFixed(2)}</td>
                                                                <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: '600', color: (pos.profit_loss ?? 0) >= 0 ? '#4ade80' : '#f87171' }}>
                                                                    {(pos.profit_loss >= 0 ? '+' : '')}¥{pos.profit_loss?.toFixed(2)}
                                                                    <span style={{ fontSize: '11px', marginLeft: '3px' }}>({(pos.profit_loss_pct >= 0 ? '+' : '')}{pos.profit_loss_pct?.toFixed(2)}%)</span>
                                                                </td>
                                                                <td style={{ padding: '7px 10px', color: 'rgba(255,255,255,0.45)', fontSize: '11px' }}>
                                                                    {entryDate ? new Date(entryDate).toLocaleDateString('zh-CN') : '-'}
                                                                </td>
                                                            </tr>
                                                        )
                                                    })}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                                {simAccount.account?.trades?.length > 0 && (
                                    <TradesHistory accountId={simAccountId} totalTrades={simAccount.account.trades.length} />
                                )}
                            </>
                        )}

                        {strategyParamsBlock}
                        {scoreWeightsBlock}
                        <div className="actions">
                            <button className="buttonPrimary" onClick={executeSimAutoTrade} disabled={!simAccountId || busy}>
                                {busy ? '执行中...' : '🔄 执行自动调仓'}
                            </button>
                            <button className="button" onClick={simAccountId ? getSimSuggestion : getSuggestion}
                                disabled={suggestionLoading || busy || (mode === 'sim' && !simAccountId)}
                                style={{ background: 'rgba(74,222,128,0.2)', border: '1px solid rgba(74,222,128,0.4)', color: '#4ade80' }}>
                                {suggestionLoading ? '计算中...' : '📋 查看调仓建议'}
                            </button>
                        </div>
                        {error && <div className="errorText" style={{ marginTop: '8px' }}>{error}</div>}
                        {suggestionPanel}
                    </div>
                )}

                {/* ════ AI 轮动 ════ */}
                {mode === 'ai' && (
                    <div>
                        {/* 角色说明 */}
                        <div style={{ ...card, background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.35)' }}>
                            <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.8)', lineHeight: '1.7' }}>
                                <div style={{ fontWeight: '600', marginBottom: '6px', color: 'rgba(196,181,253,0.95)' }}>🤖 AI 的职责</div>
                                <div>参数与得分权重由你手动配置（含义与回测模式完全相同）。</div>
                                <div>AI 读取这些参数后，对 ETF 池进行技术指标评分，并结合 LLM 判断，<strong>自主决定最终推荐哪些 ETF 持仓</strong>。</div>
                            </div>
                        </div>
                        {strategyParamsBlock}
                        {scoreWeightsBlock}
                        <div className="actions">
                            <button className="buttonPrimary" onClick={runAIRotation} disabled={!canRun || aiLoading}
                                style={{ background: 'rgba(139,92,246,0.7)' }}>
                                {aiLoading ? '🤖 AI分析中...' : '🤖 运行AI轮动分析'}
                            </button>
                        </div>
                        {error && <div className="errorText" style={{ marginTop: '8px' }}>{error}</div>}
                        {aiResult && (
                            <div style={{ ...card, marginTop: '14px', border: '1px solid rgba(139,92,246,0.4)', background: 'rgba(139,92,246,0.08)' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                                    <span style={{ fontWeight: '600', fontSize: '15px', color: 'rgba(196,181,253,0.95)' }}>🤖 AI分析结果</span>
                                    <button className="buttonSmall" onClick={() => setAiResult(null)}>关闭</button>
                                </div>
                                {aiResult.structured_data?.final_recommended?.length > 0 && (
                                    <div style={{ marginBottom: '14px', padding: '12px', background: 'rgba(74,222,128,0.1)', border: '1px solid rgba(74,222,128,0.35)', borderRadius: '8px' }}>
                                        <div style={{ fontSize: '12px', color: '#4ade80', fontWeight: '600', marginBottom: '8px' }}>✓ 最终推荐ETF</div>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                            {aiResult.structured_data.final_recommended.map(code => (
                                                <span key={code} style={{ padding: '6px 14px', background: 'rgba(74,222,128,0.25)', border: '1px solid rgba(74,222,128,0.5)', borderRadius: '6px', fontWeight: '600', fontSize: '14px', color: '#4ade80' }}>{code}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {aiResult.structured_data?.etf_scores && Object.keys(aiResult.structured_data.etf_scores).length > 0 && (
                                    <div style={{ marginBottom: '14px' }}>
                                        <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.8)', marginBottom: '8px' }}>📈 ETF得分排名</div>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px,1fr))', gap: '8px' }}>
                                            {Object.entries(aiResult.structured_data.etf_scores)
                                                .sort((a, b) => b[1] - a[1])
                                                .map(([code, score], index) => {
                                                    const isRec = aiResult.structured_data.final_recommended?.includes(code)
                                                    return (
                                                        <div key={code} style={{ padding: '10px', background: isRec ? 'rgba(74,222,128,0.12)' : 'rgba(255,255,255,0.07)', borderRadius: '6px', border: `1px solid ${isRec ? 'rgba(74,222,128,0.35)' : 'rgba(255,255,255,0.1)'}`, textAlign: 'center' }}>
                                                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}>{index + 1}. {code}</div>
                                                            <div style={{ fontSize: '17px', fontWeight: 'bold', color: isRec ? '#4ade80' : 'rgba(255,255,255,0.9)' }}>{score}</div>
                                                        </div>
                                                    )
                                                })}
                                        </div>
                                    </div>
                                )}
                                {aiResult.report && (
                                    <div>
                                        <div style={{ fontSize: '13px', fontWeight: '600', color: 'rgba(255,255,255,0.75)', marginBottom: '8px' }}>📄 详细分析报告</div>
                                        <pre style={{ fontSize: '12px', color: 'rgba(255,255,255,0.8)', whiteSpace: 'pre-wrap', lineHeight: '1.8', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)', maxHeight: '500px', overflowY: 'auto' }}>{aiResult.report}</pre>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {mode === 'backtest' && (
                <div className="panelList" aria-label="ETF轮动 结果列表">
                    {items.length === 0
                        ? <div className="empty">暂无已保存结果</div>
                        : items.map(it => (
                            <article key={it.id} className="card backtestResult">
                                <div className="cardMeta">{formatTs(it.ts)}</div>
                                <pre className="cardText">{typeof it === 'string' ? it : (it.text || it)}</pre>
                                {it.chart && (
                                    <div className="chartContainer">
                                        <img src={it.chart.startsWith('data:') ? it.chart : `data:image/png;base64,${it.chart}`}
                                            alt="回测图表" className="chartImage"
                                            onError={e => { e.target.style.display = 'none' }} />
                                    </div>
                                )}
                            </article>
                        ))
                    }
                </div>
            )}
        </section>
    )
}
