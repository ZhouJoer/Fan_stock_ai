import React, { useMemo, useState, useEffect, useRef } from 'react'
import * as poolApi from '../api/pool.js'
import { baseURL, getAuthToken } from '../api/client.js'

export default function FactorMiningPage() {
    const [universeSource, setUniverseSource] = useState('index')
    const [stockInput, setStockInput] = useState('600519,000858,601318')
    const [universeIndex, setUniverseIndex] = useState('000300')
    const [industryNames, setIndustryNames] = useState([])
    const [industryList, setIndustryList] = useState([])
    const [leadersPerIndustry, setLeadersPerIndustry] = useState(1)
    const [days, setDays] = useState(252)

    const [deepSearchBusy, setDeepSearchBusy] = useState(false)
    const [deepSearchResult, setDeepSearchResult] = useState(null)
    const [deepSearchError, setDeepSearchError] = useState('')
    const [deepSearchMaxStocks, setDeepSearchMaxStocks] = useState(60)
    const [deepSearchExcludeKechuang, setDeepSearchExcludeKechuang] = useState(false)
    const [deepSearchExcludeSmallCap, setDeepSearchExcludeSmallCap] = useState(false)
    const [deepSearchSmallCapMaxBillion, setDeepSearchSmallCapMaxBillion] = useState(30)
    const [deepSearchFactorMode, setDeepSearchFactorMode] = useState('multi')
    const [deepSearchCapScope, setDeepSearchCapScope] = useState('none')
    const [deepSearchBenchmarkCode, setDeepSearchBenchmarkCode] = useState('510300')
    const [deepSearchLabelHorizon, setDeepSearchLabelHorizon] = useState(5)
    const [deepSearchRebalanceFreq, setDeepSearchRebalanceFreq] = useState(1)
    const [deepSearchTopN, setDeepSearchTopN] = useState(10)
    const [deepSearchOrchestrateTasks, setDeepSearchOrchestrateTasks] = useState(true)
    const [deepSearchOrchestratePreference, setDeepSearchOrchestratePreference] = useState('')
    const [deepSearchMaxCombos, setDeepSearchMaxCombos] = useState(15)
    const [deepSearchNTrials, setDeepSearchNTrials] = useState(8)
    const [deepSearchProgress, setDeepSearchProgress] = useState({ current: 0, total: 0, message: '' })
    const [agentLogs, setAgentLogs] = useState([])
    const [agentLogPanelOpen, setAgentLogPanelOpen] = useState(false)
    const [availableFactorsList, setAvailableFactorsList] = useState(null)
    const [availableFactorsPanelOpen, setAvailableFactorsPanelOpen] = useState(false)
    const [rebalanceDetailsOpen, setRebalanceDetailsOpen] = useState(false)
    const [factorBacktestLoading, setFactorBacktestLoading] = useState(false)
    const [factorBacktestResult, setFactorBacktestResult] = useState(null)
    const [backtestTopN, setBacktestTopN] = useState(10)
    const [backtestRebalanceFreq, setBacktestRebalanceFreq] = useState(1)
    const [backtestDays, setBacktestDays] = useState(252)
    const [backtestExcludeKechuang, setBacktestExcludeKechuang] = useState(false)
    const [backtestCapScope, setBacktestCapScope] = useState('none')
    const [backtestSmallCapMaxBillion, setBacktestSmallCapMaxBillion] = useState(30)
    const [backtestPositionWeight, setBacktestPositionWeight] = useState('equal')
    const [robustnessCheck, setRobustnessCheck] = useState(false)
    const [backtestPoolMode, setBacktestPoolMode] = useState('same')
    const [backtestManualStocks, setBacktestManualStocks] = useState('')
    const [backtestProgress, setBacktestProgress] = useState({ phase: '', pct: 0, message: '' })
    const [savedSummaries, setSavedSummaries] = useState([])
    const [saveSummaryTitle, setSaveSummaryTitle] = useState('')
    const deepSearchEventSourceRef = useRef(null)
    const deepSearchSessionIdRef = useRef(null)
    const deepSearchAbortRef = useRef(null)

    useEffect(() => {
        if (universeSource === 'industry' && industryNames.length === 0) {
            poolApi.industryNames()
                .then(data => { if (data?.result) setIndustryNames(data.result) })
                .catch(() => {})
        }
    }, [universeSource, industryNames.length])

    // 新挖掘结果到达时，回测参数同步为挖掘参数，便于复现挖掘收益（仅同步一次，不覆盖用户后续修改）
    const lastSyncedResultRef = useRef(null)
    useEffect(() => {
        const best = deepSearchResult?.best
        const space = deepSearchResult?.search_space
        if (!best || deepSearchResult === lastSyncedResultRef.current) return
        lastSyncedResultRef.current = deepSearchResult
        const miningTopN = best.top_n ?? space?.top_n ?? 10
        const miningRebal = best.rebalance_freq ?? space?.rebalance_freq ?? 1
        const miningDays = space?.days ?? 252
        setBacktestTopN(miningTopN)
        setBacktestRebalanceFreq(miningRebal)
        setBacktestDays(Number(miningDays) || 252)
    }, [deepSearchResult])

    useEffect(() => {
        poolApi.backtestSummariesList()
            .then(data => { if (Array.isArray(data?.result)) setSavedSummaries(data.result) })
            .catch(() => {})
    }, [])

    const manualStocks = useMemo(
        () => stockInput.split(',').map(x => x.trim()).filter(Boolean),
        [stockInput]
    )

    function getUniverseValidationError() {
        if (universeSource === 'manual' && manualStocks.length === 0) return '手动股票池不能为空'
        if (universeSource === 'index' && !(universeIndex || '').trim()) return '请输入指数代码'
        if (universeSource === 'industry' && industryList.length === 0) return '请至少选择一个行业'
        return ''
    }

    function getDeepSearchValidationError() {
        const universeErr = getUniverseValidationError()
        if (universeErr) return universeErr
        if (universeSource === 'manual' && manualStocks.length < 8) return '深度搜索至少需要约 8 只股票，建议 10+ 只'
        return ''
    }

    function buildDeepSearchPayload() {
        return {
            stocks: universeSource === 'manual' ? manualStocks : [],
            universe_source: universeSource,
            universe_index: universeSource === 'index' ? ((universeIndex || '').trim() || '000300') : '',
            industry_list: universeSource === 'industry' ? industryList : null,
            leaders_per_industry: universeSource === 'industry' ? Number(leadersPerIndustry || 1) : 1,
            max_stocks: Number(deepSearchMaxStocks || 60),
            days: Number(days || 252),
            exclude_kechuang: deepSearchExcludeKechuang,
            exclude_small_cap: deepSearchExcludeSmallCap,
            small_cap_max_billion: Number(deepSearchSmallCapMaxBillion) || 30,
            factor_mode: deepSearchFactorMode,
            cap_scope: deepSearchCapScope,
            small_cap_threshold_billion: Number(deepSearchSmallCapMaxBillion) || 30,
            benchmark_code: (deepSearchBenchmarkCode || '510300').trim() || '510300',
            label_horizon: Number(deepSearchLabelHorizon) || 5,
            rebalance_freq: Math.max(1, Number(deepSearchRebalanceFreq) || 1),
            top_n: Math.max(1, Math.min(Number(deepSearchTopN) || 10, 50)),
            orchestrate_tasks: deepSearchOrchestrateTasks,
            orchestrate_user_preference: deepSearchOrchestrateTasks ? (deepSearchOrchestratePreference || '').trim() : '',
            max_combos: Math.max(1, Math.min(150, Number(deepSearchNTrials) || 8)),
            n_trials: Math.max(1, Math.min(150, Number(deepSearchNTrials) || 8))
        }
    }

    async function runDeepSearch() {
        const errMsg = getDeepSearchValidationError()
        if (errMsg) {
            setDeepSearchError(errMsg)
            return
        }
        if (deepSearchBusy) {
            setDeepSearchError('当前有任务执行中，请稍后再试')
            return
        }
        setDeepSearchBusy(true)
        setDeepSearchError('')
        setDeepSearchResult(null)
        setDeepSearchProgress({ current: 0, total: 0, message: '' })
        setAgentLogs([])
        try {
            const payload = buildDeepSearchPayload()
            const data = await poolApi.factorDeepSearchStart(payload)
            const sessionId = data?.session_id
            if (!sessionId) {
                setDeepSearchError('启动失败：未返回 session_id')
                setDeepSearchBusy(false)
                return
            }
            deepSearchSessionIdRef.current = sessionId
            const streamPath = `${baseURL || ''}/api/pool/factor-deep-search/stream?session_id=${encodeURIComponent(sessionId)}`
            const handleMsg = (msg) => {
                console.log('[deep-search] 收到消息:', msg.type, msg)
                if (msg.type === 'progress') {
                    setDeepSearchProgress({
                        current: msg.current || 0,
                        total: msg.total || 0,
                        message: msg.message || ''
                    })
                } else if (msg.type === 'agent_log') {
                    setAgentLogs(prev => [...prev, { role: msg.role || '', phase: msg.phase || '', content: msg.content || '' }])
                } else if (msg.type === 'complete') {
                    console.log('[deep-search] >>> COMPLETE，result=', msg.result)
                    deepSearchAbortRef.current = null
                    const res = msg.result || {}
                    if (res.error && !res.best) {
                        setDeepSearchError(res.error)
                    } else {
                        setDeepSearchResult(res)
                    }
                    setDeepSearchBusy(false)
                    setDeepSearchProgress({ current: 0, total: 0, message: '' })
                } else if (msg.type === 'error') {
                    console.log('[deep-search] >>> ERROR:', msg.message)
                    deepSearchAbortRef.current = null
                    setDeepSearchError(msg.message || '搜索失败')
                    setDeepSearchBusy(false)
                    setDeepSearchProgress({ current: 0, total: 0, message: '' })
                }
            }
            const MAX_RECONNECT = 30
            let reconnects = 0
            while (reconnects <= MAX_RECONNECT) {
                const token = getAuthToken()
                const headers = token ? { Authorization: `Bearer ${token}` } : {}
                const ac = new AbortController()
                deepSearchAbortRef.current = ac
                let resp, finished = false
                try {
                    resp = await fetch(streamPath, { headers, signal: ac.signal })
                } catch (e) {
                    if (e?.name === 'AbortError') return
                    console.warn('[deep-search] fetch 失败，2s 后重连', e)
                    reconnects++
                    await new Promise(r => setTimeout(r, 2000))
                    continue
                }
                if (!resp.ok) {
                    if (resp.status === 401) { setDeepSearchError('需要登录'); setDeepSearchBusy(false); return }
                    if (resp.status === 404) {
                        console.warn('[deep-search] session 已结束 (404)，搜索可能已完成')
                        setDeepSearchProgress(prev => ({ ...prev, message: '搜索已结束（结果已写入 outputs 文件夹）' }))
                        setDeepSearchBusy(false)
                        return
                    }
                    console.warn('[deep-search] stream 非 200:', resp.status)
                    reconnects++
                    await new Promise(r => setTimeout(r, 2000))
                    continue
                }
                const reader = resp.body.getReader()
                const decoder = new TextDecoder()
                let buf = ''
                console.log('[deep-search] 流已连接，开始读取 (reconnects=' + reconnects + ')')
                try {
                    while (true) {
                        const { value, done } = await reader.read()
                        if (done) { console.log('[deep-search] reader done=true'); break }
                        buf += decoder.decode(value, { stream: true })
                        const lines = buf.split('\n\n')
                        buf = lines.pop() || ''
                        for (const block of lines) {
                            const m = block.match(/^data:\s*(\{.*\})\s*$/m)
                            if (m) {
                                const msg = JSON.parse(m[1])
                                if (msg.type === 'keepalive') continue
                                handleMsg(msg)
                                if (msg.type === 'complete' || msg.type === 'error') { finished = true; return }
                            }
                        }
                    }
                } catch (e) {
                    if (e?.name === 'AbortError') return
                    console.warn('[deep-search] 流读取异常，将重连', e)
                } finally {
                    deepSearchAbortRef.current = null
                    if (finished) return
                }
                console.warn('[deep-search] 流断开（无 complete），2s 后重连 (' + (reconnects + 1) + ')')
                setDeepSearchProgress(prev => ({ ...prev, message: (prev.message || '') + '（重连中…）' }))
                reconnects++
                await new Promise(r => setTimeout(r, 2000))
            }
            setDeepSearchError('连接多次中断，请稍后重试或检查网络')
            setDeepSearchBusy(false)
        } catch (e) {
            console.log('[deep-search] >>> 外层 catch:', e)
            setDeepSearchError(String(e?.message || e))
            setDeepSearchBusy(false)
        }
    }

    function stopDeepSearch() {
        const sessionId = deepSearchSessionIdRef.current
        if (sessionId) {
            poolApi.factorDeepSearchStop(sessionId).catch(() => {})
            deepSearchSessionIdRef.current = null
        }
        deepSearchAbortRef.current?.abort()
        deepSearchAbortRef.current = null
        deepSearchEventSourceRef.current = null
        setDeepSearchBusy(false)
    }

    /** 将深度搜索最佳结果同步到深度挖掘参数 */
    function loadDeepSearchResultToMining() {
        const best = deepSearchResult?.best
        if (!best) return
        setDeepSearchLabelHorizon(Number(best.label_horizon) || 5)
        setDeepSearchError('')
    }

    /** 仅回测：用 start + stream 获取真实进度（加载数据、回测天数） */
    async function runFactorBacktest() {
        const best = deepSearchResult?.best
        if (!best?.best_factor_combo?.length) {
            setDeepSearchError('请先完成深度搜索并获得最佳组合后再执行仅回测')
            return
        }
        setFactorBacktestLoading(true)
        setFactorBacktestResult(null)
        setRebalanceDetailsOpen(false)
        setBacktestProgress({ phase: '', pct: 0, message: '准备中…' })
        const base = buildDeepSearchPayload()
        let universe_source = base.universe_source || 'manual'
        let universe_index = (base.universe_index || '').trim() || ''
        let stocks = base.stocks
        if (backtestPoolMode === 'index_000300') {
            universe_source = 'index'
            universe_index = '000300'
            stocks = []
        } else if (backtestPoolMode === 'index_000016') {
            universe_source = 'index'
            universe_index = '000016'
            stocks = []
        } else if (backtestPoolMode === 'manual') {
            universe_source = 'manual'
            universe_index = ''
            stocks = (backtestManualStocks || '').split(',').map(s => s.trim()).filter(Boolean)
        }
        const payload = {
            stocks,
            universe_source,
            universe_index,
            industry_list: backtestPoolMode === 'same' ? base.industry_list : [],
            leaders_per_industry: backtestPoolMode === 'same' ? base.leaders_per_industry : 1,
            max_stocks: base.max_stocks,
            days: Number(backtestDays) || base.days,
            benchmark_code: base.benchmark_code,
            exclude_kechuang: backtestExcludeKechuang,
            exclude_small_cap: backtestCapScope === 'only_small_cap' ? false : backtestCapScope === 'exclude_small_cap',
            cap_scope: backtestCapScope,
            small_cap_max_billion: Number(backtestSmallCapMaxBillion) || 30,
            small_cap_threshold_billion: Number(backtestSmallCapMaxBillion) || 30,
            factor_combo: best.best_factor_combo,
            weights: best.learned_weights || {},
            label_horizon: Number(best.label_horizon) || deepSearchLabelHorizon || 5,
            rebalance_freq: Math.max(1, Number(backtestRebalanceFreq) || 1),
            top_n: Math.max(1, Math.min(Number(backtestTopN) || 10, 50)),
            position_weight_method: (backtestPositionWeight || 'equal').trim().toLowerCase() || 'equal',
            robustness_check: robustnessCheck
        }
        try {
            const startRes = await poolApi.factorBacktestStart(payload)
            const sessionId = startRes?.session_id
            if (!sessionId) {
                setFactorBacktestResult({ error: '启动回测失败：未返回 session_id' })
                setFactorBacktestLoading(false)
                return
            }
            const streamPath = `${baseURL || ''}/api/pool/factor-backtest/stream?session_id=${encodeURIComponent(sessionId)}`
            const token = getAuthToken()
            const headers = token ? { Authorization: `Bearer ${token}` } : {}
            const resp = await fetch(streamPath, { headers })
            if (!resp.ok) {
                setFactorBacktestResult({ error: `请求失败: ${resp.status}` })
                setFactorBacktestLoading(false)
                return
            }
            const reader = resp.body.getReader()
            const decoder = new TextDecoder()
            let buf = ''
            while (true) {
                const { value, done } = await reader.read()
                if (done) break
                buf += decoder.decode(value, { stream: true })
                const lines = buf.split('\n\n')
                buf = lines.pop() || ''
                for (const block of lines) {
                    const m = block.match(/^data:\s*(\{.*\})\s*$/m)
                    if (m) {
                        const msg = JSON.parse(m[1])
                        if (msg.type === 'keepalive') continue
                        if (msg.type === 'progress') {
                            setBacktestProgress({ phase: msg.phase || '', pct: Number(msg.pct) || 0, message: msg.message || '' })
                            continue
                        }
                        if (msg.type === 'complete') {
                            setFactorBacktestResult(msg.result || null)
                            setBacktestProgress({ phase: '', pct: 100, message: '完成' })
                            setFactorBacktestLoading(false)
                            return
                        }
                        if (msg.type === 'error') {
                            setFactorBacktestResult({ error: msg.message || '回测失败' })
                            setFactorBacktestLoading(false)
                            return
                        }
                    }
                }
            }
            setFactorBacktestLoading(false)
        } catch (e) {
            setFactorBacktestResult({ error: String(e?.message || e) })
            setFactorBacktestLoading(false)
        }
    }

    return (
        <section className="panel">
            <header className="panelHeader">
                <div className="panelTitle">🧪 因子挖掘</div>
                <div className="panelMeta">深度因子挖掘（LLM Workflow）</div>
            </header>

            <div className="panelInput">
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <div className="formGroup" style={{ flex: 1, minWidth: 140 }}>
                        <label className="label">挖掘范围</label>
                        <select className="select" value={universeSource} onChange={(e) => setUniverseSource(e.target.value)}>
                            <option value="index">指数成分</option>
                            <option value="industry">分行业龙头</option>
                            <option value="manual">手动股票池</option>
                        </select>
                    </div>
                    <div className="formGroup" style={{ width: 100 }}>
                        <label className="label">评估窗口(天)</label>
                        <input type="number" className="input" value={days} onChange={(e) => setDays(e.target.value)} />
                    </div>
                </div>

                {universeSource === 'index' && (
                    <div className="formGroup">
                        <label className="label">指数代码</label>
                        <input className="input" value={universeIndex} onChange={(e) => setUniverseIndex(e.target.value)} placeholder="如 000300" />
                    </div>
                )}
                {universeSource === 'manual' && (
                    <div className="formGroup">
                        <label className="label">股票列表（逗号分隔）</label>
                        <input className="input" value={stockInput} onChange={(e) => setStockInput(e.target.value)} placeholder="如 600519,000858,601318" />
                    </div>
                )}
                {universeSource === 'industry' && (
                    <>
                        <div className="formGroup">
                            <label className="label">行业（可多选）</label>
                            <select className="select" onChange={(e) => {
                                const val = e.target.value
                                if (val && !industryList.includes(val)) setIndustryList(prev => [...prev, val])
                            }}>
                                <option value="">请选择行业</option>
                                {industryNames.map(name => <option key={name} value={name}>{name}</option>)}
                            </select>
                        </div>
                        <div className="stockTags">
                            {industryList.length === 0 ? <span className="emptyHint">请选择行业</span> : industryList.map(name => (
                                <span key={name} className="stockTag">{name}<button className="tagRemove" onClick={() => setIndustryList(industryList.filter(x => x !== name))}>×</button></span>
                            ))}
                        </div>
                        <div className="formGroup">
                            <label className="label">每行业龙头数</label>
                            <input type="number" className="input" value={leadersPerIndustry} onChange={(e) => setLeadersPerIndustry(e.target.value)} />
                        </div>
                    </>
                )}

                    <div style={{ marginTop: 12, padding: 14, borderRadius: 10, border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(255,255,255,0.04)' }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#9db4d8', marginBottom: 10 }}>深度因子挖掘 — 三 Agent 协作</div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, cursor: 'pointer', fontSize: 13, color: 'rgba(255,255,255,0.9)' }}>
                        <input type="checkbox" checked={deepSearchOrchestrateTasks} onChange={e => setDeepSearchOrchestrateTasks(e.target.checked)} />
                        <span>使用 Agent 工作流（编排 + 审查）</span>
                        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)' }}>— 开启后会有审查结论与推荐/不推荐</span>
                    </label>

                    {/* 第一行：核心参数 */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 10, marginBottom: 10 }}>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">因子模式</label>
                            <select className="select" value={deepSearchFactorMode} onChange={e => setDeepSearchFactorMode(e.target.value)}>
                                <option value="single">单因子</option>
                                <option value="dual">双因子</option>
                                <option value="multi">多因子</option>
                            </select>
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">预测步长(日)</label>
                            <input type="number" className="input" value={deepSearchLabelHorizon} onChange={e => setDeepSearchLabelHorizon(e.target.value)} min={1} />
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">调仓周期(日)</label>
                            <input type="number" className="input" value={deepSearchRebalanceFreq} onChange={e => setDeepSearchRebalanceFreq(e.target.value)} min={1} title="1=每日, 5=周频" />
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">迭代次数</label>
                            <input type="number" className="input" value={deepSearchNTrials} onChange={e => setDeepSearchNTrials(e.target.value)} min={1} max={150} title="探索/评价的因子组合组数（1–150）" />
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">挖掘 TopN</label>
                            <select className="select" value={deepSearchTopN} onChange={e => setDeepSearchTopN(Number(e.target.value))}>
                                {[3,5,8,10,15,20,30].map(n => <option key={n} value={n}>{n}</option>)}
                            </select>
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">最大股票数</label>
                            <input type="number" className="input" value={deepSearchMaxStocks} onChange={e => setDeepSearchMaxStocks(e.target.value)} />
                        </div>
                        <div className="formGroup" style={{ margin: 0 }}>
                            <label className="label">基准代码</label>
                            <input className="input" value={deepSearchBenchmarkCode} onChange={e => setDeepSearchBenchmarkCode(e.target.value)} placeholder="510300" />
                        </div>
                    </div>

                    {/* 第二行：排除 / 市值 */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', paddingTop: 8, borderTop: '1px solid rgba(255,255,255,0.08)', marginBottom: 10 }}>
                        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.45)' }}>挖掘股票池过滤</span>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: 12 }}>
                            <input type="checkbox" checked={deepSearchExcludeKechuang} onChange={e => setDeepSearchExcludeKechuang(e.target.checked)} />
                            排除科创板
                        </label>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <span style={{ fontSize: 12, opacity: 0.7 }}>市值筛选</span>
                            <select className="select" style={{ width: 118 }} value={deepSearchCapScope} onChange={e => setDeepSearchCapScope(e.target.value)}>
                                <option value="none">不筛</option>
                                <option value="only_small_cap">仅小市值</option>
                                <option value="exclude_small_cap">排除小市值</option>
                            </select>
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <span style={{ fontSize: 12, opacity: 0.7 }}>阈值(亿)</span>
                            <input type="number" className="input" style={{ width: 68 }} value={deepSearchSmallCapMaxBillion} onChange={e => setDeepSearchSmallCapMaxBillion(e.target.value)} />
                        </span>
                    </div>
                    {/* 可展开：可选因子列表（与后端 factor_registry 一致，便于增减因子） */}
                    <div style={{ marginBottom: 10, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, overflow: 'hidden' }}>
                        <button
                            type="button"
                            onClick={async () => {
                                const next = !availableFactorsPanelOpen
                                setAvailableFactorsPanelOpen(next)
                                if (next && availableFactorsList === null) {
                                    try {
                                        const data = await poolApi.availableFactors()
                                        setAvailableFactorsList(data?.result || [])
                                    } catch (e) {
                                        setAvailableFactorsList([])
                                    }
                                }
                            }}
                            style={{ width: '100%', padding: '6px 12px', background: 'rgba(255,255,255,0.04)', border: 'none', color: '#9db4d8', fontSize: 12, textAlign: 'left', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                        >
                            <span>📋 可选因子</span>
                            <span style={{ opacity: 0.8 }}>{availableFactorsPanelOpen ? '▼ 收起' : '▶ 展开'}</span>
                        </button>
                        {availableFactorsPanelOpen && (
                            <div style={{ maxHeight: 280, overflowY: 'auto', padding: 8, background: 'rgba(0,0,0,0.15)', fontSize: 11 }}>
                                {availableFactorsList === null && <div style={{ color: 'rgba(255,255,255,0.5)' }}>加载中…</div>}
                                {Array.isArray(availableFactorsList) && availableFactorsList.length === 0 && <div style={{ color: 'rgba(255,255,255,0.5)' }}>暂无数据</div>}
                                {Array.isArray(availableFactorsList) && availableFactorsList.length > 0 && (
                                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                        <thead>
                                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
                                                <th style={{ padding: '4px 6px', textAlign: 'left', color: '#9db4d8' }}>因子ID</th>
                                                <th style={{ padding: '4px 6px', textAlign: 'left', color: '#9db4d8' }}>中文名</th>
                                                <th style={{ padding: '4px 6px', textAlign: 'left', color: '#9db4d8' }}>类别</th>
                                                <th style={{ padding: '4px 6px', textAlign: 'left', color: '#9db4d8' }}>描述</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {availableFactorsList.map((f, i) => (
                                                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                                    <td style={{ padding: '4px 6px', fontFamily: 'monospace', color: '#c8e0ff' }}>{f.id}</td>
                                                    <td style={{ padding: '4px 6px', color: 'rgba(255,255,255,0.9)' }}>{f.name_zh}</td>
                                                    <td style={{ padding: '4px 6px', color: 'rgba(255,255,255,0.7)' }}>{f.category_label_zh || f.category}{f.sub_category_label_zh ? ` / ${f.sub_category_label_zh}` : (f.sub_category ? ` / ${f.sub_category}` : '')}</td>
                                                    <td style={{ padding: '4px 6px', color: 'rgba(255,255,255,0.6)', maxWidth: 200 }}>{f.description}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="actions">
                        <button
                            className="button"
                            disabled={deepSearchBusy || !!getDeepSearchValidationError()}
                            onClick={runDeepSearch}
                            title={getDeepSearchValidationError() || ''}
                        >
                            {deepSearchBusy ? '搜索中…' : '开始深度搜索'}
                        </button>
                        {deepSearchBusy && (
                            <button type="button" className="button" onClick={stopDeepSearch} style={{ marginLeft: 8 }}>
                                停止搜索
                            </button>
                        )}
                    </div>
                    {deepSearchBusy && (
                        <div style={{ marginTop: 10 }}>
                            <div className="label" style={{ marginBottom: 4 }}>进度</div>
                            <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.9)', marginBottom: 6 }}>
                                {deepSearchProgress.total > 0
                                    ? `${deepSearchProgress.current} / ${deepSearchProgress.total}：${deepSearchProgress.message || '...'}`
                                    : (deepSearchProgress.message || '准备中…')}
                            </div>
                            <div className="progressBar">
                                <div
                                    className="progressFill"
                                    style={{ width: `${deepSearchProgress.total > 0 ? Math.round((100 * deepSearchProgress.current) / deepSearchProgress.total) : 0}%` }}
                                />
                            </div>
                        </div>
                    )}
                    {/* 可展开：Agent 思考与决策过程（LLM 输入/输出） */}
                    {(deepSearchBusy || agentLogs.length > 0) && (
                        <div style={{ marginTop: 10, border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, overflow: 'hidden' }}>
                            <button
                                type="button"
                                onClick={() => setAgentLogPanelOpen(prev => !prev)}
                                style={{ width: '100%', padding: '8px 12px', background: 'rgba(100,200,255,0.08)', border: 'none', color: '#9db4d8', fontSize: 12, textAlign: 'left', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                            >
                                <span>🧠 Agent 思考与决策</span>
                                <span style={{ opacity: 0.8 }}>{agentLogPanelOpen ? '▼ 收起' : '▶ 展开'}</span>
                            </button>
                            {agentLogPanelOpen && (
                                <div style={{ maxHeight: 360, overflowY: 'auto', padding: 10, background: 'rgba(0,0,0,0.2)', fontSize: 11 }}>
                                    {agentLogs.length === 0 && deepSearchBusy && <div style={{ color: 'rgba(255,255,255,0.5)' }}>等待编排/审查 Agent 调用…</div>}
                                    {agentLogs.map((entry, i) => (
                                        <div key={i} style={{ marginBottom: 10, borderLeft: '3px solid ' + (entry.role === 'orchestration' ? '#7ecfff' : entry.role === 'reviewer' ? '#ffd080' : entry.role === 'evaluation' ? '#a0e8b0' : '#b0c8e8'), paddingLeft: 8, background: 'rgba(255,255,255,0.03)', borderRadius: 4 }}>
                                            <div style={{ color: '#9db4d8', marginBottom: 4 }}>
                                                {entry.role === 'orchestration' ? '编排 Agent' : entry.role === 'reviewer' ? '审查 Agent' : entry.role === 'evaluation' ? '评价结果' : entry.role} · {entry.phase === 'input' ? '输入' : entry.phase === 'output' ? '输出' : entry.phase === 'trial_result' ? '单组结果' : entry.phase}
                                            </div>
                                            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'rgba(255,255,255,0.85)', fontFamily: 'inherit', fontSize: 11 }}>{entry.content}</pre>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                    {deepSearchError && (
                        <div className="errorText" style={{ marginTop: 8 }}>{deepSearchError}</div>
                    )}
                    {deepSearchResult && (deepSearchResult.best || deepSearchResult.stopped || deepSearchResult.error) && (
                        <div style={{ marginTop: 12 }}>
                            {deepSearchResult.error && !deepSearchResult.best && (
                                <div style={{ marginBottom: 12, padding: 10, background: 'rgba(255,80,80,0.15)', borderRadius: 8, border: '1px solid rgba(255,80,80,0.4)', color: '#ff9090', fontSize: 13 }}>
                                    <strong>工作流异常：</strong>{deepSearchResult.error}
                                </div>
                            )}
                            {deepSearchResult.error && deepSearchResult.best && (
                                <div style={{ marginBottom: 12, padding: 10, background: 'rgba(255,200,60,0.12)', borderRadius: 8, border: '1px solid rgba(255,200,60,0.35)', color: '#ffd080', fontSize: 13 }}>
                                    <strong>注意：</strong>{deepSearchResult.error}
                                </div>
                            )}
                            {/* 审查结论：置顶醒目展示（Agent 工作流必有；无 reviewer 时也占位） */}
                            {(() => {
                                const rv = deepSearchResult.reviewer
                                const hasReviewer = rv && typeof rv === 'object'
                                const verdictRaw = hasReviewer ? (rv.verdict || '') : ''
                                const verdict = verdictRaw === 'reject' ? '不推荐' : verdictRaw === 'recommend' ? '推荐' : verdictRaw
                                const score = hasReviewer ? rv.quality_score : null
                                const verdictColor = verdict === '推荐' ? '#80e8a0' : verdict === '谨慎推荐' ? '#ffe080' : verdict === '不推荐' ? '#ff9090' : '#b0c8e8'
                                const verdictBg = verdict === '推荐' ? 'rgba(80,200,120,0.25)' : verdict === '谨慎推荐' ? 'rgba(255,200,60,0.2)' : verdict === '不推荐' ? 'rgba(255,80,80,0.2)' : 'rgba(180,210,255,0.15)'
                                if (hasReviewer && (verdict || score != null || rv.cap_recommendation)) {
                                    return (
                                        <div style={{ marginBottom: 12, padding: '10px 14px', background: verdictBg, borderRadius: 10, border: `1px solid ${verdictColor}50`, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                                            <span style={{ fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.85)' }}>审查结论</span>
                                            {verdict && <span style={{ fontSize: 14, fontWeight: 700, color: verdictColor }}>{verdict}</span>}
                                            {score != null && <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.8)' }}>质量 {Number(score).toFixed(1)}/10</span>}
                                            {rv.cap_recommendation && <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.7)' }}>适用：{rv.cap_recommendation}</span>}
                                        </div>
                                    )
                                }
                                if (deepSearchResult.best && (deepSearchResult.orchestrated || deepSearchResult.agent_driven)) {
                                    return (
                                        <div style={{ marginBottom: 12, padding: '10px 14px', background: 'rgba(255,180,80,0.12)', borderRadius: 10, border: '1px solid rgba(255,160,60,0.35)', fontSize: 13, color: '#ffd080' }}>
                                            审查结论：当前为 Agent 工作流，但未返回审查报告。请确认后端已正常调用审查 Agent 或查看「Agent 思考与决策」中审查输出。
                                        </div>
                                    )
                                }
                                if (deepSearchResult.best && !deepSearchResult.reviewer) {
                                    return (
                                        <div style={{ marginBottom: 12, padding: '8px 12px', background: 'rgba(255,255,255,0.05)', borderRadius: 8, fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>
                                            审查结论：工作流模式，无审查 Agent 报告。若需审查结论，请勾选「使用 Agent 工作流」后重新搜索。
                                        </div>
                                    )
                                }
                                return null
                            })()}
                            {deepSearchResult.best && (() => {
                                const annAlpha = deepSearchResult.annualized_alpha ?? deepSearchResult.best?.backtest_stats?.annualized_alpha
                                const maxDd = deepSearchResult.backtest_stats?.max_drawdown ?? deepSearchResult.best?.backtest_stats?.max_drawdown
                                const badAlpha = annAlpha != null && Number(annAlpha) <= 0
                                const badDrawdown = maxDd != null && Number(maxDd) < -0.20
                                if (!badAlpha && !badDrawdown) return null
                                return (
                                    <div style={{ marginBottom: 12, padding: '10px 14px', background: 'rgba(255,120,80,0.15)', borderRadius: 8, border: '1px solid rgba(255,100,80,0.4)', fontSize: 13, color: '#ffb090' }}>
                                        ⚠ 当前结果较差：{badAlpha ? '年化 Alpha 非正，超额收益不足。' : ''}{badDrawdown ? ' 最大回撤超过 20%，波动较大。' : ''} 建议谨慎使用或重新调整参数/因子后再挖掘。
                                    </div>
                                )
                            })()}
                            {(deepSearchResult.workflow || deepSearchResult.alpha != null || deepSearchResult.beta != null || deepSearchResult.strategy_logic || deepSearchResult.rotation_logic) && (
                                <>
                                <div className="cardMeta" style={{ marginBottom: 8, fontSize: 14 }}>工作摘要</div>
                                <div style={{ marginBottom: 12, padding: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.12)' }}>
                                    {deepSearchResult.workflow && <div style={{ marginBottom: 6 }}><span style={{ opacity: 0.8 }}>模式</span> LangGraph 工作流 + LLM 选因子{deepSearchResult.orchestrated ? '（Agent 逐步编排）' : ''}</div>}
                                    {(deepSearchResult.alpha != null || deepSearchResult.beta != null) && (
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
                                            {deepSearchResult.alpha != null && <span><span style={{ opacity: 0.8 }} title="日度 Alpha，年化 Alpha = 日度 Alpha × 252">Alpha(日)</span> <strong>{Number(deepSearchResult.alpha).toFixed(6)}</strong></span>}
                                            {deepSearchResult.beta != null && <span><span style={{ opacity: 0.8 }}>Beta</span> <strong>{Number(deepSearchResult.beta).toFixed(4)}</strong></span>}
                                            {deepSearchResult.annualized_alpha != null && <span><span style={{ opacity: 0.8 }} title="年化 Alpha = Alpha(日) × 252">年化 Alpha</span> <strong>{Number(deepSearchResult.annualized_alpha).toFixed(4)}</strong></span>}
                                            {deepSearchResult.r_squared != null && <span><span style={{ opacity: 0.8 }}>R²</span> <strong>{Number(deepSearchResult.r_squared).toFixed(4)}</strong></span>}
                                        </div>
                                    )}
                                    {(() => {
                                        const stats = deepSearchResult.backtest_stats || deepSearchResult.best?.backtest_stats
                                        if (!stats || typeof stats !== 'object') return null
                                        return (
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
                                                {stats.total_return != null && <span><span style={{ opacity: 0.8 }}>总收益</span> <strong>{(Number(stats.total_return) * 100).toFixed(2)}%</strong></span>}
                                                {stats.sharpe_annual != null && <span><span style={{ opacity: 0.8 }}>夏普(年化)</span> <strong>{Number(stats.sharpe_annual).toFixed(4)}</strong></span>}
                                                {stats.max_drawdown != null && <span><span style={{ opacity: 0.8 }}>最大回撤</span> <strong>{(Number(stats.max_drawdown) * 100).toFixed(2)}%</strong></span>}
                                            </div>
                                        )
                                    })()}
                                    {deepSearchResult.strategy_logic && (
                                        <div style={{ marginBottom: 8 }}>
                                            <div className="label" style={{ marginBottom: 4 }}>策略逻辑</div>
                                            <div className="cardText" style={{ margin: 0, fontSize: 13, whiteSpace: 'pre-wrap' }}>{deepSearchResult.strategy_logic}</div>
                                        </div>
                                    )}
                                    {deepSearchResult.rotation_logic && (
                                        <div>
                                            <div className="label" style={{ marginBottom: 4 }}>轮仓逻辑</div>
                                            <div className="cardText" style={{ margin: 0, fontSize: 13, whiteSpace: 'pre-wrap' }}>{deepSearchResult.rotation_logic}</div>
                                        </div>
                                    )}
                                </div>
                                </>
                            )}
                            {/* ── 审查 Agent 报告 ── */}
                            {deepSearchResult.reviewer && typeof deepSearchResult.reviewer === 'object' && (() => {
                                const rv = deepSearchResult.reviewer
                                const score = rv.quality_score
                                const verdictRaw = rv.verdict || ''
                                const verdict = verdictRaw === 'reject' ? '不推荐' : verdictRaw === 'recommend' ? '推荐' : verdictRaw
                                const verdictColor = verdict === '推荐' ? '#80e8a0' : verdict === '谨慎推荐' ? '#ffe080' : verdict === '不推荐' ? '#ff9090' : '#b0c8e8'
                                const verdictBg = verdict === '推荐' ? 'rgba(80,200,120,0.2)' : verdict === '谨慎推荐' ? 'rgba(255,200,60,0.18)' : verdict === '不推荐' ? 'rgba(255,80,80,0.18)' : 'rgba(180,210,255,0.12)'
                                const cap = rv.cap_recommendation || ''
                                const capIcon = cap === '大盘' ? '🏦' : cap === '中小盘' ? '📊' : '🌐'
                                const capColor = cap === '大盘' ? '#7ecfff' : cap === '中小盘' ? '#ffd080' : '#a0d0b0'
                                return (
                                <div style={{ marginBottom: 14, padding: 12, background: 'rgba(100,200,255,0.07)', borderRadius: 10, border: '1px solid rgba(100,200,255,0.25)' }}>
                                    {/* 标题行 */}
                                    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                                        <span style={{ fontSize: 13, fontWeight: 600, color: '#7ecfff' }}>🔍 审查 Agent 报告</span>
                                        {/* 结论 Badge */}
                                        {verdict && (
                                            <span style={{ fontSize: 13, fontWeight: 700, padding: '3px 12px', borderRadius: 14, background: verdictBg, color: verdictColor, border: `1px solid ${verdictColor}40` }}>
                                                {verdict}
                                            </span>
                                        )}
                                        {/* 质量分 */}
                                        {score != null && (
                                            <span style={{ fontSize: 12, padding: '2px 10px', borderRadius: 12, background: score >= 7 ? 'rgba(80,200,120,0.2)' : score >= 5 ? 'rgba(255,200,60,0.15)' : 'rgba(255,80,80,0.15)', color: score >= 7 ? '#80e8a0' : score >= 5 ? '#ffe080' : '#ff9090' }}>
                                                质量 {Number(score).toFixed(1)}/10
                                            </span>
                                        )}
                                        {/* 可靠性 */}
                                        {rv.reliability && (
                                            <span style={{ fontSize: 12, padding: '2px 10px', borderRadius: 12, background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.7)' }}>
                                                可靠性：{rv.reliability}
                                            </span>
                                        )}
                                        {/* 市值适用范围 */}
                                        {cap && (
                                            <span style={{ fontSize: 12, fontWeight: 600, padding: '2px 10px', borderRadius: 12, background: 'rgba(255,255,255,0.06)', color: capColor, border: `1px solid ${capColor}50` }}>
                                                {capIcon} 适合：{cap}
                                            </span>
                                        )}
                                    </div>
                                    {/* 本组合实际指标（与挖掘报告一致），避免评审意见中 LLM 引用错 trial 导致不一致 */}
                                    {rv.selected_trial_metrics && typeof rv.selected_trial_metrics === 'object' && (() => {
                                        const m = rv.selected_trial_metrics
                                        const alpha = m.alpha != null ? Number(m.alpha) : null
                                        const beta = m.beta != null ? Number(m.beta) : null
                                        const annAlpha = m.annualized_alpha != null ? Number(m.annualized_alpha) : null
                                        const r2 = m.r_squared != null ? Number(m.r_squared) : null
                                        const totalRet = m.total_return != null ? Number(m.total_return) : null
                                        const sharpe = m.sharpe_annual != null ? Number(m.sharpe_annual) : null
                                        const maxDd = m.max_drawdown != null ? Number(m.max_drawdown) : null
                                        if (alpha == null && beta == null && annAlpha == null && sharpe == null) return null
                                        return (
                                            <div style={{ marginBottom: 10, padding: '8px 10px', background: 'rgba(80,180,120,0.08)', borderRadius: 8, border: '1px solid rgba(80,200,120,0.3)', fontSize: 12 }}>
                                                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)', marginBottom: 6 }}>本组合实际指标（与挖掘报告一致）</div>
                                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px 16px', color: 'rgba(255,255,255,0.85)' }}>
                                                    {alpha != null && <span>Alpha(日) {alpha.toFixed(6)}</span>}
                                                    {beta != null && <span>Beta {beta.toFixed(4)}</span>}
                                                    {annAlpha != null && <span>年化 Alpha {annAlpha.toFixed(4)}</span>}
                                                    {r2 != null && <span>R² {r2.toFixed(4)}</span>}
                                                    {totalRet != null && <span>总收益 {(totalRet * 100).toFixed(2)}%</span>}
                                                    {sharpe != null && <span>夏普(年化) {sharpe.toFixed(4)}</span>}
                                                    {maxDd != null && <span>最大回撤 {(maxDd * 100).toFixed(2)}%</span>}
                                                </div>
                                            </div>
                                        )
                                    })()}
                                    {/* 策略逻辑 */}
                                    {rv.strategy_logic && (
                                        <div style={{ marginBottom: 8, padding: '7px 10px', background: 'rgba(255,255,255,0.04)', borderRadius: 6, fontSize: 12, color: 'rgba(255,255,255,0.8)', lineHeight: 1.7, borderLeft: '3px solid rgba(126,207,255,0.4)' }}>
                                            <span style={{ fontSize: 11, color: '#7ecfff', display: 'block', marginBottom: 3 }}>策略逻辑</span>
                                            {rv.strategy_logic}
                                        </div>
                                    )}
                                    {/* 评审意见 */}
                                    {(rv.comments || []).length > 0 && (
                                        <div style={{ marginBottom: 8 }}>
                                            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', marginBottom: 4 }}>评审意见</div>
                                            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.9 }}>
                                                {rv.comments.map((c, i) => <li key={i}>{c}</li>)}
                                            </ul>
                                        </div>
                                    )}
                                    {/* 轮仓说明 */}
                                    {rv.rotation_logic && (
                                        <div style={{ marginBottom: 8, fontSize: 12, color: 'rgba(255,255,255,0.6)', lineHeight: 1.7 }}>
                                            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>调仓：</span>{rv.rotation_logic}
                                        </div>
                                    )}
                                    {/* 风险提示 */}
                                    {(rv.risks || []).length > 0 && (
                                        <div>
                                            <div style={{ fontSize: 11, color: 'rgba(255,180,60,0.85)', marginBottom: 4 }}>⚠ 风险提示</div>
                                            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.9, color: '#ffd080' }}>
                                                {rv.risks.map((r, i) => <li key={i}>{r}</li>)}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                                )
                            })()}
                            {deepSearchResult.stopped && !deepSearchResult.best && (
                                <div className="cardMeta" style={{ marginBottom: 6 }}>已停止，未得到完整结果</div>
                            )}
                            {deepSearchResult.stopped && deepSearchResult.best && (
                                <div className="cardMeta" style={{ marginBottom: 6 }}>已停止，当前最佳组合</div>
                            )}
                            {!deepSearchResult.stopped && deepSearchResult.best && (
                                <div className="cardMeta" style={{ marginBottom: 6 }}>最佳因子组合</div>
                            )}
                            {deepSearchResult.best && (
                            <>
                            {/* ── 因子详情表 ── */}
                            {(() => {
                                const best = deepSearchResult.best
                                const combo = best.best_factor_combo || []
                                const weights = best.learned_weights || {}
                                const quality = best.factor_quality || {}
                                const bellList = best.bell_transforms || []
                                if (!combo.length) return null
                                // 归一化权重：相对贡献百分比（基于绝对值之和）
                                const absSum = combo.reduce((s, f) => s + Math.abs(weights[f] ?? 0), 0)
                                return (
                                <div style={{ marginBottom: 10, overflowX: 'auto' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                                        <thead>
                                            <tr style={{ background: '#1a2540' }}>
                                                {['因子名','钟形变换','权重(相对贡献)','Spread','IC','IC_IR','方向'].map(h => (
                                                    <th key={h} style={{ position: 'sticky', top: 0, background: '#1a2540', padding: '5px 8px', textAlign: h === '权重(相对贡献)' || h === 'Spread' || h === 'IC' || h === 'IC_IR' ? 'right' : 'left', fontWeight: 500, color: '#9db4d8', whiteSpace: 'nowrap', zIndex: 1 }}>{h}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {combo.map((f, i) => {
                                                const w = weights[f] ?? 0
                                                const pct = absSum > 1e-9 ? (w / absSum * 100) : 0
                                                const baseName = f.endsWith('_bell') ? f.slice(0, -5) : f
                                                const q = quality[f] || quality[baseName] || {}
                                                const isBell = f.endsWith('_bell') || bellList.includes(f.replace(/_bell$/, ''))
                                                const dirLabel = q.direction === 'up' ? '正' : q.direction === 'down' ? '反' : q.direction === 'bell' ? '钟形' : q.direction === 'mixed' ? '混合' : (q.direction && q.direction !== 'unknown' ? q.direction : '—')
                                                return (
                                                <tr key={f} style={{ borderTop: '1px solid rgba(255,255,255,0.06)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.025)' }}>
                                                    <td style={{ padding: '5px 8px', fontFamily: 'monospace', color: '#c8e0ff' }}>{baseName}</td>
                                                    <td style={{ padding: '5px 8px', textAlign: 'center' }}>
                                                        {isBell
                                                            ? <span title="(x − 截面均值)²" style={{ color: '#ffd080', fontSize: 11, padding: '1px 6px', background: 'rgba(255,200,60,0.15)', borderRadius: 4 }}>钟形 (x−μ)²</span>
                                                            : <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 11 }}>原始</span>}
                                                    </td>
                                                    <td style={{ padding: '5px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: w > 0 ? '#80e8a0' : w < 0 ? '#ff9090' : '#ccc', fontWeight: 600 }}>
                                                        <span title={`原始权重: ${w.toFixed(6)}`}>
                                                            {pct !== 0 ? (pct > 0 ? '+' : '') + pct.toFixed(1) + '%' : '0%'}
                                                        </span>
                                                        <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', marginLeft: 4 }}>
                                                            ({w > 0 ? '+' : ''}{Number(w).toFixed(4)})
                                                        </span>
                                                    </td>
                                                    <td style={{ padding: '5px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: 'rgba(255,255,255,0.75)' }}>{q.spread != null ? Number(q.spread).toFixed(4) : '—'}</td>
                                                    <td style={{ padding: '5px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: (q.ic || 0) > 0.01 ? '#80e8a0' : (q.ic || 0) < -0.01 ? '#ff9090' : 'rgba(255,255,255,0.5)' }}>
                                                        {q.ic != null ? Number(q.ic).toFixed(4) : '—'}
                                                    </td>
                                                    <td style={{ padding: '5px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: 'rgba(255,255,255,0.75)' }}>{q.ic_ir != null ? Number(q.ic_ir).toFixed(3) : '—'}</td>
                                                    <td style={{ padding: '5px 8px' }}>
                                                        <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 3, background: q.direction === 'up' ? 'rgba(80,200,120,0.2)' : q.direction === 'down' ? 'rgba(255,100,100,0.2)' : q.direction === 'bell' ? 'rgba(255,200,60,0.2)' : 'rgba(255,255,255,0.08)', color: q.direction === 'up' ? '#80e8a0' : q.direction === 'down' ? '#ff9090' : q.direction === 'bell' ? '#ffd080' : '#aaa' }}>
                                                            {dirLabel}
                                                        </span>
                                                    </td>
                                                </tr>
                                                )
                                            })}
                                        </tbody>
                                    </table>
                                    <div style={{ marginTop: 5, fontSize: 11, color: 'rgba(255,255,255,0.4)', display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                                        <span>步长 {best.label_horizon}日 · 调仓 {best.rebalance_freq}日</span>
                                        {best.top_n != null && <span>TopN={best.top_n}</span>}
                                        {best.metrics?.val_rank_ic != null && <span>val IC={Number(best.metrics.val_rank_ic).toFixed(4)}</span>}
                                        {best.backtest_stats?.sharpe_annual != null && <span>夏普={Number(best.backtest_stats.sharpe_annual).toFixed(3)}</span>}
                                        {best.backtest_stats?.total_return != null && <span>总收益={( Number(best.backtest_stats.total_return)*100).toFixed(1)}%</span>}
                                        {best.backtest_stats?.max_drawdown != null && <span>最大回撤={(Number(best.backtest_stats.max_drawdown)*100).toFixed(1)}%</span>}
                                    </div>
                                </div>
                                )
                            })()}
                            {/* ── 回测控件 ── */}
                            <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', marginBottom: 8 }}>
                                <div style={{ fontSize: 11, color: '#9db4d8', marginBottom: 4, fontWeight: 500 }}>回测参数</div>
                                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', marginBottom: 8 }}>与挖掘一致时可复现「本组合实际指标」。若故意用不同参数验证鲁棒性：少许改动（如 TopN 10→5、调仓 1→8 日）若导致收益从大幅正收益变为大幅亏损，可能表示策略对参数或调仓周期敏感、存在过拟合风险，建议用多组参数对比收益区间以评估稳健性。</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>回测股票池</span>
                                        <select className="select" style={{ width: 160 }} value={backtestPoolMode} onChange={e => setBacktestPoolMode(e.target.value)}>
                                            <option value="same">与挖掘一致</option>
                                            <option value="index_000300">沪深300(000300)</option>
                                            <option value="index_000016">上证50(000016)</option>
                                            <option value="manual">手动输入</option>
                                        </select>
                                    </span>
                                    {backtestPoolMode === 'manual' && (
                                        <input type="text" className="input" placeholder="股票代码，逗号分隔" style={{ minWidth: 180, fontSize: 12 }} value={backtestManualStocks} onChange={e => setBacktestManualStocks(e.target.value)} />
                                    )}
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>TopN</span>
                                        <select className="select" style={{ width: 62 }} value={backtestTopN} onChange={e => setBacktestTopN(Number(e.target.value))}>
                                            {[3,5,8,10,15,20,30].map(n => <option key={n} value={n}>{n}</option>)}
                                        </select>
                                    </span>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>调仓周期(日)</span>
                                        <input type="number" className="input" style={{ width: 52 }} min={1} value={backtestRebalanceFreq} onChange={e => setBacktestRebalanceFreq(e.target.value)} />
                                    </span>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>回测天数</span>
                                        <select className="select" style={{ width: 82 }} value={backtestDays} onChange={e => setBacktestDays(Number(e.target.value))}>
                                            <option value={63}>3 个月</option>
                                            <option value={126}>半 年</option>
                                            <option value={252}>1 年</option>
                                            <option value={504}>2 年</option>
                                            <option value={756}>3 年</option>
                                        </select>
                                    </span>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>个股权重</span>
                                        <select className="select" style={{ width: 110 }} value={backtestPositionWeight} onChange={e => setBacktestPositionWeight(e.target.value)}>
                                            <option value="equal">等权</option>
                                            <option value="score_weighted">按得分加权</option>
                                            <option value="kelly">凯利公式</option>
                                        </select>
                                    </span>
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10 }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: 12 }}>
                                        <input type="checkbox" checked={backtestExcludeKechuang} onChange={e => setBacktestExcludeKechuang(e.target.checked)} />
                                        排除科创板
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: 12 }} title="同一数据下多组 TopN/调仓周期回测，评估参数敏感性">
                                        <input type="checkbox" checked={robustnessCheck} onChange={e => setRobustnessCheck(e.target.checked)} />
                                        稳健性检查
                                    </label>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>市值筛选</span>
                                        <select className="select" style={{ width: 118 }} value={backtestCapScope} onChange={e => setBacktestCapScope(e.target.value)}>
                                            <option value="none">不筛</option>
                                            <option value="only_small_cap">仅小市值</option>
                                            <option value="exclude_small_cap">排除小市值</option>
                                        </select>
                                    </span>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                        <span style={{ fontSize: 12, opacity: 0.7 }}>阈值(亿)</span>
                                        <input type="number" className="input" style={{ width: 68 }} value={backtestSmallCapMaxBillion} onChange={e => setBacktestSmallCapMaxBillion(e.target.value)} />
                                    </span>
                                </div>
                                {factorBacktestLoading && (
                                    <div style={{ marginBottom: 10 }}>
                                        <div style={{ fontSize: 11, color: '#9db4d8', marginBottom: 4 }}>{backtestProgress.message || '回测进行中…'}</div>
                                        <div className="progressBar" style={{ height: 6, borderRadius: 3, overflow: 'hidden', background: 'rgba(255,255,255,0.1)' }}>
                                            <div style={{ height: '100%', width: `${Math.min(100, Math.max(0, backtestProgress.pct))}%`, transition: 'width 0.3s ease', background: 'rgba(126,207,255,0.8)', borderRadius: 3 }} />
                                        </div>
                                    </div>
                                )}
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                                    <button className="buttonPrimary" onClick={loadDeepSearchResultToMining} style={{ fontSize: 12 }}>
                                        加载到挖掘
                                    </button>
                                    <button
                                        type="button"
                                        className="button"
                                        disabled={factorBacktestLoading || !(deepSearchResult?.best?.best_factor_combo?.length)}
                                        onClick={runFactorBacktest}
                                    >
                                        {factorBacktestLoading ? '回测中…' : '执行回测'}
                                    </button>
                                    {(factorBacktestResult && !factorBacktestResult.error && (factorBacktestResult.backtest_stats || factorBacktestResult.alpha != null)) && (
                                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <input type="text" className="input" placeholder="摘要标题" style={{ width: 100, fontSize: 11 }} value={saveSummaryTitle} onChange={e => setSaveSummaryTitle(e.target.value)} />
                                            <button type="button" className="button" style={{ fontSize: 11 }} onClick={async () => {
                                                const best = deepSearchResult?.best
                                                const combo = best?.best_factor_combo || factorBacktestResult?.factor_combo || []
                                                const rawWeights = best?.learned_weights?.flat != null ? best.learned_weights.flat : (best?.learned_weights || factorBacktestResult?.weights || {})
                                                const weights = typeof rawWeights === 'object' && rawWeights !== null && !Array.isArray(rawWeights) ? { ...rawWeights } : {}
                                                if (!combo.length) return
                                                const alpha = factorBacktestResult?.alpha ?? deepSearchResult?.alpha
                                                const beta = factorBacktestResult?.beta ?? deepSearchResult?.beta
                                                const annualized_alpha = factorBacktestResult?.annualized_alpha ?? deepSearchResult?.annualized_alpha
                                                const r_squared = factorBacktestResult?.r_squared ?? deepSearchResult?.r_squared
                                                try {
                                                    const base = buildDeepSearchPayload()
                                                    await poolApi.backtestSummarySave({
                                                        title: (saveSummaryTitle || '').trim() || `回测 ${combo.join(', ')}`,
                                                        factor_combo: combo,
                                                        weights,
                                                        backtest_stats: factorBacktestResult?.backtest_stats || deepSearchResult?.backtest_stats || {},
                                                        alpha: alpha != null ? Number(alpha) : undefined,
                                                        beta: beta != null ? Number(beta) : undefined,
                                                        annualized_alpha: annualized_alpha != null ? Number(annualized_alpha) : undefined,
                                                        r_squared: r_squared != null ? Number(r_squared) : undefined,
                                                        alpha_beta: (alpha != null || beta != null) ? {
                                                            alpha: alpha != null ? Number(alpha) : undefined,
                                                            beta: beta != null ? Number(beta) : undefined,
                                                            annualized_alpha: annualized_alpha != null ? Number(annualized_alpha) : undefined,
                                                            r_squared: r_squared != null ? Number(r_squared) : undefined
                                                        } : null,
                                                        position_weight_method: backtestPositionWeight || 'equal',
                                                        label_horizon: Number(deepSearchResult?.best?.label_horizon) || deepSearchLabelHorizon || 5,
                                                        rebalance_freq: Number(backtestRebalanceFreq) || 1,
                                                        top_n: Number(backtestTopN) || 10,
                                                        days: Number(backtestDays) || base.days || 252,
                                                        universe_source: base.universe_source || '',
                                                        universe_index: base.universe_index || '',
                                                        benchmark_code: base.benchmark_code || '510300',
                                                        strategy_logic: deepSearchResult?.strategy_logic || '',
                                                        rotation_logic: deepSearchResult?.rotation_logic || '',
                                                        rebalance_details_count: Array.isArray(factorBacktestResult?.rebalance_details) ? factorBacktestResult.rebalance_details.length : 0,
                                                        max_drawdown: factorBacktestResult?.backtest_stats?.max_drawdown,
                                                        total_return: factorBacktestResult?.backtest_stats?.total_return,
                                                        sharpe_annual: factorBacktestResult?.backtest_stats?.sharpe_annual
                                                    })
                                                    setSaveSummaryTitle('')
                                                    const data = await poolApi.backtestSummariesList()
                                                    if (Array.isArray(data?.result)) setSavedSummaries(data.result)
                                                } catch (e) {
                                                    console.warn('保存摘要失败', e)
                                                }
                                            }}>
                                                保存摘要
                                            </button>
                                        </span>
                                    )}
                                </div>
                                {savedSummaries.length > 0 && (
                                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                        <div style={{ fontSize: 11, color: '#9db4d8', marginBottom: 6 }}>已保存的回测摘要</div>
                                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.8 }}>
                                            {savedSummaries.slice(0, 20).map(s => (
                                                <li key={s.id} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                                    <span style={{ color: 'rgba(255,255,255,0.85)' }}>{s.title || (s.factor_combo || []).join(', ')}</span>
                                                    <span style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11 }}>
                                                        {s.backtest_stats?.sharpe_annual != null ? `夏普 ${Number(s.backtest_stats.sharpe_annual).toFixed(3)}` : ''}
                                                        {s.backtest_stats?.total_return != null ? ` 收益 ${(Number(s.backtest_stats.total_return) * 100).toFixed(1)}%` : ''}
                                                        {(s.alpha != null || s.beta != null) && (
                                                            <span style={{ marginLeft: 6, color: 'rgba(126,207,255,0.85)' }}>
                                                                α {s.alpha != null ? Number(s.alpha).toFixed(4) : '-'} β {s.beta != null ? Number(s.beta).toFixed(3) : '-'}
                                                            </span>
                                                        )}
                                                        {s.weights && Object.keys(s.weights).length > 0 && (
                                                            <span style={{ marginLeft: 4, color: 'rgba(255,255,255,0.4)' }} title={Object.entries(s.weights).map(([k, v]) => `${k}: ${Number(v).toFixed(3)}`).join(' ')}>
                                                                权重✓
                                                            </span>
                                                        )}
                                                    </span>
                                                    <button type="button" onClick={async () => {
                                                        try {
                                                            await poolApi.backtestSummaryDelete(s.id)
                                                            const data = await poolApi.backtestSummariesList()
                                                            if (Array.isArray(data?.result)) setSavedSummaries(data.result)
                                                        } catch (e) { console.warn(e) }
                                                    }} style={{ background: 'rgba(255,80,80,0.2)', border: '1px solid rgba(255,80,80,0.4)', color: '#ff9090', borderRadius: 4, cursor: 'pointer', fontSize: 10, padding: '2px 6px' }}>删除</button>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                            {factorBacktestResult && (
                                <div style={{ marginTop: 4, padding: 10, background: factorBacktestResult.error ? 'rgba(255,80,80,0.1)' : 'rgba(255,255,255,0.05)', borderRadius: 8, border: `1px solid ${factorBacktestResult.error ? 'rgba(255,80,80,0.3)' : 'rgba(255,255,255,0.1)'}`, fontSize: 13 }}>
                                    <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginBottom: 6 }}>
                                        回测结果（TopN={backtestTopN} · 调仓{backtestRebalanceFreq}日）
                                    </div>
                                    {factorBacktestResult.error ? (
                                        <div className="errorText">{factorBacktestResult.error}</div>
                                    ) : (
                                        <>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 8 }}>
                                            {factorBacktestResult.backtest_stats?.total_return != null && <span><span style={{ opacity: 0.7 }}>总收益</span> <strong style={{ color: factorBacktestResult.backtest_stats.total_return > 0 ? '#80e8a0' : '#ff9090' }}>{(Number(factorBacktestResult.backtest_stats.total_return) * 100).toFixed(2)}%</strong></span>}
                                            {factorBacktestResult.backtest_stats?.sharpe_annual != null && <span><span style={{ opacity: 0.7 }}>年化夏普</span> <strong>{Number(factorBacktestResult.backtest_stats.sharpe_annual).toFixed(3)}</strong></span>}
                                            {factorBacktestResult.backtest_stats?.max_drawdown != null && <span><span style={{ opacity: 0.7 }}>最大回撤</span> <strong style={{ color: '#ff9090' }}>{(Number(factorBacktestResult.backtest_stats.max_drawdown) * 100).toFixed(2)}%</strong></span>}
                                            {factorBacktestResult.alpha != null && <span><span style={{ opacity: 0.7 }} title="日度 Alpha，年化 Alpha = 日度 × 252">Alpha(日)</span> <strong>{Number(factorBacktestResult.alpha).toFixed(6)}</strong></span>}
                                            {factorBacktestResult.annualized_alpha != null && <span><span style={{ opacity: 0.7 }} title="年化 Alpha = Alpha(日) × 252">年化Alpha</span> <strong style={{ color: factorBacktestResult.annualized_alpha > 0 ? '#80e8a0' : '#ff9090' }}>{Number(factorBacktestResult.annualized_alpha).toFixed(4)}</strong></span>}
                                            {factorBacktestResult.beta != null && <span><span style={{ opacity: 0.7 }}>Beta</span> <strong>{Number(factorBacktestResult.beta).toFixed(3)}</strong></span>}
                                            {factorBacktestResult.r_squared != null && <span><span style={{ opacity: 0.7 }}>R²</span> <strong>{Number(factorBacktestResult.r_squared).toFixed(3)}</strong></span>}
                                        </div>
                                        {/* 与挖掘差异过大时提示过拟合/敏感 */}
                                        {deepSearchResult?.best?.backtest_stats?.total_return != null && factorBacktestResult.backtest_stats?.total_return != null && (() => {
                                            const miningRet = Number(deepSearchResult.best.backtest_stats.total_return)
                                            const backtestRet = Number(factorBacktestResult.backtest_stats.total_return)
                                            const signFlip = (miningRet > 0.1 && backtestRet < -0.05) || (miningRet < -0.05 && backtestRet > 0.1)
                                            if (!signFlip && Math.abs(miningRet - backtestRet) < 0.3) return null
                                            return (
                                                <div style={{ fontSize: 11, color: '#ffd080', background: 'rgba(255,200,60,0.1)', padding: '6px 10px', borderRadius: 6, marginBottom: 8, border: '1px solid rgba(255,200,60,0.25)' }}>
                                                    与挖掘结果差异较大（挖掘总收益 {(miningRet * 100).toFixed(1)}% vs 当前 {(backtestRet * 100).toFixed(1)}%）。若当前参数是故意用于验证鲁棒性，这种程度差异可能表示策略对 TopN/调仓周期或样本区间较敏感，存在过拟合风险，建议用「与挖掘一致」参数复现后，再对比多组参数下的收益区间评估稳健性。
                                                </div>
                                            )
                                        })()}
                                        {/* 收益图表 */}
                                        {factorBacktestResult.chart_base64 && (
                                            <div style={{ borderRadius: 8, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', marginBottom: 8 }}>
                                                <img src={`data:image/png;base64,${factorBacktestResult.chart_base64}`} alt="回测收益图" style={{ width: '100%', display: 'block' }} />
                                            </div>
                                        )}
                                        {/* 调仓明细 */}
                                        {Array.isArray(factorBacktestResult.rebalance_details) && factorBacktestResult.rebalance_details.length > 0 && (
                                            <div>
                                                <button type="button" onClick={() => setRebalanceDetailsOpen(v => !v)}
                                                    style={{ background: 'none', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 5, color: '#bbb', cursor: 'pointer', fontSize: 11, padding: '3px 10px', marginBottom: 5 }}>
                                                    {rebalanceDetailsOpen ? '▼' : '▶'} 调仓明细（{factorBacktestResult.rebalance_details.length} 期）
                                                </button>
                                                {rebalanceDetailsOpen && (
                                                    <div style={{ maxHeight: 300, overflowY: 'auto', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 7 }}>
                                                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                                                            <thead>
                                                                <tr>
                                                                    {['调仓日期','持仓股票','期间收益'].map((h, hi) => (
                                                                        <th key={h} style={{ position: 'sticky', top: 0, background: '#1a2540', padding: '5px 8px', textAlign: hi === 2 ? 'right' : 'left', fontWeight: 500, color: '#9db4d8', zIndex: 1, borderBottom: '1px solid rgba(255,255,255,0.1)' }}>{h}</th>
                                                                    ))}
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                {factorBacktestResult.rebalance_details.map((d, i) => (
                                                                    <tr key={i} style={{ borderTop: '1px solid rgba(255,255,255,0.05)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)' }}>
                                                                        <td style={{ padding: '4px 8px', whiteSpace: 'nowrap', color: 'rgba(255,255,255,0.6)' }}>{d.date}</td>
                                                                        <td style={{ padding: '4px 8px', color: 'rgba(255,255,255,0.8)', lineHeight: 1.6 }}>{(d.stocks || []).join('  ') || '—'}</td>
                                                                        <td style={{ padding: '4px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: 600, color: d.period_return > 0 ? '#80e8a0' : d.period_return < 0 ? '#ff9090' : '#ccc' }}>
                                                                            {d.period_return != null ? `${(d.period_return * 100).toFixed(2)}%` : '—'}
                                                                        </td>
                                                                    </tr>
                                                                ))}
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                        {/* 稳健性检查结果 */}
                                        {Array.isArray(factorBacktestResult.robustness_results) && factorBacktestResult.robustness_results.length > 0 && (
                                            <div style={{ marginTop: 10, border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6, overflow: 'hidden' }}>
                                                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)', padding: '6px 10px', background: 'rgba(0,0,0,0.2)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>稳健性检查结果</div>
                                                <div style={{ overflowX: 'auto' }}>
                                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                                                        <thead>
                                                            <tr>
                                                                {['TopN', '调仓周期(日)', '总收益(%)', '年化夏普', '最大回撤(%)'].map((h, hi) => (
                                                                    <th key={h} style={{ padding: '6px 10px', textAlign: hi >= 2 ? 'right' : 'left', fontWeight: 500, color: '#9db4d8', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>{h}</th>
                                                                ))}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {factorBacktestResult.robustness_results.map((row, i) => {
                                                                const isCurrent = Number(row.top_n) === Number(backtestTopN) && Number(row.rebalance_freq) === Number(backtestRebalanceFreq)
                                                                return (
                                                                    <tr key={i} style={{ background: isCurrent ? 'rgba(126,207,255,0.15)' : (i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.03)'), borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                                                        <td style={{ padding: '5px 10px' }}>{row.top_n}</td>
                                                                        <td style={{ padding: '5px 10px' }}>{row.rebalance_freq}</td>
                                                                        <td style={{ padding: '5px 10px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: row.total_return != null ? (row.total_return > 0 ? '#80e8a0' : '#ff9090') : '#888' }}>
                                                                            {row.total_return != null ? `${(row.total_return * 100).toFixed(2)}%` : '—'}
                                                                        </td>
                                                                        <td style={{ padding: '5px 10px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{row.sharpe_annual != null ? Number(row.sharpe_annual).toFixed(3) : '—'}</td>
                                                                        <td style={{ padding: '5px 10px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: '#ff9090' }}>{row.max_drawdown != null ? `${(row.max_drawdown * 100).toFixed(2)}%` : '—'}</td>
                                                                    </tr>
                                                                )
                                                            })}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        )}
                                        </>
                                    )}
                                </div>
                            )}
                            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 6 }}>
                                加载后步长同步，可继续调整参数重新搜索
                            </div>
                            </>
                            )}
                        </div>
                    )}
                </div>

            </div>
        </section>
    )
}
