import React, { useMemo, useRef, useState } from 'react'
import { usePersistedList } from '../hooks/usePersistedList.js'
import { formatTs } from '../utils/format.js'
import * as researchApi from '../api/research.js'

export default function ResearchPage({ user }) {
    const { items, addItem, clear, count } = usePersistedList('my_stock:research_news', 10, user?.user_id)
    const [query, setQuery] = useState('')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState('')
    const [progress, setProgress] = useState('')
    const abortRef = useRef(null)

    const canRun = useMemo(() => query.trim().length > 0 && !busy, [query, busy])

    async function runResearch() {
        const q = query.trim()
        if (!q) return

        const controller = new AbortController()
        abortRef.current = controller
        setBusy(true)
        setError('')
        setProgress('正在连接…')
        try {
            const data = await researchApi.researchStream(q, {
                signal: controller.signal,
                onProgress: (p) => setProgress(p?.message ? `步骤 ${p.step ?? ''}：${p.message}` : '')
            })
            const result = data?.result
            const resultStr = typeof result === 'string' ? result : (result != null ? String(result) : '')
            const text = `【Research 问题】\n${q}\n\n【Research 结果】\n${resultStr}`
            addItem(text)
            setQuery('')
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
            setProgress('')
            abortRef.current = null
        }
    }

    function stopResearch() {
        if (abortRef.current) {
            abortRef.current.abort()
        }
    }

    async function runNews() {
        const q = query.trim()
        if (!q) return

        setBusy(true)
        setError('')
        try {
            const data = await researchApi.news(q)
            const result = data?.result
            const resultStr = typeof result === 'string' ? result : (result != null ? String(result) : '')
            const text = `【News 请求】\n${q}\n\n【News 结果】\n${resultStr}`
            addItem(text)
            setQuery('')
        } catch (e) {
            setError(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    return (
        <section className="panel">
            <header className="panelHeader">
                <div className="panelTitle">Research / News</div>
                <div className="panelMeta">已保存：{count}</div>
            </header>

            <div className="panelInput">
                <textarea
                    className="textarea"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="输入研究/新闻请求（Research 会分析 CNINFO 等；News 会抓取市场/个股新闻）"
                    rows={5}
                />
                <div className="actions">
                    <button type="button" className="buttonPrimary" disabled={!canRun} onClick={runResearch}>
                        {busy ? '执行中…' : '调用 Research Agent'}
                    </button>
                    {busy && abortRef.current && (
                        <button type="button" className="button" onClick={stopResearch}>
                            停止
                        </button>
                    )}
                    <button type="button" className="button" disabled={!canRun || busy} onClick={runNews}>
                        {busy ? '处理中...' : '调用 News Agent'}
                    </button>
                    <button type="button" className="button" onClick={clear} disabled={busy}>清空</button>
                </div>
                {progress ? <div className="progressText">{progress}</div> : null}
                {error ? <div className="errorText">{error}</div> : null}
            </div>

            <div className="panelList" aria-label="Research / News 结果列表">
                {items.length === 0 ? (
                    <div className="empty">暂无已保存结果</div>
                ) : (
                        items.map((it) => (
                        <article key={it.id} className="card">
                            <div className="cardMeta">{formatTs(it.ts)}</div>
                            <pre className="cardText">{typeof it === 'string' ? it : (it?.text ?? '')}</pre>
                        </article>
                    ))
                )}
            </div>
        </section>
    )
}
