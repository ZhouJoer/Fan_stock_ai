import React, { useMemo, useState } from 'react'
import { usePersistedList } from '../hooks/usePersistedList.js'
import { formatTs } from '../utils/format.js'

export function Panel({ title, storageKey, placeholder }) {
    const { items, addItem, clear, count } = usePersistedList(storageKey)
    const [text, setText] = useState('')

    const canSubmit = useMemo(() => text.trim().length > 0, [text])

    return (
        <section className="panel">
            <header className="panelHeader">
                <div className="panelTitle">{title}</div>
                <div className="panelMeta">已保存：{count}</div>
            </header>

            <div className="panelInput">
                <textarea
                    className="textarea"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder={placeholder}
                    rows={5}
                />
                <div className="actions">
                    <button
                        type="button"
                        className="buttonPrimary"
                        disabled={!canSubmit}
                        onClick={() => {
                            addItem(text)
                            setText('')
                        }}
                    >
                        保存结果
                    </button>
                    <button type="button" className="button" onClick={clear}>清空</button>
                </div>
            </div>

            <div className="panelList" aria-label={`${title} 结果列表`}>
                {items.length === 0 ? (
                    <div className="empty">暂无已保存结果</div>
                ) : (
                    items.map((it) => (
                        <article key={it.id} className="card">
                            <div className="cardMeta">{formatTs(it.ts)}</div>
                            <pre className="cardText">{it.text}</pre>
                        </article>
                    ))
                )}
            </div>
        </section>
    )
}
