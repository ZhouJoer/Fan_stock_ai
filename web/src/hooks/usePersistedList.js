import { useCallback, useEffect, useMemo, useState } from 'react'

function safeParse(jsonText, fallback) {
    try {
        const parsed = JSON.parse(jsonText)
        return parsed ?? fallback
    } catch {
        return fallback
    }
}

function nowIso() {
    return new Date().toISOString()
}

function makeId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID()
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

/**
 * 持久化列表，支持按用户隔离
 * @param {string} baseKey - 存储键前缀，如 'my_stock:etf_rotation'
 * @param {number} maxItems - 最大条目数
 * @param {string|null} userId - 用户 ID，多用户模式下用于隔离数据；null 或 'default' 使用共享键（兼容单用户）
 */
export function usePersistedList(baseKey, maxItems = 10, userId = null) {
    const storageKey = (userId && userId !== 'default')
        ? `${baseKey}:${userId}`
        : baseKey

    const [items, setItems] = useState(() => {
        const raw = localStorage.getItem(storageKey)
        const parsed = Array.isArray(safeParse(raw, [])) ? safeParse(raw, []) : []
        // 限制初始加载的数量
        return parsed.slice(0, maxItems)
    })

    useEffect(() => {
        try {
            // 限制保存的数量，防止localStorage超出大小限制
            const itemsToSave = items.slice(0, maxItems)
            localStorage.setItem(storageKey, JSON.stringify(itemsToSave))
        } catch (e) {
            console.error('[usePersistedList] localStorage保存失败:', e)
            // 如果保存失败（可能是超出大小限制），尝试只保存文本，不保存图表
            try {
                const itemsWithoutCharts = items.slice(0, maxItems).map(item => ({
                    ...item,
                    chart: null  // 移除图表数据
                }))
                localStorage.setItem(storageKey, JSON.stringify(itemsWithoutCharts))
                console.warn('[usePersistedList] 已移除图表数据以节省空间')
            } catch (e2) {
                console.error('[usePersistedList] 即使移除图表后仍无法保存:', e2)
            }
        }
    }, [items, storageKey, maxItems])

    // 账户切换时，从新 storageKey 重新加载
    useEffect(() => {
        const raw = localStorage.getItem(storageKey)
        const parsed = Array.isArray(safeParse(raw, [])) ? safeParse(raw, []) : []
        setItems(parsed.slice(0, maxItems))
    }, [storageKey, maxItems])

    const addItem = useCallback((data) => {
        // 支持字符串或对象 { text, chart }
        let text, chart
        if (typeof data === 'string') {
            text = data.trim()
            chart = null
        } else if (data && typeof data === 'object') {
            text = (data.text ?? '').trim()
            chart = data.chart || null
        } else {
            console.warn('[usePersistedList] 无效的数据格式:', data)
            return
        }

        if (!text) {
            console.warn('[usePersistedList] 文本为空，跳过保存')
            return
        }

        console.log('[usePersistedList] 添加项目:', { 
            textLength: text.length, 
            hasChart: !!chart,
            chartLength: chart ? chart.length : 0
        })

        setItems((prev) => {
            const newItems = [
                { id: makeId(), ts: nowIso(), text, chart },
                ...prev
            ]
            // 限制最大数量，自动删除最旧的记录
            return newItems.slice(0, maxItems)
        })
    }, [maxItems])

    const clear = useCallback(() => {
        setItems([])
    }, [])

    const count = useMemo(() => items.length, [items])

    return { items, addItem, clear, count }
}
