import { request, getAuthToken } from './client.js'

/** 研究/新闻接口可能较慢，仅 News 使用固定超时；Research 使用流式接口，无自动超时，由用户停止 */
const RESEARCH_TIMEOUT_MS = 120000

export async function research(query) {
    return request('/api/research', {
        method: 'POST',
        body: { query: String(query ?? '').trim() },
        timeout: RESEARCH_TIMEOUT_MS
    })
}

/**
 * 研究 Agent 流式接口：无超时，可被 signal 取消。
 * @param {string} query - 研究问题
 * @param {{ signal?: AbortSignal, onProgress?: (data: { step?: number, message?: string }) => void }} options
 * @returns {Promise<{ result: string, steps: Array }} - 最终 result 与 steps
 */
export function researchStream(query, options = {}) {
    const { signal, onProgress } = options
    const baseURL = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE != null
        ? import.meta.env.VITE_API_BASE
        : ''
    const token = typeof getAuthToken === 'function' ? getAuthToken() : null
    const url = baseURL ? `${baseURL}/api/research/stream` : '/api/research/stream'
    const headers = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {})
    }
    return new Promise((resolve, reject) => {
        fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({ query: String(query ?? '').trim() }),
            signal
        }).then(async (res) => {
            if (!res.ok) {
                const data = await res.json().catch(() => ({}))
                throw new Error(data?.detail || `HTTP ${res.status}`)
            }
            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''
            let result = null
            let steps = []
            while (true) {
                const { value, done } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6))
                            const type = data.type
                            if (type === 'progress' && onProgress) {
                                onProgress({ step: data.step, message: data.message })
                            } else if (type === 'done') {
                                result = data.result ?? ''
                                steps = data.steps ?? []
                            } else if (type === 'error') {
                                reject(new Error(data.detail || '研究请求失败'))
                                return
                            } else if (type === 'cancelled') {
                                reject(new Error('已取消'))
                                return
                            }
                        } catch (_) { /* ignore parse error */ }
                    }
                }
            }
            resolve({ result: result ?? '', steps })
        }).catch((err) => {
            if (err?.name === 'AbortError') {
                reject(new Error('已取消'))
            } else {
                reject(err)
            }
        })
    })
}

export async function news(query) {
    return request('/api/news', {
        method: 'POST',
        body: { query: String(query ?? '').trim() },
        timeout: RESEARCH_TIMEOUT_MS
    })
}
