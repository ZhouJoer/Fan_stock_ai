import { request } from './client.js'

export async function getDefaultEtfs() {
    return request('/api/etf-rotation/default-etfs')
}

export async function suggestion(body) {
    return request('/api/etf-rotation/suggestion', { method: 'POST', body })
}

export async function ai(body) {
    return request('/api/etf-rotation/ai', { method: 'POST', body })
}

export function backtestStop(sessionId) {
    return request('/api/etf-rotation/backtest/stop', {
        method: 'POST',
        body: { session_id: sessionId }
    }).catch(() => ({}))
}
