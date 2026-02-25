import { request } from './client.js'

export async function agentQuery(body) {
    return request('/api/quant/agent', { method: 'POST', body })
}

export async function aiDecision(body) {
    return request('/api/quant/ai-decision', { method: 'POST', body })
}

export async function aiBacktest(body) {
    return request('/api/quant/ai-backtest', { method: 'POST', body })
}

export async function backtest(body) {
    return request('/api/quant/backtest', { method: 'POST', body })
}
