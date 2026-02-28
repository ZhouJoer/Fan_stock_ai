import { request } from './client.js'

export function normalizeFactorConfig(config = {}) {
    return {
        selection_mode: config.selection_mode || 'none',
        selection_top_n: Number.isFinite(Number(config.selection_top_n)) ? parseInt(config.selection_top_n, 10) : 10,
        selection_interval: Number.isFinite(Number(config.selection_interval)) ? parseInt(config.selection_interval, 10) : 0,
        factor_set: config.factor_set || 'hybrid',
        score_weights: config.score_weights || null,
        weight_source: config.weight_source || 'manual',
        model_name: config.model_name || ''
    }
}

export function encodeScoreWeights(scoreWeights) {
    if (!scoreWeights || typeof scoreWeights !== 'object') return ''
    try {
        return JSON.stringify(scoreWeights)
    } catch {
        return ''
    }
}

export async function signals(body) {
    return request('/api/pool/signals', { method: 'POST', body })
}

export async function allocation(body) {
    return request('/api/pool/allocation', { method: 'POST', body })
}

export async function simList() {
    return request('/api/pool/sim/list')
}

export async function simCreate(body) {
    return request('/api/pool/sim/create', { method: 'POST', body })
}

export async function simGet(accountId) {
    return request(`/api/pool/sim/${accountId}`)
}

export async function simRebalance(accountId, body) {
    return request(`/api/pool/sim/${accountId}/rebalance`, { method: 'POST', body })
}

export async function industryNames() {
    return request('/api/pool/industry-names')
}

/** 可选因子列表（供因子挖掘使用，便于前后端统一增减） */
export async function availableFactors() {
    return request('/api/pool/available-factors')
}

/** 数据源检查（akshare + 东方财富） */
export async function dataSourceCheck() {
    return request('/api/pool/data-source/check')
}

export function backtestStop(sessionId) {
    return request('/api/pool/backtest/stop', {
        method: 'POST',
        body: { session_id: sessionId }
    }).catch(() => ({}))
}

export async function backtest(body) {
    return request('/api/pool/backtest', { method: 'POST', body })
}

/** 深度因子组合搜索（同步）：无超时，建议用 start + EventSource(stream) + stop 以显示进度 */
export async function factorDeepSearch(body) {
    return request('/api/pool/factor-deep-search', { method: 'POST', body })
}

/** 启动深度搜索，返回 session_id，用于连接 stream 与 stop */
export async function factorDeepSearchStart(body) {
    return request('/api/pool/factor-deep-search/start', { method: 'POST', body })
}

/** 停止深度搜索 */
export async function factorDeepSearchStop(sessionId) {
    return request('/api/pool/factor-deep-search/stop', { method: 'POST', body: { session_id: sessionId } })
}

/** 因子策略仅回测（同步，无进度） */
export async function factorBacktest(body) {
    return request('/api/pool/factor-backtest', { method: 'POST', body })
}

/** 启动因子回测（后台），返回 session_id，用于连接 stream 获取进度 */
export async function factorBacktestStart(body) {
    return request('/api/pool/factor-backtest/start', { method: 'POST', body })
}

/** 批量因子回测（单次加载数据 + 单次日历遍历） */
export async function factorBacktestBatch(body) {
    return request('/api/pool/factor-backtest-batch', { method: 'POST', body })
}

/** 回测摘要：列表 */
export async function backtestSummariesList() {
    return request('/api/pool/backtest-summaries')
}

/** 回测摘要：保存 */
export async function backtestSummarySave(body) {
    return request('/api/pool/backtest-summaries', { method: 'POST', body })
}

/** 回测摘要：按 id 删除 */
export async function backtestSummaryDelete(summaryId) {
    return request(`/api/pool/backtest-summaries/${encodeURIComponent(summaryId)}`, { method: 'DELETE' })
}

export async function savePool(body) {
    return request('/api/pool/save', { method: 'POST', body })
}

export async function poolList() {
    return request('/api/pool/list')
}

export async function poolLoad(name) {
    return request(`/api/pool/load/${encodeURIComponent(name)}`)
}

export async function poolDelete(name) {
    return request(`/api/pool/delete/${encodeURIComponent(name)}`, { method: 'DELETE' })
}

export async function industryLeaders(params) {
    const q = new URLSearchParams(params).toString()
    return request(`/api/pool/industry-leaders?${q}`)
}
