import { request } from './client.js'

export async function getAccountTrades(accountId, page = 1, pageSize = 20) {
    return request(
        `/api/etf-sim/account/${accountId}/trades?page=${page}&page_size=${pageSize}`
    )
}

export async function getAccounts() {
    return request('/api/etf-sim/accounts')
}

export async function createAccount(body) {
    return request('/api/etf-sim/account/create', { method: 'POST', body })
}

export async function getAccount(accountId) {
    return request(`/api/etf-sim/account/${accountId}`)
}

export async function deleteAccount(accountId) {
    return request(`/api/etf-sim/account/${accountId}`, { method: 'DELETE' })
}

export async function etfPoolAdd(accountId, etfCode) {
    return request(
        `/api/etf-sim/account/${accountId}/etf-pool/add?etf_code=${encodeURIComponent(etfCode)}`,
        { method: 'POST' }
    )
}

export async function etfPoolRemove(accountId, etfCode, autoSell = true) {
    const q = autoSell ? '?auto_sell=true' : ''
    return request(
        `/api/etf-sim/account/${accountId}/etf-pool/${encodeURIComponent(etfCode)}${q}`,
        { method: 'DELETE' }
    )
}

export async function autoTrade(body) {
    return request('/api/etf-sim/auto-trade', { method: 'POST', body })
}

/** 模拟盘调仓建议（GET，query 在 path 中） */
export async function getAccountSuggestion(accountId, queryString) {
    return request(`/api/etf-sim/account/${accountId}/suggestion?${queryString}`)
}
