import { request, getAuthToken, setAuthToken } from './client'

let _multiUserMode = null
let _cachedUser = null

export async function checkBackendHealth() {
    try {
        const res = await request('/api/health', { timeout: 3000 })
        return !!res?.ok
    } catch {
        return false
    }
}

export async function getAuthMode() {
    if (_multiUserMode !== null) return _multiUserMode
    try {
        const res = await request('/api/auth/mode', { timeout: 5000 })
        _multiUserMode = !!res?.multi_user
        return _multiUserMode
    } catch {
        _multiUserMode = null
        return false
    }
}

export async function getCurrentUser() {
    const token = getAuthToken()
    if (!token) return null
    try {
        const res = await request('/api/auth/me', { timeout: 5000 })
        _cachedUser = res
        return res
    } catch (err) {
        if (err.code === 'UNAUTHORIZED') {
            setAuthToken(null)
        }
        return null
    }
}

export function clearAuthCache() {
    _multiUserMode = null
    _cachedUser = null
}

export async function deleteAccount(password) {
    const res = await request('/api/auth/account/delete', {
        method: 'POST',
        body: { password },
        timeout: 10000
    })
    setAuthToken(null)
    return res
}
