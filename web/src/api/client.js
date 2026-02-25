export const baseURL = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE != null
    ? import.meta.env.VITE_API_BASE
    : ''

const AUTH_TOKEN_KEY = 'my_stock_auth_token'
const REMEMBER_ME_KEY = 'my_stock_remember_me'

/**
 * 获取 token：勾选「记住我」时从 localStorage 读取（跨标签页、关浏览器后仍有效），否则从 sessionStorage 读取（每标签页独立）
 */
export function getAuthToken() {
    const rememberMe = localStorage.getItem(REMEMBER_ME_KEY) === '1'
    return rememberMe ? localStorage.getItem(AUTH_TOKEN_KEY) : sessionStorage.getItem(AUTH_TOKEN_KEY)
}

/**
 * 设置 token
 * @param {string|null} token - JWT 或 null（登出）
 * @param {boolean} [rememberMe] - 是否记住我，仅 token 存在时有效
 */
export function setAuthToken(token, rememberMe = false) {
    if (token) {
        if (rememberMe) {
            localStorage.setItem(REMEMBER_ME_KEY, '1')
            localStorage.setItem(AUTH_TOKEN_KEY, token)
            sessionStorage.removeItem(AUTH_TOKEN_KEY)
        } else {
            localStorage.removeItem(REMEMBER_ME_KEY)
            localStorage.removeItem(AUTH_TOKEN_KEY)
            sessionStorage.setItem(AUTH_TOKEN_KEY, token)
        }
    } else {
        localStorage.removeItem(REMEMBER_ME_KEY)
        localStorage.removeItem(AUTH_TOKEN_KEY)
        sessionStorage.removeItem(AUTH_TOKEN_KEY)
    }
}

/**
 * 统一请求封装
 * @param {string} path - 路径（如 '/api/xxx'）
 * @param {RequestInit & { body?: object; timeout?: number }} options - fetch 选项，body 为对象时会 JSON.stringify，timeout 为毫秒超时
 * @returns {Promise<{ result?: unknown; [k: string]: unknown }>}
 */
export async function request(path, options = {}) {
    const { body, headers: optHeaders, timeout, ...rest } = options
    const token = getAuthToken()
    const headers = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...optHeaders,
    }
    const url = path.startsWith('http') ? path : `${baseURL}${path}`
    const isGet = (rest.method || 'GET').toUpperCase() === 'GET'
    let signal = rest.signal
    let timeoutId
    if (timeout != null && timeout > 0 && !signal) {
        const controller = new AbortController()
        signal = controller.signal
        timeoutId = setTimeout(() => controller.abort(), timeout)
    }
    const fetchOptions = {
        ...rest,
        signal,
        headers,
        body: isGet ? undefined : (body != null ? JSON.stringify(body) : options.body),
    }
    try {
        const response = await fetch(url, fetchOptions)
        if (timeoutId) clearTimeout(timeoutId)
        const data = await response.json().catch(() => ({}))
        if (response.status === 401) {
            setAuthToken(null)
            const err = new Error(data?.detail || '需要登录')
            err.code = 'UNAUTHORIZED'
            throw err
        }
        if (!response.ok || data?.error) {
            throw new Error(data?.error || data?.detail || `HTTP ${response.status}`)
        }
        return data
    } catch (err) {
        if (timeoutId) clearTimeout(timeoutId)
        if (err?.name === 'AbortError') {
            throw new Error('请求超时，请稍后重试（研究/新闻分析可能需 1–2 分钟）')
        }
        throw err
    }
}
