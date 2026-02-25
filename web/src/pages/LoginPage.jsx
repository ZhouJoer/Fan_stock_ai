import React, { useState } from 'react'
import { request, setAuthToken } from '../api/client'

export default function LoginPage({ onLogin }) {
    const [mode, setMode] = useState('login') // 'login' | 'register'
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [rememberMe, setRememberMe] = useState(true)
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setLoading(true)
        try {
            const endpoint = mode === 'login' ? '/api/auth/login' : '/api/auth/register'
            const res = await request(endpoint, {
                method: 'POST',
                body: { username: username.trim(), password: password.trim() }
            })
            if (res.token) {
                setAuthToken(res.token, rememberMe)
            }
            onLogin(res.user)
        } catch (err) {
            setError(err.message || '请求失败')
        } finally {
            setLoading(false)
        }
    }

    return (
        <main className="app loginPage">
            <div className="loginCard">
                <div className="loginLogo">
                    <h1>量化选股</h1>
                    <p>登录以使用选股池、模拟盘与回测功能</p>
                </div>
                <div className="loginTabs">
                    <button
                        type="button"
                        className={`loginTab ${mode === 'login' ? 'loginTabActive' : ''}`}
                        onClick={() => { setMode('login'); setError('') }}
                    >
                        登录
                    </button>
                    <button
                        type="button"
                        className={`loginTab ${mode === 'register' ? 'loginTabActive' : ''}`}
                        onClick={() => { setMode('register'); setError('') }}
                    >
                        注册
                    </button>
                </div>
                <form onSubmit={handleSubmit} className="loginForm">
                    <div className="loginField">
                        <label>用户名</label>
                        <input
                            type="text"
                            className="input"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="至少 2 个字符"
                            required
                            autoComplete="username"
                        />
                    </div>
                    <div className="loginField">
                        <label>密码</label>
                        <input
                            type="password"
                            className="input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder={mode === 'register' ? '至少 6 位' : '请输入密码'}
                            required
                            minLength={mode === 'register' ? 6 : undefined}
                            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                        />
                    </div>
                    <label className="loginCheckbox">
                        <input
                            type="checkbox"
                            checked={rememberMe}
                            onChange={(e) => setRememberMe(e.target.checked)}
                        />
                        记住我（关闭浏览器后仍保持登录）
                    </label>
                    {error && (
                        <div className="loginError">{error}</div>
                    )}
                    <button
                        type="submit"
                        className="loginSubmit"
                        disabled={loading}
                    >
                        {loading ? '处理中...' : (mode === 'login' ? '登录' : '注册')}
                    </button>
                </form>
                <div className="loginSwitch">
                    {mode === 'login' ? '没有账号？' : '已有账号？'}
                    <a
                        href="#"
                        onClick={(e) => {
                            e.preventDefault()
                            setMode(mode === 'login' ? 'register' : 'login')
                            setError('')
                        }}
                    >
                        {mode === 'login' ? '注册' : '登录'}
                    </a>
                </div>
            </div>
        </main>
    )
}
