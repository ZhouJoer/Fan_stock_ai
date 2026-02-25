import React, { useState, useEffect } from 'react'
import QuantPage from './pages/QuantPage.jsx'
import ResearchPage from './pages/ResearchPage.jsx'
import StockPoolPage from './pages/StockPoolPage.jsx'
import FactorMiningPage from './pages/FactorMiningPage.jsx'
import ETFRotationPage from './pages/ETFRotationPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import AccountMenu from './components/AccountMenu.jsx'
import { checkBackendHealth, deleteAccount, getAuthMode, getCurrentUser } from './api/auth'

export default function App() {
    const [activeTab, setActiveTab] = useState('quant')
    const [authReady, setAuthReady] = useState(false)
    const [showLogin, setShowLogin] = useState(false)
    const [user, setUser] = useState(null)
    const [connError, setConnError] = useState(false)

    const runAuthFlow = async () => {
        setConnError(false)
        const ok = await checkBackendHealth()
        if (!ok) {
            setConnError(true)
            setAuthReady(true)
            setShowLogin(false)
            return
        }
        try {
            const multiUser = await getAuthMode()
            if (!multiUser) {
                setUser({ user_id: 'default', username: 'default' })
                setAuthReady(true)
                return
            }
            const u = await getCurrentUser()
            setAuthReady(true)
            setShowLogin(!u)
            setUser(u || { user_id: 'default', username: 'default' })
        } catch (e) {
            setAuthReady(true)
            setShowLogin(false)
        }
    }

    useEffect(() => {
        runAuthFlow()
    }, [])

    const handleLogin = (userData) => {
        setUser(userData)
        setShowLogin(false)
    }

    const handleLogout = () => {
        import('./api/client').then(({ setAuthToken }) => {
            setAuthToken(null)
            setUser(null)
            setShowLogin(true)
        })
    }

    const handleDeleteAccount = async (password) => {
        await deleteAccount(password)
        setUser(null)
        setShowLogin(true)
    }

    if (!authReady) {
        return <div style={{ padding: 40, textAlign: 'center' }}>加载中...</div>
    }
    if (connError) {
        return (
            <main className="app">
                <div style={{ maxWidth: 400, margin: '80px auto', padding: 24, textAlign: 'center', border: '1px solid #e0e0e0', borderRadius: 8, background: '#fff' }}>
                    <h2 style={{ color: '#c00', marginBottom: 16 }}>无法连接后端</h2>
                    <p style={{ marginBottom: 20, color: '#666' }}>请确保已启动 API 服务：在项目目录运行 <code>npm run api</code></p>
                    <button onClick={() => { setAuthReady(false); setConnError(false); setTimeout(runAuthFlow, 100) }} style={{ padding: '10px 24px', cursor: 'pointer', fontSize: 15 }}>重试</button>
                    <p style={{ marginTop: 20, fontSize: 13, color: '#999' }}>
                        或 <button type="button" onClick={() => { setConnError(false); setAuthReady(true) }} style={{ background: 'none', border: 'none', color: '#2563eb', cursor: 'pointer', textDecoration: 'underline' }}>继续使用</button>（部分功能将不可用）
                    </p>
                </div>
            </main>
        )
    }
    if (showLogin) {
        return <LoginPage onLogin={handleLogin} />
    }

    const tabs = [
        { id: 'quant', label: '📊 量化回测', icon: '📊' },
        { id: 'factor', label: '🧪 因子挖掘', icon: '🧪' },
        { id: 'pool', label: '📋 选股池', icon: '📋' },
        { id: 'etf', label: '🔄 ETF轮动', icon: '🔄' },
        { id: 'research', label: '🔍 研究资讯', icon: '🔍' },
    ]

    return (
        <main className="app">
            <header className="topbar">
                <h1 className="title">股票助手（前端）</h1>
                <div className="subtitle">多功能面板：量化回测 / 选股池管理 / ETF轮动 / 研究资讯（本地持久化）</div>
                <div style={{ position: 'absolute', right: 16, top: 16, fontSize: 13 }}>
                    {user ? (
                        user.user_id === 'default' ? (
                            <span style={{ opacity: 0.8 }}>单用户模式</span>
                        ) : (
                            <AccountMenu
                                user={user}
                                onLogout={handleLogout}
                                onDeleteAccount={handleDeleteAccount}
                            />
                        )
                    ) : (
                        <button
                            onClick={() => setShowLogin(true)}
                            style={{
                                padding: '6px 12px',
                                cursor: 'pointer',
                                background: 'rgba(86,185,255,0.2)',
                                border: '1px solid rgba(86,185,255,0.4)',
                                borderRadius: 8,
                                color: 'var(--primary)'
                            }}
                        >
                            登录 / 注册
                        </button>
                    )}
                </div>
            </header>

            <nav className="tabNav">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        className={activeTab === tab.id ? 'tabBtnActive' : 'tabBtn'}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </nav>

            <div className="tabContent">
                {activeTab === 'quant' && <QuantPage user={user} />}
                {activeTab === 'factor' && <FactorMiningPage user={user} onSwitchToPool={() => setActiveTab('pool')} />}
                {activeTab === 'pool' && <StockPoolPage user={user} />}
                {activeTab === 'etf' && <ETFRotationPage user={user} />}
                {activeTab === 'research' && <ResearchPage user={user} />}
            </div>
        </main>
    )
}
