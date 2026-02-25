import React, { useState, useRef, useEffect } from 'react'

const menuStyle = {
    position: 'relative',
    display: 'inline-block'
}

const triggerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    background: 'rgba(255,255,255,0.08)',
    border: '1px solid rgba(255,255,255,0.14)',
    borderRadius: 8,
    color: 'inherit',
    cursor: 'pointer',
    fontSize: 13
}

const dropdownStyle = {
    position: 'absolute',
    top: '100%',
    right: 0,
    marginTop: 6,
    minWidth: 160,
    padding: 8,
    background: 'rgba(11,16,32,0.98)',
    border: '1px solid rgba(255,255,255,0.14)',
    borderRadius: 10,
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
    zIndex: 1000
}

const itemStyle = {
    display: 'block',
    width: '100%',
    padding: '8px 12px',
    border: 'none',
    background: 'none',
    color: 'inherit',
    cursor: 'pointer',
    fontSize: 13,
    textAlign: 'left',
    borderRadius: 6,
    transition: 'background 0.15s'
}

const dangerItemStyle = {
    ...itemStyle,
    color: '#f87171'
}

const dividerStyle = {
    height: 1,
    margin: '6px 0',
    background: 'rgba(255,255,255,0.14)'
}

export default function AccountMenu({ user, onLogout, onDeleteAccount }) {
    const [open, setOpen] = useState(false)
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
    const [deletePassword, setDeletePassword] = useState('')
    const [deleteError, setDeleteError] = useState('')
    const [deleting, setDeleting] = useState(false)
    const ref = useRef(null)

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (ref.current && !ref.current.contains(e.target)) {
                setOpen(false)
                setShowDeleteConfirm(false)
            }
        }
        document.addEventListener('click', handleClickOutside)
        return () => document.removeEventListener('click', handleClickOutside)
    }, [])

    const handleDelete = async () => {
        if (!deletePassword.trim()) {
            setDeleteError('请输入密码')
            return
        }
        setDeleteError('')
        setDeleting(true)
        try {
            await onDeleteAccount(deletePassword)
            setShowDeleteConfirm(false)
        } catch (e) {
            setDeleteError(e.message || '注销失败')
        } finally {
            setDeleting(false)
        }
    }

    return (
        <div ref={ref} style={menuStyle}>
            <button
                type="button"
                style={triggerStyle}
                onClick={() => setOpen(!open)}
                title={user?.username}
            >
                <span style={{ maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {user?.username || '账户'}
                </span>
                <span style={{ opacity: 0.7 }}>▾</span>
            </button>

            {open && !showDeleteConfirm && (
                <div style={dropdownStyle}>
                    <div style={{ padding: '6px 12px', fontSize: 12, color: 'rgba(255,255,255,0.65)' }}>
                        已登录：{user?.username}
                    </div>
                    <div style={dividerStyle} />
                    <button
                        type="button"
                        style={itemStyle}
                        onMouseEnter={(e) => { e.target.style.background = 'rgba(255,255,255,0.08)' }}
                        onMouseLeave={(e) => { e.target.style.background = 'none' }}
                        onClick={() => { onLogout(); setOpen(false) }}
                    >
                        退出登录
                    </button>
                    <button
                        type="button"
                        style={dangerItemStyle}
                        onMouseEnter={(e) => { e.target.style.background = 'rgba(248,113,113,0.15)' }}
                        onMouseLeave={(e) => { e.target.style.background = 'none' }}
                        onClick={() => setShowDeleteConfirm(true)}
                    >
                        注销账户
                    </button>
                </div>
            )}

            {open && showDeleteConfirm && (
                <div style={{
                    ...dropdownStyle,
                    minWidth: 260,
                    padding: 16
                }}>
                    <div style={{ marginBottom: 12, fontSize: 14, fontWeight: 500 }}>
                        注销账户
                    </div>
                    <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.7)', marginBottom: 12 }}>
                        将永久删除账户及所有数据（选股池、模拟仓等），且无法恢复。
                    </p>
                    <input
                        type="password"
                        placeholder="请输入密码确认"
                        value={deletePassword}
                        onChange={(e) => { setDeletePassword(e.target.value); setDeleteError('') }}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            marginBottom: 8,
                            background: 'rgba(255,255,255,0.08)',
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: 6,
                            color: 'inherit',
                            boxSizing: 'border-box'
                        }}
                    />
                    {deleteError && (
                        <div style={{ fontSize: 12, color: '#f87171', marginBottom: 8 }}>{deleteError}</div>
                    )}
                    <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                        <button
                            type="button"
                            style={itemStyle}
                            onClick={() => { setShowDeleteConfirm(false); setDeletePassword(''); setDeleteError('') }}
                        >
                            取消
                        </button>
                        <button
                            type="button"
                            style={{ ...itemStyle, ...dangerItemStyle }}
                            onClick={handleDelete}
                            disabled={deleting}
                        >
                            {deleting ? '处理中...' : '确认注销'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
