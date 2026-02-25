import React, { useState, useEffect } from 'react'
import * as etfSimApi from '../api/etfSim.js'

export function TradesHistory({ accountId, totalTrades }) {
    const [trades, setTrades] = useState([])
    const [loading, setLoading] = useState(false)
    const [page, setPage] = useState(1)
    const [pageSize] = useState(20)
    const [pagination, setPagination] = useState({ total: 0, total_pages: 1, page: 1 })

    useEffect(() => {
        if (accountId) loadTrades(page)
    }, [accountId, page])

    async function loadTrades(pageNum) {
        if (!accountId) return
        try {
            setLoading(true)
            const data = await etfSimApi.getAccountTrades(accountId, pageNum, pageSize)
            if (data.result) {
                setTrades(data.result.trades || [])
                setPagination(data.result.pagination || {})
            }
        } catch (e) {
            console.error('加载交易记录失败:', e)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="trades-table-wrap">
            <div className="trades-header">
                <div className="trades-title">📝 交易记录（共{totalTrades}笔）：</div>
                {pagination.total_pages > 1 && (
                    <div className="trades-pagination">
                        <button
                            type="button"
                            className="buttonSmall"
                            onClick={() => setPage(p => Math.max(1, p - 1))}
                            disabled={page === 1 || loading}
                        >
                            上一页
                        </button>
                        <span className="trades-page-info">
                            第 {pagination.page} / {pagination.total_pages} 页
                        </span>
                        <button
                            type="button"
                            className="buttonSmall"
                            onClick={() => setPage(p => Math.min(pagination.total_pages, p + 1))}
                            disabled={page === pagination.total_pages || loading}
                        >
                            下一页
                        </button>
                    </div>
                )}
            </div>
            {loading ? (
                <div className="empty trades-loading">
                    <span className="loadingSpinner" aria-hidden="true" />
                    加载中...
                </div>
            ) : trades.length === 0 ? (
                <div className="empty">暂无交易记录</div>
            ) : (
                <div className="trades-scroll">
                    <table className="trades-table">
                        <thead>
                            <tr className="trades-thead-row">
                                <th className="trades-th trades-th-left">时间</th>
                                <th className="trades-th trades-th-center">类型</th>
                                <th className="trades-th trades-th-left">ETF代码</th>
                                <th className="trades-th trades-th-right">股数</th>
                                <th className="trades-th trades-th-right">价格</th>
                                <th className="trades-th trades-th-right">金额</th>
                                <th className="trades-th trades-th-left">原因</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades.map((trade, idx) => (
                                <tr key={idx} className="trades-tbody-row">
                                    <td className="trades-td trades-td-muted">
                                        {new Date(trade.date).toLocaleString('zh-CN')}
                                    </td>
                                    <td className="trades-td trades-td-center">
                                        <span className={`badge ${trade.type === 'buy' ? 'badge--buy' : 'badge--sell'}`}>
                                            {trade.type === 'buy' ? '买入' : '卖出'}
                                        </span>
                                    </td>
                                    <td className="trades-td trades-td-code">{trade.etf_code}</td>
                                    <td className="trades-td trades-td-right">{trade.shares}股</td>
                                    <td className="trades-td trades-td-right trades-td-num">
                                        ¥{trade.price.toFixed(2)}
                                    </td>
                                    <td className="trades-td trades-td-right trades-td-num">
                                        {trade.type === 'buy' ? (
                                            <span className="amount--negative">
                                                -¥{trade.cost?.toFixed(2) || (trade.shares * trade.price * 1.001 * 1.0003).toFixed(2)}
                                            </span>
                                        ) : (
                                            <span className="amount--positive">
                                                +¥{trade.revenue?.toFixed(2) || (trade.shares * trade.price * 0.999 * 0.9997).toFixed(2)}
                                            </span>
                                        )}
                                    </td>
                                    <td className="trades-td trades-td-muted trades-td-reason">
                                        {trade.reason || '-'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
