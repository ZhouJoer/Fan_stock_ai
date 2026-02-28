import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '')
    const host = env.VITE_SERVER_HOST || 'localhost'
    const port = parseInt(env.VITE_SERVER_PORT || '5173', 10)
    const apiTarget = env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

    return {
        plugins: [react()],
        server: {
            host,
            port,
            strictPort: true,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    timeout: 0,
                    configure: (proxy) => {
                        proxy.on('proxyReq', (proxyReq, req) => {
                            req.on('aborted', () => proxyReq.destroy())
                        })
                        // SSE/流式响应：服务端关闭时同步关闭客户端，避免连接挂起
                        proxy.on('proxyRes', (proxyRes, req, res) => {
                            proxyRes.on('close', () => {
                                if (proxyRes.errored && !res.closed) {
                                    res.destroy(proxyRes.errored)
                                }
                            })
                        })
                    }
                }
            }
        }
    }
})
