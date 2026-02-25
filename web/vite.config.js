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
                    }
                }
            }
        }
    }
})
