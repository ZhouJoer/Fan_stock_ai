"""为前端提供的最小 API 服务。

用途：
- 前端通过 HTTP 调用 research / news、量化回测、选股池、ETF 轮动等。

运行：
- 安装依赖：pip install -r requirements.txt
- 启动：python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

# 在任何可能导入 matplotlib 的代码之前设置非交互式后端，避免研究/回测等模块
# 在异步请求中触发 tkinter，导致 "main thread is not in main loop" 的 RuntimeError
import matplotlib
matplotlib.use("Agg")
# 配置中文字体，避免组合回测、ETF轮动等图表中文乱码
from utils.matplotlib_chinese import setup_chinese_font
setup_chinese_font()

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router
from api.routers import (
    research_news_router,
    quant_router,
    pool_router,
    etf_router,
)

app = FastAPI(title="my_stock API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(research_news_router)
app.include_router(quant_router)
app.include_router(pool_router)
app.include_router(etf_router)
