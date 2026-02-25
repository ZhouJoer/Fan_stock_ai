# Research Agent 使用指南

## 知识库功能

参照 valuecell/research_agent 的设计，本系统现已集成知识库功能：

### 1. 架构设计

**向量数据库**：
- 使用 FAISS + HuggingFace Embeddings (all-MiniLM-L6-v2)
- 数据存储在内存中（session bound），重启后清空
- 支持语义搜索和相似度匹配

**知识库工具**：
- `search_knowledge_base(query)`: 搜索内部知识库
- `save_important_info(content, title, source)`: 保存重要信息到知识库

### 2. 自动知识积累

**CNINFO 公告自动存储**：
- 调用 `fetch_cninfo_filings` 后，前 3 条结果自动存入知识库
- 包含公告标题、公司信息、日期、PDF 链接
- 如果 PDF 可下载，会提取前 10000 字符的内容
- 支持后续语义检索

### 3. Research Agent 工作流程

参照 valuecell 的最佳实践，Research Agent 现在会：

1. **接收用户问题** → 分析查询类型（事实性/分析性/探索性）
2. **调用相关工具** → 获取行情数据或 CNINFO 公告
3. **立即搜索知识库** → 查找相同公司/时间段的历史信息
4. **综合分析** → 结合新数据 + 知识库内容给出答案
5. **保存重要发现** → 将关键信息存入知识库供未来使用

### 4. 使用示例

**问题 1：茅台 2024 年报的营收是多少？**
- Agent 会调用 `fetch_cninfo_filings('600519', 'annual', year=2024)`
- 自动将年报信息存入知识库
- 搜索知识库中关于"茅台 2024 营收"的历史信息
- 综合给出答案并引用来源

**问题 2：比亚迪近期发展如何？**
- Agent 优先搜索知识库中已有的比亚迪分析
- 如需要最新数据，再调用行情工具或 CNINFO
- 将最新发现存入知识库

### 5. 前端调用方式

**调用 Research Agent**（右侧面板）：
```
输入：茅台 2024 年年报主要内容
点击：调用 Research Agent
```

**调用 News Agent**（右侧面板）：
```
输入：最近茅台的新闻
点击：调用 News Agent
```

### 6. 技术指标

所有行情工具现已包含完整技术指标：
- EMA12/EMA26
- MACD/MACD_SIGNAL/MACD_HIST
- 布林带 (BOLL_HIGH/BOLL_LOW)
- RSI14
- ATR14
- KDJ_K/KDJ_D/KDJ_J

### 7. 持久化说明

**当前实现**：
- 知识库数据存储在内存中（FAISS in-memory）
- 前端结果持久化在浏览器 localStorage

**扩展建议**：
如需持久化知识库，可以：
```python
# 保存到磁盘
vector_store = get_vector_store()
vector_store.save_local("knowledge_db")

# 下次启动时加载
from langchain_community.vectorstores import FAISS
vector_store = FAISS.load_local("knowledge_db", embeddings)
```

### 8. 运行环境

**后端**：
```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
# 或使用 npm run api（推荐 Python 3.10）
```

**前端**：
```bash
npm run dev
# 访问 http://localhost:5173
```

**测试知识库**：
```bash
python test_knowledge.py
```

### 9. 核心改进对比

参照 valuecell 架构的主要增强：

| 功能 | 之前 | 现在 |
|------|------|------|
| 知识库 | ❌ 无 | ✅ FAISS + Embeddings |
| 公告存储 | ❌ 仅返回 JSON | ✅ 自动存入知识库 |
| PDF 解析 | ❌ 仅返回链接 | ✅ 下载并提取内容 |
| 语义搜索 | ❌ 无 | ✅ 支持 |
| Agent 工作流 | 简单工具调用 | 工具 + 知识库综合分析 |
| Prompt 质量 | 基础 | 详细（参照 valuecell） |

### 10. 依赖项

已安装的新依赖：
- `langchain-community`: 向量数据库支持
- `faiss-cpu`: 向量索引
- `sentence-transformers`: Embedding 模型
- `langchain-huggingface`: HuggingFace 集成
- `pypdf`: PDF 内容提取

依赖在 Python 3.x 环境中安装即可（推荐 3.10）。
