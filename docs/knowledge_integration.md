# 知识库系统集成完成

已成功参照 valuecell/research_agent 建立知识库系统并集成到 Research Agent 中。

## 主要改动

### 1. 知识库模块 (`graph/agents/research_agent/knowledge.py`)
- ✅ 使用 FAISS + HuggingFace Embeddings (all-MiniLM-L6-v2)
- ✅ 提供语义搜索工具 `search_knowledge_base`
- ✅ 提供信息保存工具 `save_important_info`
- ✅ 支持从 PDF URL 下载并提取内容
- ✅ 自动将 CNINFO 公告存入知识库

### 2. Research Agent 增强 (`graph/agents/research_agent/`)
- ✅ 更新 prompts.py - 详细指令（参照 valuecell 架构）
- ✅ 集成知识库工具到 research_agent_tools
- ✅ 强调"工具调用后立即搜索知识库"的工作流程

### 3. CNINFO 工具增强 (`tools/cninfo_tools.py`)
- ✅ 自动将前 3 条公告结果存入知识库
- ✅ 支持异步 PDF 下载和内容提取
- ✅ 容错处理（事件循环兼容）

### 4. 前端 API (`api_server.py`)
- ✅ 已有 `/api/research` - 调用 Research Agent
- ✅ 已有 `/api/news` - 调用 News Agent

### 5. 前端 UI (`web/src/App.jsx`)
- ✅ 研究资讯面板：调用 Research Agent（分析 CNINFO 等）+ 调用 News Agent

## 测试结果

运行 `python test_knowledge.py` 显示：
- ✅ 知识库初始化成功
- ✅ 添加文档成功
- ✅ 语义搜索成功（正确返回相关结果）
- ✅ Research Agent 能够：
  - 调用 CNINFO 工具获取公告
  - 搜索知识库中的历史信息
  - 综合给出包含来源的答案

示例输出：
```
搜索结果：
--- Result 1 ---
Source: test | Title:
Content: 贵州茅台（600519）是中国最大的白酒生产企业，2024年营收超过1400亿元。

Agent 回答：
基于我获取的信息，以下是贵州茅台的基本情况：
【包含了股价、市值、行业地位、最新公告等综合信息】
```

## 核心优势

相比之前的实现，现在的 Research Agent：

1. **知识积累**：自动将 CNINFO 公告存入知识库，避免重复查询
2. **语义搜索**：支持模糊查询，能找到相关历史信息
3. **综合分析**：结合新数据 + 历史知识给出更全面的答案
4. **可追溯性**：所有答案都标注来源（CNINFO、知识库、行情数据）
5. **PDF 支持**：自动下载并提取 PDF 内容（前 10000 字符）

## 使用方式

### 启动后端
```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
# 或
npm run api
```

### 启动前端
```bash
npm run dev
# 访问 http://localhost:5173
```

### 前端操作
- **研究资讯面板**：
  - 输入研究问题 → 点击"调用 Research Agent" → 分析 CNINFO 等数据并使用知识库
  - 输入新闻查询 → 点击"调用 News Agent" → 获取市场/个股新闻

### 测试知识库
```bash
python test_knowledge.py
```

## 依赖项

已安装（Python 3.x，推荐 3.10）：
- langchain-community
- faiss-cpu
- sentence-transformers
- langchain-huggingface
- pypdf

## 后续建议

1. **持久化**：如需持久化知识库，可使用 `FAISS.save_local()` / `load_local()`
2. **Embedding 模型**：可升级到更大的模型（如 `paraphrase-multilingual-mpnet-base-v2`）提升中文语义理解
3. **Reranker**：可添加 reranker 模块提升检索准确性
4. **批量导入**：可批量导入历史研报、公告到知识库

## 文档

详细使用说明见 [knowledge.md](knowledge.md)
