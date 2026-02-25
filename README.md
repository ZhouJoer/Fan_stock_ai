# 股票多 Agent 分析系统

基于 LangGraph + LangChain + DeepSeek 的智能股票分析系统，支持 A 股与港股市场的研究分析。

![量化示例](docs/images/factor_backtest_result.png)  

---

## 总体介绍

本系统是一个面向 **A 股与港股** 的一体化智能分析平台，将 **多 Agent 协作**、**研究资讯**、**量化回测**、**选股池**、**ETF 轮动** 和 **因子挖掘** 整合在同一套 Web 界面与 API 中。通过 LangGraph 编排工作流、DeepSeek 大模型驱动决策与审计，并结合 AkShare / CNINFO 等数据源与本地知识库，为个人投资者与研究者在「查公告、看行情、做回测、管组合」等场景提供可追溯、可复现的分析结果。

前端采用 Vite + React，后端为 FastAPI，支持 Docker 一键部署或本地虚拟环境开发；所有分析结果可在浏览器中查看并持久化保存。

---

## 动机与设计取舍

本项目的主要动机是**学习和实践 AI Agent / LangGraph 工作流在实际量化场景中的使用方式**，同时保留一套「不用 Agent 时也能正常工作」的传统流水线。核心做法：在同一个工程里同时保留**传统可控流水线**与**多 Agent 实验链路**——既可当作「偏保守的量化+工具链」使用，也可把多 Agent 模块当作在真实业务约束下试验 LangGraph/LangChain 的实验场。

- **多 Agent 协作**：根据用户意图切换工作流，Data_Researcher、Quant_Trader、Quant_Auditor 等节点协同完成数据收集、策略回测与风险审计。
- **研究可追溯**：Research Agent 调用 CNINFO 公告、知识库与行情，回答中列出参考文档与 PDF 链接。
- **量化与回测**：多策略回测、AI 实时决策、7 合 1 回测图表与买卖点标注，可选 LLM 信号增强。
- **选股池与因子挖掘**：选股池动态回测与模拟仓调仓；因子挖掘提供三 Agent 深度挖掘、多因子组合与回测、稳健性检查。
- **ETF 轮动**：技术指标打分、调仓建议、模拟盘与（可选）QMT 实盘对接。
- **知识库与文档**：FAISS 向量库 + 公告/PDF 入库，研究结论可保存并参与语义检索。

---

## 功能概览

| 模块 | 说明 | 文档 |
|------|------|------|
| **研究资讯** | Research Agent（CNINFO 公告、知识库、行情）；News Agent 暂不可用 | [研究资讯](docs/module/研究资讯.md) |
| **量化回测** | AI Agent 量化、普通回测、AI 实时决策、LLM 信号回测 | [量化回测与AI决策](docs/module/量化回测与AI决策.md) |
| **选股池** | 股票池管理、信号与仓位、组合回测、模拟仓 | [选股池与模拟仓](docs/module/选股池与模拟仓.md) |
| **因子挖掘** | 深度挖掘与回测、Agent 工作流、稳健性检查、保存摘要 | [因子挖掘与回测](docs/module/因子挖掘与回测.md) |
| **ETF 轮动** | 回测分析、调仓建议、模拟盘、实盘（QMT） | [ETF轮动前端与模拟盘](docs/module/ETF轮动前端与模拟盘.md)、[ETF轮动策略交易逻辑](docs/module/detail/etf_rotation_trading_logic.md) |
| **知识库** | FAISS 向量库、公告/PDF 入库、语义检索 | [knowledge](docs/module/detail/knowledge.md) |

---

## 快速开始

**环境要求**：Python 3.x（推荐 3.10）、Node.js 16+、DeepSeek API Key。

- **Docker（推荐）**：运行 `deploy.bat`（Windows）或 `./deploy.sh`（Linux/Mac），或手动 `cp .env.example .env` 配置密钥后 `docker-compose up -d`。
- **虚拟环境（开发）**：运行 `setup_venv.ps1` / `setup_venv.bat` / `setup_venv.sh`，激活 venv 后 `npm run install:web`，再 `npm run api` 与 `npm run dev`。
- **手动安装**：`pip install -r requirements.txt`、`npm run install:web`，配置 `.env` 后分别启动 uvicorn 与前端。

**详见 [部署指南](docs/部署指南.md)。**

访问：前端 http://localhost:5173，后端 API http://localhost:8000。

---

## 使用示例

各功能（量化回测、因子挖掘、研究与新闻、ETF 轮动模拟盘等）的典型操作步骤与输入输出示例见 **[使用示例](docs/使用示例.md)**。

---

## 系统架构

业务模块在 `modules/`，纯工具在 `tools/`，API 按领域在 `api/routers` 下拆分。

```
stock/
├── modules/          # etf_rotation、stock_pool、factor_mining、strategy_config、backtest、chanlun、qmt
├── api/routers/      # research_news、quant、pool、etf
├── graph/            # agents（research_agent、news_agent）、nodes、workflow
├── llm/              # DeepSeek / OpenAI 配置
├── tools/            # akshare、cninfo、stock_data、backtest、etf_rotation、deep_factor_search 等
├── web/              # 前端（Vite + React）
├── docs/             # 文档
├── api_server.py     # FastAPI 入口
└── requirements.txt
```

---

## 技术栈

**后端**：LangGraph、LangChain、DeepSeek、AkShare、CNINFO、FAISS、FastAPI。  
**前端**：Vite 5、React 18、localStorage。

---

## 文档索引

- [部署指南](docs/部署指南.md) — 环境要求与 Docker / 虚拟环境 / 手动部署步骤
- [常见问题](docs/常见问题.md) — akshare 网络错误、Python、API 密钥、端口等
- [使用示例](docs/使用示例.md) — 各功能操作步骤与示例
- [因子挖掘与回测](docs/module/因子挖掘与回测.md)
- [量化回测与AI决策](docs/module/量化回测与AI决策.md)
- [选股池与模拟仓](docs/module/选股池与模拟仓.md)
- [ETF轮动前端与模拟盘](docs/module/ETF轮动前端与模拟盘.md)、[ETF轮动策略交易逻辑](docs/module/detail/etf_rotation_trading_logic.md)
- [研究资讯](docs/module/研究资讯.md)
- [knowledge](docs/module/detail/knowledge.md)

---

## 测试

```bash
python test_knowledge.py
python test_document_citation.py
# 推荐使用 Python 3.10
```

更多见 [部署指南](docs/部署指南.md)。

---

## 功能特色

- **知识库增强**：自动积累研究材料、语义搜索、PDF 提取。
- **文档可追溯**：答案标注来源、原始文档链接、数据获取时间。
- **技术指标完整**：K 线、均线、MACD、布林带、RSI/ATR/KDJ。
- **双窗口设计**：研究与新闻、独立持久化、功能分区清晰。

---

## 注意事项

- 使用中遇到的环境、网络、密钥、端口等问题，请参见 **[常见问题](docs/常见问题.md)**。
- **回测说明**：除 ETF 轮动外，量化回测、因子挖掘回测、选股池组合回测**均未考虑滑点与手续费**，回测结果仅供参考。详见 [常见问题 - 回测是否考虑滑点与手续费](docs/常见问题.md)。

---

## 贡献

欢迎提 Issue 和 PR。

## 许可

MIT License

---

**最后更新**：2026-02-25
