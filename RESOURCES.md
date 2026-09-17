# 学习资源索引（按周）

> 原则：**官方文档 > 论文 > 课程 > 博客**。每周只允许打开本周对应的资源，避免"收藏夹学习"。

---

## 通用工具与入口

| 资源 | 链接 | 用途 |
|---|---|---|
| OpenAI Platform Docs | https://platform.openai.com/docs | Function Calling / Structured Outputs / Streaming 权威定义 |
| Anthropic Engineering | https://www.anthropic.com/engineering | Building effective agents / Context Engineering（**强烈推荐**） |
| MCP 官方 | https://modelcontextprotocol.io | 协议规范 + SDK + Inspector |
| LangGraph 文档 | https://langchain-ai.github.io/langgraph/ | Persistence / HITL / Multi-agent |
| Langfuse | https://langfuse.com/docs | 可观测性与评测 |
| Ollama | https://ollama.com | 本地模型 |
| AI Agent 面试指南（开源） | https://github.com/bcefghj/ai-agent-interview-guide | 200+ 八股题 + Java/Python/Go 项目对照 |
| Agent 岗位 JD 汇总 | https://github.com/harrisliangsu/ai-agent-engineer-handbook | 对标招聘要求 |

---

## W1：大模型底层原理

**必做**
- Karpathy《Neural Networks: Zero to Hero》全系列（重点：micrograd → makemore → GPT）
- Karpathy《Let's build the GPT Tokenizer》：https://www.youtube.com/watch?v=zduSFxRajkE
- nanoGPT 仓库：https://github.com/karpathy/nanoGPT

**论文（各读出 3 句话即可）**
1. Attention Is All You Need — https://arxiv.org/abs/1706.03762
2. LoRA: Low-Rank Adaptation — https://arxiv.org/abs/2106.09685
3. Lost in the Middle — https://arxiv.org/abs/2307.03172

**概念自查**：BPE / Self-Attention / 多头 / RoPE / KV Cache / Prefill-Decode / 量化 / 采样参数 / 幻觉成因

---

## W2：Prompt 工程与 RAG

**必读**
- Anthropic《Context Engineering》与《Building Effective Agents》
- OpenAI Structured Outputs 文档 + JSON Schema 说明
- 《Chain-of-Thought Prompting》— https://arxiv.org/abs/2201.11903
- RAG 原论文 — https://arxiv.org/abs/2005.11401
- 《Lost in the Middle》再读一遍（分块与上下文顺序的依据）

**框架文档**
- LlamaIndex：Ingestion / Node Parser / Retriever / Reranker 章节
- LangChain：TextSplitter / VectorStore / EnsembleRetriever(BM25+向量)
- 向量库：Milvus 或 Qdrant 官方 Quickstart + HNSW 参数说明
- Rerank：BGE-reranker（FlagEmbedding）说明

**工具**
- DeepLearning.AI 短课：《LangChain for LLM Application Development》《Building and Evaluating Advanced RAG》

**产出**：分块策略对比表、检索评测报告（Recall@K / MRR）、注入攻击样例集

---

## W3：Agent 内核

**论文**
1. ReAct — https://arxiv.org/abs/2210.03629（**精读，能默画循环**）
2. Reflexion — https://arxiv.org/abs/2303.11366
3. Toolformer — https://arxiv.org/abs/2302.04761
4. Plan-and-Solve — https://arxiv.org/abs/2305.04091
5. Tree of Thoughts — https://arxiv.org/abs/2305.10601

**文档/源码**
- LangGraph：State / Checkpointer / `interrupt` / Multi-agent 章节
- OpenAI Agents SDK 源码（Agent Loop 部分）：https://github.com/openai/openai-agents-python
- Spring AI 官方文档：https://docs.spring.io/spring-ai/reference/
- LangChain4j：https://docs.langchain4j.dev

**产出**：手写 `my_react.py`、手写 `memory.py`、手写 vs LangGraph 对比表

---

## W4：MCP + 多智能体 + 可观测性

- MCP 规范全文（Server / Client / Transports / Authorization）
- MCP Python SDK：https://github.com/modelcontextprotocol/python-sdk
- MCP Inspector（调试工具）
- MCP 安全最佳实践（工具投毒、提示注入、越权、混淆代理攻击）
- LangGraph Multi-agent 教程（Supervisor / Swarm）
- AutoGen 或 CrewAI（对照了解即可，不必深挖）
- OpenTelemetry GenAI 语义约定：https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Langfuse：Tracing / Prompt Management / Datasets & Evaluation

**产出**：自研 MCP Server（3 Tool + 2 Resource + 1 Prompt）、Supervisor 多 Agent Demo、Trace 截图

---

## W5–W7：旗舰项目

**项目相关**
- 数据：TPC-H / Kaggle 电商数据集 / 自造业务表（5–10 张 + 指标口径文档）
- SQL 安全：`sqlglot` 做 AST 解析与只读校验
- 前端：React + Vite + SSE（`EventSource` / fetch stream）+ ECharts/AntV 图表
- 压测：k6 或 Locust
- 部署：Docker Compose（app + PG + Redis + Qdrant + Langfuse）

**评测相关**
- RAGAS 指标说明（忠实度 / 答案相关性 / 上下文精确率）
- τ-bench（衡量工具调用与多轮一致性的思路）
- LLM-as-Judge 的偏差与校准（位置偏差、长度偏差、自我偏好）

**产出**：可访问 Demo、演示视频、评测报告、架构图、ADR 文档、压测数据

---

## W8：面试与转化

- https://github.com/bcefghj/ai-agent-interview-guide —— 200+ 八股题，按模块刷
- https://github.com/harrisliangsu/ai-agent-engineer-handbook —— JD 对标 + 面试准备
- 目标岗位 JD 收集：至少 20 份真实 JD，提取高频关键词做词频统计，反向补齐

**投递技巧**
- 定向搜索关键词：`Spring AI`、`LangChain4j`、`大模型应用开发`、`Agent 开发`、`RAG`、`MCP`、`智能体平台`
- 每个岗位定制简历前两行（把 JD 关键词自然嵌入项目描述）

---

## 每天的时间分配建议（防资源过载）

| 类型 | 占比 | 说明 |
|---|---|---|
| 动手编码 | 50% | 唯一真正决定成败的部分 |
| 官方文档/论文 | 30% | 建立第一性原理 |
| 视频课程 | 15% | 加速理解，不做主力 |
| 社区/资讯 | 5% | 保持方向感，不刷焦虑 |
