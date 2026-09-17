# 6 年 Java + 前端全栈 → Agent 开发工程师 · 2 个月转型计划

> 版本：v1.0 ｜ 适用对象：6 年全栈（Java 后端为主 + 前端）｜ 周期：8 周 ｜ 每日投入：工作日 3h + 周末 6h（≈ 30h/周，合计 ≈ 240h）
> 目标：2 个月后具备投递「大模型应用/Agent 开发工程师」并通过一二面的能力，产出 1 个可讲 30 分钟深度细节的实战项目，并对 Agent 底层运行原理（Token→注意力→解码→工具调用→循环控制→记忆→上下文工程）有第一性原理级认知。

---

## 目录

- [0. 先读这一节：你的优势、劣势与真实预期](#0-先读这一节你的优势劣势与真实预期)
- [1. 对标岗位：市面 Agent 开发岗到底要什么](#1-对标岗位市面-agent-开发岗到底要什么)
- [2. 能力差距矩阵与补齐策略](#2-能力差距矩阵与补齐策略)
- [3. 8 周总览路线图](#3-8-周总览路线图)
- [4. 每周详细计划（Day-by-Day）](#4-每周详细计划day-by-day)
- [5. 旗舰项目设计（贯穿 W5–W7，面试主战场）](#5-旗舰项目设计贯穿-w5w7面试主战场)
- [6. 底层原理深挖清单（必须能白板讲清）](#6-底层原理深挖清单必须能白板讲清)
- [7. 每日/每周固定动作模板](#7-每日每周固定动作模板)
- [8. 资源清单（精选，不贪多）](#8-资源清单精选不贪多)
- [9. 第 8 周：简历、项目包装与面试转化](#9-第-8-周简历项目包装与面试转化)
- [10. 高频面试题预演（含答题骨架）](#10-高频面试题预演含答题骨架)
- [11. 验收标准：什么叫做"可以投递了"](#11-验收标准什么叫做可以投递了)
- [12. 风险与备选方案（Plan B）](#12-风险与备选方案plan-b)
- [13. 附录：技术选型决策记录](#13-附录技术选型决策记录)

---

## 0. 先读这一节：你的优势、劣势与真实预期

### 0.1 你被严重低估的存量资产

转型最容易犯的错，是把 6 年经验当包袱，去跟应届生拼"谁 Prompt 写得好"。实际上你的存量资产在 Agent 工程化阶段是硬通货：

| 你的存量能力 | 在 Agent 领域的对应价值 |
|---|---|
| Spring Boot / 微服务 / 事务 / 中间件 | Agent 平台的服务化、状态持久化、幂等、限流、多租户——**大量 Agent Demo 死在这一步** |
| 分布式与高可用（熔断、降级、重试） | 模型路由、多 Provider 容灾、Token 限流、超时与重试策略，直接可迁移（LLM 网关本质就是一个网关） |
| MySQL / Redis / ES / 消息队列 | 记忆系统（Redis 滑窗）、向量库选型、异步任务编排（长任务 Agent 必须靠 MQ + 状态机） |
| 前端 + BFF 能力 | Agent 产品的核心体验是流式输出、工具调用过程可视化、Human-in-the-loop 审批、Artifact 渲染——**懂前端 + 懂后端的 Agent 工程师非常稀缺** |
| 工作流引擎思维（Activiti/状态机/DAG） | LangGraph 本质是"带循环的状态机"，你比纯 Python 算法背景的人更容易理解 Checkpointer、中断恢复、并行分支 |
| 线上问题排查经验 | Agent 可观测性（Trace/Span、Token 归因、失败回放）是 2025 年岗位明确写进 JD 的能力项 |

**结论：你的定位不是"转行新手"，而是"带着工程化能力的 Agent 应用/平台工程师"。** 这个定位在金融、ERP、政企、客服中台类岗位上是明显的加分项，比拼 RAG Prompt 的新人更有优势。

### 0.2 必须诚实面对的三个短板

1. **Python 生态熟练度**：Python 岗位占绝对多数，语法不是问题（1 周可解决），但 `asyncio`、类型标注、`pydantic`、FastAPI 生命周期、虚拟环境/依赖管理必须写到能过 code review 的程度。
2. **没有生产级 Agent 落地案例**：多数 JD 写"1 年以上 Agent 研发经验"。2 个月无法伪造时间，但可以用**一个架构复杂度足够的项目 + 可复现的量化收益 + 源码级原理理解**去对冲。
3. **数学与模型训练层**：不需要会预训练，但必须能讲清 Transformer 注意力、KV Cache、位置编码、LoRA 微调原理、采样参数、为什么出现幻觉。这一层是二面分水岭。

### 0.3 真实预期管理

- **2 个月能拿到的**：Agent 应用开发岗（Python 或 Java 技术栈）、大模型应用开发岗、AI 平台/中台岗的面试机会与通过率；能独立设计并实现一个生产可用的 Agent 系统。
- **2 个月拿不到的**：算法/预训练/RLHF 岗、需要 3 年大模型经验的资深岗、大厂"Agent 算法工程师"。
- **成功率的关键变量**：① 旗舰项目的完成度与真实数据；② 是否真的手写了 ReAct/向量检索/MCP 而非只会调框架；③ 简历是否按 STAR 量化包装。三者任一缺失，投递转化率会腰斩。

---

## 1. 对标岗位：市面 Agent 开发岗到底要什么

综合公开 JD（参考：[资深大模型 Agent 开发工程师 JD](http://www.zftii.com./talent/846f3927138d4d2a95c2857a0baa3f1f.html)、[Grab Senior AI Agent Engineer](https://www.grab.careers/en/jobs/744000098287310/senior-ai-agent-engineer/)、[AI Agent 面试指南 JD 汇总](https://github.com/harrisliangsu/ai-agent-engineer-handbook/blob/main/interview-prep/jd-requirements.md)、[Agent 面试八股文项目](https://github.com/bcefghj/ai-agent-interview-guide)），可归纳出**五个能力簇**：

### 簇 A：大模型基础与 Prompt 工程（权重 15%）
- Transformer / Attention 原理、上下文窗口、Token 计费与压缩
- 采样参数（temperature / top_p / top_k / 惩罚项）的实际影响与调参经验
- Prompt 结构设计：角色、约束、Few-shot、CoT、输出格式约束
- Prompt 注入攻击与防御（生产环境必问）

### 簇 B：RAG 与知识工程（权重 25%）
- 文档解析（PDF/Word/表格/OCR）→ 分块策略（固定/递归/语义/父子块）→ 向量化 → 入库
- Embedding 模型选型与维度/成本权衡、向量库（Milvus/pgvector/Qdrant/ES kNN）
- 混合检索（向量 + BM25）+ RRF 融合 + Rerank 重排
- 召回评估：Recall@K、MRR、NDCG；RAGAS/自建评测集；多轮改写（Query Rewriting / HyDE）
- 进阶：GraphRAG、知识图谱、结构感知检索

### 簇 C：Agent 核心机制（权重 30%，核心中的核心）
- ReAct / Plan-and-Execute / Reflexion / Tree-of-Thought 等范式与其代价
- Function Calling / Tool Calling 的完整链路与失败模式
- **MCP（Model Context Protocol）**：Client/Server、Resources、Tools、Prompts、传输层（stdio/SSE/Streamable HTTP）、鉴权、与 Function Calling 的差异
- 记忆系统：短期滑窗、摘要压缩、长期向量记忆、记忆写入/遗忘策略
- 多智能体：Supervisor、Handoff、群聊、黑板模式；通信协议与冲突消解
- 上下文工程（Context Engineering）：这是 2025 年取代"Prompt 工程"的关键词——上下文预算分配、压缩、裁剪、子 Agent 隔离
- 循环控制：终止条件、最大步数、死循环检测、成本熔断

### 簇 D：工程化与平台（权重 20%，你的主场）
- Python 异步编程、FastAPI、Docker/K8s、Linux 常用排障
- Agent 服务化：会话状态持久化（Checkpointer）、断点续跑（Human-in-the-loop）、幂等、并发
- 可观测性：全链路 Trace（Langfuse / LangSmith / OpenTelemetry）、Token 与成本归因、失败回放
- 稳定性：模型路由与降级、熔断、重试退避、限流、缓存（语义缓存）
- 评测与回归：离线评测集 + LLM-as-Judge + 线上 A/B；上线护栏（Guardrails）
- 安全：工具权限分级、敏感操作人工确认、沙箱执行、审计日志

### 簇 E：模型层可选加分（权重 10%）
- 微调：LoRA/QLoRA、SFT 数据构造、评测；推理加速（vLLM、量化、KV Cache 复用）
- 部署：私有化部署（Ollama/vLLM）、模型路由网关

> **战略判断**：簇 A/C 是"必须从零补"，簇 D 是你的护城河，簇 B 是岗位需求量最大、最容易被问到细节的部分，簇 E 只到"能讲清 + 跑通一次 LoRA"即可。

---

## 2. 能力差距矩阵与补齐策略

| 能力项 | 你的现状 | 目标水平 | 补齐方式 | 验收 |
|---|---|---|---|---|
| Python 工程化 | 语法可读，异步弱 | 能写 FastAPI + asyncio 服务并 code review 通过 | W0 集中突击 + 全程用 Python 写脚本 | 能独立写出带超时/重试/并发的异步工具调用 |
| LLM 原理 | 概念模糊 | 能白板画 Attention + 讲 KV Cache 与采样 | W1 手写 mini-GPT + 精读 1 篇论文 | 手写 nanoGPT 级实现并跑通 |
| Prompt 工程 | 会用 ChatGPT | 能设计稳定结构化输出 + 防注入 | W2 结构化输出实战 | 输出 Schema 一次通过率 > 95% |
| RAG | 只会调用 | 能手写混合检索 + RRF + Rerank 并评测 | W2 先手写再上框架 | 自建 50 题评测集，Recall@5 提升可量化 |
| Agent 机制 | 无 | 能手写 ReAct 循环，讲清每种范式代价 | W3 手写 + 框架对照 | 手写版与 LangGraph 版可对比讲解 |
| MCP | 无 | 能开发 MCP Server 并接入多个 Client | W4 实战 | 自研 Server 在 Claude/Cursor 中可用 |
| 多智能体 | 无 | 能设计 Supervisor 架构并解决冲突 | W4 | 3 Agent 协作完成一个复合任务 |
| 可观测性 | 有传统 APM 经验 | 能搭 Trace 并做 Token 归因 | W5–W7 融入项目 | 一次请求全链路可视化 |
| 评测体系 | 无 | 能建评测集并做回归 | W5–W7 融入项目 | 有 before/after 数据对比 |
| 模型微调 | 无 | 能讲清 LoRA 并跑通一次 | W6 选修 | 一次 SFT 前后效果对比 |
| Java 侧 LLM 栈 | 无 | 能用 Spring AI / LangChain4j 搭建 | W3 选修（差异化亮点） | Java 版 Agent 可跑通并对比 |

---

## 3. 8 周总览路线图

```
W0 预热     Python 突击 + 环境搭建 + 账户/额度准备          （不产出项目，只清障）
W1 地基     大模型底层原理 + 手写 mini-GPT + API/流式/Function Calling
W2 检索     Prompt 工程 → 结构化输出 → RAG 从零手写到框架化 + 评测
W3 内核     Agent 从零手写（ReAct/工具/记忆） → LangGraph → Spring AI 对照
W4 协议     MCP 深度实战 + 多智能体 + 可观测性基础
W5 项目 1   旗舰项目：骨架 + 检索子系统 + Agent 内核（可跑通端到端）
W6 项目 2   旗舰项目：工具/审批/HITL + 多 Agent + 评测体系 + 前端可视化
W7 项目 3   旗舰项目：可观测性 + 稳定性 + Docker 部署 + 压测 + README/演示视频
W8 转化     简历 + 项目深挖稿 + 八股冲刺 + 模拟面试 + 开始投递
```

**里程碑（Gate）**——未通过不要进入下一阶段，否则后面全是夹生饭：

- **Gate 1（W1 末）**：能手写 BPE/注意力，并说清 `temperature=0` 仍不稳定的三个原因。
- **Gate 2（W2 末）**：RAG 评测集跑出数据，能解释为什么加 Rerank 后 Recall 提升而首字延迟增加多少。
- **Gate 3（W3 末）**：不看框架源码能从零写出可用的 ReAct Agent（200 行内），并讲清每行在做什么。
- **Gate 4（W4 末）**：自研 MCP Server 被至少 2 个客户端成功调用。
- **Gate 5（W7 末）**：项目有线上可访问 Demo 或一键启动 + 演示视频 + 量化指标 + 架构图。

---

## 4. 每周详细计划（Day-by-Day）

> 时间约定：工作日 3h（建议 20:00–23:00），周末各 6h。每周日 1h 写复盘（见 §7）。
> 每日结构固定为：**理论 40% / 编码 50% / 输出笔记 10%**。不写笔记 = 没学。

---

### W0（预热周，若时间紧可压缩到 3 天）：清障，不学新概念

**目标**：把"环境、额度、编辑器、Python 手感"四件事一次性解决，避免后面每天被环境问题打断。

| Day | 内容 | 产出 |
|---|---|---|
| D1 | Python 突击：类型标注、dataclass/pydantic、装饰器、生成器、`async/await`/`asyncio.gather`/`TaskGroup`、异常与上下文管理器、`uv`/`poetry` 依赖管理 | 30 个自测小题全过（见 `WEEK0_SETUP.md`） |
| D2 | 环境搭建：Python 3.11+、uv、VS Code + Ruff + Pylance、Docker Desktop、Node/pnpm；创建 `agent-study` 仓库骨架 | 环境自检脚本全绿 |
| D3 | 账号与额度：OpenAI/DeepSeek/通义/智谱任一主用 + 一个 Embedding API；本地 Ollama（Qwen2.5 / Llama3.1）备胎；Langfuse 云端或自托管 | 主用模型 + 备胎模型均可流式回复 |
| D4 | 精读官方文档：Chat Completions 的完整参数表、Function Calling 报文结构、Streaming 的 SSE 分片格式 | 手写笔记：一次 Function Calling 的完整 JSON 请求/响应 |
| D5 | 用 `httpx` 裸调 API（**禁止用任何 SDK**）：实现非流式 + 流式 + 多轮上下文 | `raw_chat.py` 可运行 |
| D6 | 裸调 Function Calling：手写 tools schema、解析 tool_calls、回填 tool 消息、二次请求 | `raw_tools.py`：模型能完成"查天气→计算"两步 |
| D7 | 复盘 + 补齐 + 预热阅读：精读 ReAct 论文（arXiv:2210.03629）、Toolformer 摘要 | 一页 ReAct 论文图解 |

**W0 关键认知**：Function Calling 不是"模型执行了函数"，而是**你在请求里注入工具 JSON Schema，模型被训练成输出一段符合该 Schema 的结构化文本，由你的代码去真正执行并把结果回填**。这个认知必须在本周建立，否则后面永远是黑盒。

---

### W1：大模型底层原理 + API 层彻底打通

**目标**：从 Token 到解码到工具调用的每一层都能讲透；手写 mini-GPT 建立"模型内部在干什么"的直觉。

| Day | 主题 | 具体任务 | 产出/验收 |
|---|---|---|---|
| D1 | 分词与向量化 | 手写 BPE 训练 + 编码；理解词表、`tiktoken`、中英文 Token 效率差异、为什么 Token 决定成本与上下文 | `my_bpe.py` 可训练小语料并正确编码 |
| D2 | 注意力机制 | 手写 Scaled Dot-Product Attention + 多头 + 因果掩码 + RoPE 位置编码；推演复杂度 O(n²) 的来源 | 纯 NumPy 实现，注释逐行 |
| D3 | 完整 Transformer | 拼装 Block：LayerNorm/RMSNorm、FFN、残差；实现自回归生成循环 + KV Cache 版本 | `mini_gpt.py` 能过拟合一个玩具数据集 |
| D4 | 训练与推理的工程面 | Cross-Entropy、AdamW、学习率 warmup；KV Cache 为什么能省算力却吃显存；量化（INT8/INT4）原理与代价；采样：greedy/beam/temperature/top-p/top-k/repetition penalty | 笔记 + 实验：同一 prompt 下不同采样参数输出对比 |
| D5 | 上下文窗口与幻觉 | 位置外推、上下文"中间遗忘"（Lost in the Middle）、长上下文成本；幻觉的三类成因（数据、解码、提示）与三种缓解手段 | 笔记 + 3 个小实验 |
| D6 | 微调与对齐（认知层） | SFT / LoRA / QLoRA 原理（低秩分解为什么有效）、RLHF/DPO 概览、什么场景才需要微调（**RAG vs 微调 vs Prompt 的决策树**） | 决策树图（面试高频） |
| D7 | 复盘 + 论文精读 | 《Attention Is All You Need》+《LoRA》扫读 | 一页 A4：Attention 公式推导 + LoRA 参数量计算 |

**W1 深挖问题（必须能答）**：
- 为什么 KV Cache 能加速解码？为什么它随上下文线性增长、且 batch 下会爆显存？（PagedAttention 要解决什么）
- `temperature=0` 为什么还不稳定？（浮点累加顺序、batch 内不同 prompt 的 kernel 差异、MoE 路由、推理框架的非确定性）
- Embedding 和生成模型的输出层有什么关系？（很多 Embedding 模型就是生成模型去掉 LM head 的隐藏态）

---

### W2：Prompt 工程 + RAG 从零到框架

**目标**：RAG 是你的"业绩主证据"来源，必须能手写核心组件并给出量化指标。

| Day | 主题 | 具体任务 | 产出/验收 |
|---|---|---|---|
| D1 | Prompt 工程系统化 | 角色/约束/示例/格式四段式；CoT 与 Zero-shot CoT；Few-shot 示例选择策略；Prompt 模板与版本管理 | `prompts/` 目录 + 版本化模板 |
| D2 | 结构化输出 | JSON Schema / Pydantic 校验 + 重试修复；Function Calling 强制 Schema；`response_format`/JSON mode；解析失败的 3 种兜底 | 结构化抽取器，Schema 通过率 > 95% |
| D3 | 注入防御与护栏 | 直接/间接注入（RAG 文档中藏指令）、越狱、数据外泄；防御：输入净化、指令隔离、输出校验、工具白名单、双模型校验 | 攻击样例集 20 条 + 防御后拦截率数据 |
| D4 | RAG 手写（上） | 文档解析（PDF/Word/Markdown → 结构化）、分块策略对比实验（固定/递归/语义/父子块）、Embedding 与余弦相似度、向量入库（先用 NumPy/pgvector） | 分块策略对比表格（块数/召回/延迟） |
| D5 | RAG 手写（下） | BM25 手写实现、向量+BM25 混合检索、RRF 融合、Rerank（Cross-Encoder / bge-reranker）、Query 改写与 HyDE | `retriever.py` 手写版可运行 |
| D6 | RAG 评测与框架化 | 建 50 题评测集（含答案与来源 chunk）；指标 Recall@K / MRR / 忠实度 / 答案相关性；用 LangChain/LlamaIndex 重写并**对比手写版差异**；分析失败 case | 评测报告：手写 vs 框架、Rerank 前后对比 |
| D7 | 进阶检索 + 复盘 | GraphRAG/知识图谱索引思路、多路召回、元数据过滤、父子块召回父块；向量库选型（Milvus/pgvector/Qdrant/ES） | 选型决策记录（写入 §13） |

**W2 深挖问题**：
- 分块大小 512 和 1024 分别适合什么场景？父子块解决什么问题？
- RRF 的 `k` 参数作用？为什么融合比单路好？
- 向量检索的 HNSW 索引原理与 `ef`/`M` 参数对召回率和延迟的影响？
- RAG 答错的四类归因（检索未召回 / 召回不相关 / 模型忽略上下文 / 上下文冲突）及各自对策？
- 上下文里放 20 个 chunk 为什么效果反而变差？

---

### W3：Agent 内核——从零手写，再用框架对照

**目标**：这是整个计划的**心脏**。不做完手写版，前面所有 RAG 和后面所有项目都只是调包。

| Day | 主题 | 具体任务 | 产出/验收 |
|---|---|---|---|
| D1 | 循环控制原理 | ReAct 的 Thought/Action/Observation 循环手写：Prompt 模板、输出解析（含解析失败重试）、工具注册表、循环终止条件（Final Answer / 最大步数 / 超时 / 成本熔断） | `my_react.py` ≤ 200 行，完成"查资料+计算+总结"任务 |
| D2 | 工具调用工程化 | 工具描述怎么写模型才选得对；参数校验与自纠错；并发工具调用；工具失败如何回灌给模型；工具分级（只读/写/危险） | 5 个工具 + 失败回灌机制 |
| D3 | 范式对照 | Plan-and-Execute（先规划后执行，省 Token 但僵化）、Reflexion（自我批判重试）、ToT；各自适用场景与代价；手写 Plan-and-Execute 版 | 两版对比表：Token 消耗 / 步数 / 成功率 |
| D4 | 记忆系统 | 短期：滑窗 + 摘要压缩（何时压缩、压缩什么）；长期：向量记忆 + 重要度打分 + 遗忘策略；工作记忆 vs 情景记忆 vs 语义记忆；Token 预算分配器 | `memory.py`：多轮对话 20 轮不丢关键信息且 Token 不爆 |
| D5 | LangGraph 深挖 | State / Node / Edge / 条件边 / 循环 / 并行 / Checkpointer / `interrupt`（HITL）/ 时间旅行回放；把 D1–D4 用 LangGraph 重写 | LangGraph 版与手写版**行为一致**，并列出框架替你做了什么 |
| D6 | Java 侧对照（差异化） | Spring AI / LangChain4j：ChatClient、Advisor、ToolCallback、ChatMemory、VectorStore；搭一个最小 Java Agent | Java 版 Agent 可跑通（简历亮点：双栈） |
| D7 | 复盘 + 源码阅读 | 读 LangGraph 或 OpenAI Agents SDK 核心源码（Agent Loop 部分）；写"框架的 5 个抽象与它们解决的问题" | 一页笔记 + 一张 Agent Loop 流程图 |

**W3 深挖问题（面试必考）**：
- ReAct 和 Function Calling 的关系？（ReAct 是范式，Function Calling 是让范式稳定落地的机制；纯文本 ReAct 解析脆弱，原生 FC 把解析交给模型训练）
- Agent 死循环怎么发现和打断？（步数上限、动作重复检测、语义相似度检测、成本/时间熔断、人工接管）
- 上下文工程的核心矛盾是什么？（信息完整性 vs 窗口/成本/注意力稀释；解法：分层摘要、子 Agent 隔离、检索式记忆、结构化状态外置）
- 为什么多轮工具调用后模型会"忘记"最初目标？如何缓解？

---

### W4：MCP 深度实战 + 多智能体 + 可观测性

**目标**：MCP 是 2025 年 JD 出现频率飙升的关键词，必须动手写 Server。

| Day | 主题 | 具体任务 | 产出/验收 |
|---|---|---|---|
| D1 | MCP 协议原理 | 为什么需要 MCP（M×N → M+N）、Host/Client/Server 三角色、原语（Tools / Resources / Prompts / Sampling / Roots）、传输（stdio / SSE / Streamable HTTP）、能力协商与生命周期 | 协议时序图（手绘） |
| D2 | 写一个 MCP Server | Python SDK 实现：3 个 Tool + 2 个 Resource + 1 个 Prompt；参数 Schema、错误返回、进度通知、日志规范 | Server 可被 Inspector 调通 |
| D3 | 接入与鉴权 | 在 Claude Desktop / Cursor / 自研 Client 中接入；OAuth2 鉴权、多租户隔离、敏感工具人工确认、审计日志 | 至少 2 个客户端调用成功 + 审计表 |
| D4 | MCP vs Function Calling vs A2A | 三者边界与组合方式；远程 MCP 的安全风险（提示注入、工具投毒、越权）；企业内网私有 MCP 网关设计 | 对比笔记 + 网关架构草图 |
| D5 | 多智能体（上） | Supervisor / Handoff / 群聊 / 黑板四种模式；任务分解与路由；共享状态设计；用 LangGraph 实现 Supervisor + 2 Worker | 3 Agent 协作完成"研究→撰写→审校" |
| D6 | 多智能体（下）+ 冲突消解 | 死锁与循环委派、Token 爆炸、上下文污染、结果冲突仲裁；成本对比：单 Agent vs 多 Agent 做同一任务 | 对比数据表（成本/质量/延迟） |
| D7 | 可观测性基础 | Langfuse / OpenTelemetry GenAI 语义约定：Trace/Span 层级设计、每步 Token 与成本归因、Prompt 版本关联、失败样本回放；搭一个最小 Trace Demo | 一次 Agent 请求在 UI 中完整可视 |

**W4 深挖问题**：
- MCP Server 的 Tool 和 Function Calling 的 tool 是同一个东西吗？（不是：前者是跨进程协议暴露的能力，后者是模型输出格式；MCP 的价值在复用、隔离、动态发现与权限边界）
- 高风险工具"人工确认"具体在链路哪一环拦截？（模型提议 → 编排层策略判定 → 挂起并持久化状态 → 推送审批 → 恢复执行 → 写审计）
- 多 Agent 什么时候是负优化？

---

### W5–W7：旗舰项目（详见 §5）

按"检索子系统 → Agent 内核 → 工具/审批 → 多 Agent → 评测 → 可观测性 → 部署"的顺序增量交付，**每周必须有一个可演示状态**（W5 末端到端跑通 / W6 末全功能 + 评测报告 / W7 末可访问 Demo + 演示视频）。

---

### W8：转化与投递（详见 §9）

D1 简历重写 → D2 项目深挖 30 问 → D3–D4 八股冲刺（§10 题库 + 自测）→ D5 模拟面试（找人或用录音自评）→ D6 投递 + 定向补弱 → D7 复盘与滚动优化。

---

## 5. 旗舰项目设计（贯穿 W5–W7，面试主战场）

### 5.1 项目选择标准

一个能撑住 30 分钟深挖的项目必须同时具备：**真实痛点 + 非平凡架构 + 可量化收益 + 可演示 + 与你存量能力结合**。因此：

- ✅ 选**企业数据分析/运营 Agent**：天然需要 SQL 工具、RAG、多步规划、高风险操作审批、结果可视化（前端优势）、成本敏感（工程优化有故事）。
- ❌ 不选：二次封装的"ChatGPT 套壳"（无技术深度）、纯 Benchmark 刷分项目（无法讲工程权衡）、需要重型沙箱的通用 Coding Agent（2 个月难以做稳）。

### 5.2 项目名与一句话定位

> **DataPilot —— 面向企业数据中台的智能分析 Agent 平台**
> 用户用自然语言提问，Agent 自主完成「意图澄清 → 元数据检索 → SQL 生成与安全校验 → 执行 → 异常检测 → 图表生成 → 结论归因」，高风险操作（数据导出/写库/发报表）经人工审批后执行；全链路可观测、可评测、可回滚。

### 5.3 架构（面试直接画这张图）

```
                        ┌──────────────────────────────────────┐
   Web (React + SSE)    │  Agent Platform (FastAPI)            │
  ┌───────────────┐     │                                      │
  │ 会话/流式输出  │────▶│  Gateway: 鉴权 / 限流 / 会话持久化     │
  │ 工具调用可视化 │◀────│  Router: 意图识别 → 选 Agent          │
  │ 审批中心 UI   │     │  Orchestrator (LangGraph)            │
  │ Artifact 渲染 │     │   ├─ SQL Agent   (规划+生成+自纠错)    │
  └───────────────┘     │   ├─ RAG Agent   (元数据/指标口径)     │
                        │   ├─ Analysis Agent (异常/归因)        │
                        │   └─ Report Agent  (结论/图表)         │
                        ├──────────────────────────────────────┤
   支撑层                │  Tool Registry (含 MCP Client)        │
  ┌──────────────┐      │  Memory: Redis 滑窗 + 向量长期记忆      │
  │ Milvus/Qdrant│◀────▶│  Model Router: 多模型 + 熔断 + 降级     │
  │ Redis / PG   │      │  Guardrails: SQL 白名单/只读/注入防御   │
  │ ClickHouse   │      │  Eval: 评测集 + LLM-as-Judge + 回归    │
  │ Langfuse     │      │  Observability: Trace/Token/成本归因   │
  └──────────────┘      └──────────────────────────────────────┘
```

### 5.4 分周迭代计划

**W5：骨架 + 检索 + Agent 内核跑通**
- D1：项目初始化、数据准备（用公开数据集如电商/TPC-H 造 5–10 张业务表 + 指标口径文档）、PostgreSQL/ClickHouse 起容器
- D2：元数据/口径文档 ETL → 分块 → Embedding → 向量库；复用 W2 的混合检索 + RRF + Rerank
- D3：SQL Agent：Schema 检索 → SQL 生成 → 语法/权限校验 → 执行 → 报错自纠错（最多 3 次）→ 结果表
- D4：LangGraph 编排：Router + SQL Agent + 状态持久化 + 条件分支
- D5：流式输出（SSE）+ 前端最小可用界面（对话 + 结果表格 + 工具调用过程）
- D6：Memory 接入（多轮追问"那上个月呢"必须正确改写 SQL）
- D7：**端到端 Demo 跑通 + 写第一版 README**

**W6：工具/审批/多 Agent/评测**
- D1：Tool Registry 重构（权限分级、超时、重试、幂等、结果裁剪）
- D2：MCP Server 化：把内部工具（查指标口径、执行 SQL、发报表）暴露为 MCP Server，同时以 MCP Client 接入第三方 Server
- D3：Guardrails：SQL AST 解析（只允许 SELECT、强制 LIMIT、行级权限注入）+ 写操作/导出走审批
- D4：HITL 审批流：`interrupt` 挂起 → 前端审批 → 恢复执行；审计日志
- D5：多 Agent：Analysis Agent（异常检测 + 维度下钻）+ Report Agent（结论 + 图表 + 归因），Supervisor 调度
- D6：评测体系：60 题评测集（SQL 正确性 / 答案忠实度 / 拒答准确率），LLM-as-Judge + 规则校验双轨，输出报告
- D7：**全功能版 + 评测报告 + 失败 case 分析文档**

**W7：稳定性 + 可观测性 + 部署 + 包装**
- D1：模型路由与容灾：主备模型切换、熔断、超时退避、语义缓存（相同问题命中缓存）
- D2：Token/成本优化：上下文压缩、工具结果裁剪、小模型路由（简单意图用小模型），量化 before/after 成本
- D3：全链路 Trace 接入 Langfuse，Token 与成本按 Span 归因，失败回放
- D4：并发与压测：异步并发工具、连接池、限流；Locust/k6 压测，给出 QPS/P95 数据
- D5：Docker Compose 一键启动 + 部署到公网（或内网）+ 健康检查
- D6：前端完善（工具调用时间线、审批中心、图表渲染）+ 录制 3–5 分钟演示视频
- D7：**文档收口：架构图、技术决策记录（ADR）、指标看板、演示视频、README**

### 5.5 必须产出的量化指标（写进简历的数字）

| 指标 | 怎么测 | 示例表述 |
|---|---|---|
| SQL 正确率 | 60 题评测集 + 规则校验 | 从 62% → 89% |
| 检索召回 Recall@10 | 标注评测集 | 单路向量 0.71 → 混合+RRF+Rerank 0.91 |
| 首 Token 延迟 | 前端埋点 P50/P95 | P95 从 4.2s → 1.8s（流式 + 缓存 + 小模型路由） |
| 单次会话成本 | Token 归因统计 | 下降 43%（上下文压缩 + 结果裁剪） |
| 高风险操作拦截率 | 注入/越权用例集 | 30 条用例拦截率 100%，误拦 0 |
| 并发能力 | k6 压测 | 50 并发下 P95 < 3s |

> ⚠️ **诚实原则**：所有数字必须来自你自己的评测脚本，能现场跑、能解释统计口径。面试官一旦让你"现场跑一下"，编的数字会立刻崩塌。

### 5.6 备选精简版（若时间只剩 4 周）

保留：RAG 检索子系统（混合检索 + Rerank）+ SQL Agent（含自纠错）+ 审批流 + Trace + 评测报告。砍掉多 Agent、前端精修、压测。**宁可功能少而深，不要功能多而浅**。

---

## 6. 底层原理深挖清单（必须能白板讲清）

按"从输入到输出的完整链路"排列，每一项都要求：能画图 + 能讲权衡 + 能说清失败模式。

**L1 Token 层**
1. BPE/WordPiece 原理、词表大小权衡、中英文 Token 效率、Token 与成本/上下文的关系

**L2 模型内部**
2. Self-Attention 计算过程（Q/K/V、缩放、softmax、多头）、复杂度 O(n²·d)
3. 位置编码演进（绝对 → 相对 → RoPE → ALiBi）与长度外推
4. 归一化与残差（LayerNorm vs RMSNorm、Pre-LN 为什么更稳）
5. FFN 与激活（GELU/SwiGLU）、参数量分布
6. KV Cache：存什么、省什么、显存公式、PagedAttention/MLA 解决什么
7. 推理三阶段（Prefill/Decode）、吞吐与延迟的权衡（batch、continuous batching）
8. 量化原理（对称/非对称、per-channel、GPTQ/AWQ 思路）与精度损失

**L3 输出控制**
9. 采样全参数作用与组合效果；`temperature=0` 为何仍不确定
10. 结构化输出实现路径：Prompt 约束 → JSON mode → 受限解码（Grammar/FSM）→ 为什么受限解码能做到 100% 合法
11. Function Calling 的真实机制：Schema 注入、模型输出结构化片段、并行调用、流式下的增量解析

**L4 训练与对齐（认知级）**
12. 预训练目标（Next Token Prediction）与损失
13. SFT 数据构造与灾难性遗忘
14. LoRA/QLoRA：低秩分解、秩的选择、参数量计算
15. RLHF/DPO 在解决什么（对齐 vs 能力），为什么 Agent 场景常用 RL 微调（工具调用轨迹）

**L5 Agent 系统层（最重要）**
16. Agent 最小完备定义：模型 + 循环 + 工具 + 状态 + 终止条件
17. ReAct / Plan-and-Execute / Reflexion / ToT 的范式差异与代价矩阵
18. 上下文工程：预算分配、分层压缩、子 Agent 隔离、结构化状态外置
19. 记忆系统：写入策略（何时记）、检索策略（怎么取）、遗忘策略（何时删）
20. 工具调用可靠性：描述工程、参数校验、失败回灌、幂等与副作用分级
21. 终止与熔断：步数/时间/成本/重复动作检测/人工接管
22. 多智能体拓扑与失败模式（循环委派、上下文污染、成本爆炸）
23. MCP 协议设计哲学与安全边界
24. 评测：离线集、LLM-as-Judge 的偏差与校准、线上 A/B、回归门禁
25. 可观测性：Span 设计、Token 归因、失败回放、Prompt 版本关联

**自检方式**：每周挑 3 项，对着白板讲 5 分钟并录音，回听后找卡壳点。**讲不出来 = 没学会。**

---

## 7. 每日/每周固定动作模板

### 每日（3h 范例）

| 时段 | 时长 | 内容 |
|---|---|---|
| 20:00–20:40 | 40min | 理论学习（文档/论文/视频），只记 3 个关键结论 |
| 20:40–22:30 | 110min | 编码：只做今天清单里的事，**禁止边看教程边抄代码**，先自己想 15 分钟再看 |
| 22:30–22:50 | 20min | 写笔记：今天解决了什么问题、踩了什么坑、还有什么不懂 |
| 22:50–23:00 | 10min | 提交代码（commit message 规范）+ 明天计划 |

### 每周日（1h）

1. 跑通本周产出（旧代码是否还能跑？）
2. 回答本周"深挖问题"清单，卡壳项记入下周
3. 更新 `PROGRESS.md`：本周产出 / 未达标项 / 下周调整
4. 回看 Gate 是否通过，未通过则下周前两天补课

### 三条纪律（决定成败）

1. **手写优先**：任何核心机制（ReAct、BM25、RRF、注意力）先手写再用框架，否则面试第一个"底层怎么实现的"就露馅。
2. **产出物驱动**：每周必须有可运行代码 + 文档，不接受"看完了理解了"。
3. **面向面试学习**：每学一个点，问自己"面试官会怎么追问三层？"并把答案写下来。

---

## 8. 资源清单（精选，不贪多）

### 必读论文（8 篇，每篇读出 3 句话即可）
1. Attention Is All You Need — arXiv:1706.03762
2. ReAct: Synergizing Reasoning and Acting — arXiv:2210.03629
3. Reflexion — arXiv:2303.11366
4. Toolformer — arXiv:2302.04761
5. RAG 原始论文 — arXiv:2005.11401
6. Lost in the Middle — arXiv:2307.03172
7. LoRA — arXiv:2106.09685
8. Chain-of-Thought — arXiv:2201.11903

### 课程与教材
- Karpathy《Neural Networks: Zero to Hero》+ nanoGPT（**W1 核心，必做**）
- Karpathy《Let's build GPT》/《Let's build the GPT Tokenizer》
- DeepLearning.AI 短课：Building Systems with the ChatGPT API / Functions, Tools and Agents / LangChain for LLM App Development

### 官方文档（比教程重要）
- OpenAI Platform Docs：Function Calling、Structured Outputs、Streaming
- Anthropic Docs：Building effective agents、Context Engineering、Tool use
- MCP 官方规范与 SDK（modelcontextprotocol.io）
- LangGraph 文档（Persistence / Human-in-the-loop / Multi-agent 章节）
- Spring AI / LangChain4j 官方文档（Java 对照）
- Langfuse / OpenTelemetry GenAI 语义约定

### 框架
- 编排：LangGraph（主）、OpenAI Agents SDK、Spring AI（Java）
- RAG：LlamaIndex / LangChain（对照学习，不作为主线）
- 向量库：Milvus 或 Qdrant（二选一）+ pgvector（轻量场景）
- 可观测：Langfuse（自托管或云）
- 本地模型：Ollama + Qwen2.5（备胎与成本控制）

### 社区与信息源
- GitHub 面试资料库：[ai-agent-interview-guide](https://github.com/bcefghj/ai-agent-interview-guide)（八股 + 三语言项目）、[ai-agent-engineer-handbook](https://github.com/harrisliangsu/ai-agent-engineer-handbook)（JD 汇总）
- 关注方向：MCP 生态更新、Agent 评测基准（τ-bench、SWE-bench 思路）、上下文工程实践

---

## 9. 第 8 周：简历、项目包装与面试转化

### 9.1 简历定位（一句话）

> 6 年全栈工程师｜专注 LLM Agent 应用与平台建设｜擅长将 Agent 能力工程化落地（RAG 召回优化、工具治理、HITL 审批、可观测性与成本优化）

**不要**写"转行""自学""热爱 AI"。**要**写"把 Agent 做进生产系统"。

### 9.2 项目描述的 STAR + 数字模板

> **DataPilot · 企业数据中台智能分析 Agent 平台**（个人项目 / 或与 X 人协作）
> **S/T**：业务同学取数依赖数据团队，平均需求周期 2–3 天，指标口径不统一导致返工。
> **A**：
> - 设计并实现基于 **LangGraph** 的多 Agent 编排（Router + SQL/RAG/Analysis/Report），以 Redis Checkpointer 支持会话持久化与断点恢复
> - 构建混合检索链路（向量 + **手写 BM25** + **RRF 融合** + Cross-Encoder Rerank）检索指标口径与元数据，配合 Query 改写与父子块召回，**Recall@10 从 0.71 提升至 0.91**
> - 实现 SQL 安全护栏：AST 解析强制只读 + 行级权限注入 + 强制 LIMIT，高风险操作（导出/写库）经 **Human-in-the-loop 审批**后执行，30 条越权/注入用例**拦截率 100%**
> - 自研 **MCP Server** 暴露 5 个内部能力（口径查询/SQL 执行/报表生成等），支持多客户端复用与权限隔离
> - 全链路接入 **Langfuse**，按 Span 归因 Token 与成本；通过上下文压缩 + 工具结果裁剪 + 小模型路由，**单次会话成本下降 43%，P95 首 Token 延迟从 4.2s 降至 1.8s**
> - 建立 60 题评测集（规则校验 + LLM-as-Judge 双轨），**SQL 正确率 62% → 89%**，并作为 CI 回归门禁
> **R**：取数自助率提升，平均需求响应从天级降至分钟级；平台沉淀为内部 Agent 基础设施。

### 9.3 你的三个"故事弹药"

面试一定会问"讲讲你怎么解决一个难题"。准备三个可深挖 10 分钟的故事：
1. **召回优化故事**：从"答不准"到定位到分块策略 + 混合检索缺失 + Rerank 缺位，逐项做实验给出数据。
2. **成本与延迟故事**：发现 Token 爆炸源于工具结果全量回灌 + 历史无压缩，通过分层摘要 + 结果裁剪 + 小模型路由解决。
3. **Agent 失控故事**：SQL Agent 陷入"报错→改 SQL→再报错"死循环，通过重复动作检测 + 错误分类重试 + 步数熔断 + 兜底澄清解决。

### 9.4 目标公司分层投递

| 层次 | 类型 | 说明 |
|---|---|---|
| 第一梯队 | 金融科技 / 政企数字化 / 传统软件公司的 AI 部门 | 最看重 Java 背景 + 工程化 + 私有化部署，你的匹配度最高 |
| 第二梯队 | AI 应用创业公司（客服/数据分析/办公 Agent） | 看重项目深度与动手能力，Python 为主，需证明能写生产 Python |
| 第三梯队 | 大厂大模型应用/平台岗 | 竞争最激烈，作为冲刺目标；差异化在于"工程化 + 可观测性 + 成本优化" |
| 特别关注 | 岗位 JD 中出现 "Spring AI / LangChain4j / Java + 大模型" | 竞争者极少，命中率最高，务必定向投 |

---

## 10. 高频面试题预演（含答题骨架）

**A. 基础与原理**
1. Transformer 的注意力怎么算？为什么要除以 √d_k？
2. KV Cache 原理？显存怎么估算？PagedAttention 解决什么？
3. temperature/top_p 怎么影响输出？你线上怎么设的？
4. 大模型为什么会幻觉？你的系统怎么抑制？→ 骨架：成因三类 → RAG 提供事实 → 引用溯源 → 拒答策略 → 输出校验 → 评测验证
5. RAG 和微调怎么选？→ 骨架：知识是否高频变化 / 是否需要改变风格与格式 / 数据量 / 成本 / 可解释性，给决策树

**B. RAG**
6. 你的分块策略是什么？为什么？
7. 混合检索为什么比纯向量好？RRF 怎么算？
8. Rerank 的作用与代价？什么情况不值得加？
9. 怎么评估 RAG？指标有哪些？怎么定位是检索问题还是生成问题？
10. 多轮对话中 query 怎么改写？指代消解怎么做？

**C. Agent 核心**
11. 讲清 ReAct 的完整循环，以及它和 Function Calling 的关系
12. Plan-and-Execute vs ReAct，各自适合什么任务？
13. Agent 死循环/跑偏怎么处理？
14. 上下文越来越长怎么办？→ 骨架：预算分配 → 分层摘要 → 工具结果裁剪 → 结构化状态外置 → 子 Agent 隔离 → 检索式记忆
15. 记忆系统怎么设计？短期和长期怎么配合？什么时候遗忘？
16. 多 Agent 什么时候是负优化？你怎么判断？
17. 工具调用失败/参数错误怎么处理？
18. 怎么让模型稳定选择正确的工具？→ 工具命名与描述工程、减少工具数量、分类路由、Few-shot 示例、参数默认值

**D. MCP 与生态**
19. MCP 是什么？解决什么问题？和 Function Calling 的区别？
20. MCP 的传输方式有哪些？各自适用场景与安全考虑？
21. 高风险工具怎么做人工确认？链路怎么实现？
22. MCP Server 的安全风险有哪些？怎么防？

**E. 工程化（你的加分项）**
23. Agent 服务怎么做状态持久化与断点续跑？
24. 怎么保证 Agent 系统的高可用？模型挂了怎么办？
25. 怎么监控 Agent？你会看哪些指标？
26. Token 成本怎么优化？给出你项目里的具体数字
27. 怎么给 Agent 做评测和回归？LLM-as-Judge 靠谱吗？偏差怎么校准？
28. 并发场景下会话状态怎么隔离？长任务怎么做异步化？
29. 提示注入怎么防？RAG 文档里藏指令怎么办？
30. 如果让你从零设计一个企业级 Agent 平台，你会怎么分层？

**答题通用骨架**：**结论先行 → 原理 → 我的实现/数据 → 权衡与取舍 → 如果重做会怎么改**。这个结构能显著提升面试官评价。

---

## 11. 验收标准：什么叫做"可以投递了"

全部满足才投递：

- [ ] 手写 ReAct Agent（≤200 行）能从零默写核心逻辑
- [ ] 手写 BM25 + 向量检索 + RRF，能解释每一步
- [ ] 能白板画 Attention 并讲 KV Cache 显存估算
- [ ] 自研 MCP Server 被 2 个客户端成功调用
- [ ] 旗舰项目可一键启动，有 Demo 视频与在线地址
- [ ] 有 60 题评测集与 before/after 指标报告，能现场跑
- [ ] Langfuse（或同类）Trace 可视化一次完整请求，能讲 Token 归因
- [ ] 简历项目描述含 ≥5 个量化数字，且每个数字你都能解释口径
- [ ] §10 的 30 题能连续答完 25 题以上且不卡壳
- [ ] 3 个"难题故事"能讲 10 分钟并回答 3 层追问
- [ ] `PROGRESS.md` 完整记录 8 周轨迹（面试可展示学习与执行力）

---

## 12. 风险与备选方案（Plan B）

| 风险 | 触发信号 | 应对 |
|---|---|---|
| 时间不够，项目做不完 | W5 末未端到端跑通 | 立刻启用 §5.6 精简版，砍多 Agent 与前端精修 |
| Python 拖后腿 | W1 末写异步工具调用仍吃力 | W2 全程用 Python 不碰 Java；用 AI 辅助但要逐行读懂 |
| 检索效果调不上去 | W2 末 Recall 无明显提升 | 先确保评测集质量；退化为"元数据过滤 + 关键词 + 向量"三路，别硬啃 GraphRAG |
| 框架版本变动踩坑 | 依赖频繁报错 | 锁定版本 + Docker 固定环境；核心逻辑保持手写，隔离框架依赖 |
| 无 GPU / 成本压力 | API 费用上升 | 主用国产模型（DeepSeek/Qwen）；本地 Ollama 跑小模型做开发与评测；缓存复用 |
| 投递无回音 | 2 周内 30 投 0 面 | 检查简历是否有量化数字；转向 Java + 大模型的定向岗位；补充开源贡献或技术博客提升可信度 |
| 面试卡在"没有生产经验" | 一面后无进展 | 用"项目 + 指标 + 失败 case + 取舍决策"四件套应对；主动讲工程化细节把话题引到你的主场 |

---

## 13. 附录：技术选型决策记录（ADR，面试加分项）

**ADR-1 为什么用 LangGraph 而不是自己写编排？**
自研编排灵活但需自己做状态持久化、中断恢复、并行与重放。LangGraph 的 Checkpointer 与 `interrupt` 直接满足 HITL 与断点续跑需求；代价是抽象层带来的调试成本与版本耦合。取舍：核心循环仍手写理解，生产编排用 LangGraph，并保留替换能力（编排层与业务逻辑解耦）。

**ADR-2 为什么向量库选 Milvus/Qdrant 而非 pgvector？**
数据量与过滤复杂度：pgvector 在千万级向量 + 复杂元数据过滤时性能下降明显，且与业务库争资源。若规模在百万级以内且已有 PG，pgvector 更省运维。**决策依据是数据规模与运维成本，而非技术时髦度。**

**ADR-3 为什么坚持混合检索 + Rerank？**
纯向量对专有名词、指标编号、缩写召回差（Embedding 语义化丢失精确匹配能力）；BM25 补精确匹配，Rerank 用 Cross-Encoder 修正排序。代价是延迟增加约 200–400ms 与一次额外模型调用，通过缓存与并行召回缓解。

**ADR-4 为什么记忆用"滑窗 + 摘要 + 向量"三层而非全量历史？**
全量历史导致 Token 成本线性增长与注意力稀释（Lost in the Middle）；纯摘要丢失细节。三层结构：近 N 轮原文保细节、更早对话摘要保脉络、关键事实写入长期向量记忆按需检索。Token 预算在每轮动态分配。

**ADR-5 为什么做 MCP Server 化？**
把内部能力（口径查询、SQL 执行、报表生成）从应用内函数提升为协议化服务，实现跨客户端复用（自研 Web / Claude Desktop / Cursor / IDE）、进程隔离与独立权限边界，避免把工具实现与宿主应用强耦合。

---

## 附：本仓库文件说明

| 文件 | 用途 |
|---|---|
| `PLAN.md` | 本文件，8 周总计划 |
| `WEEK0_SETUP.md` | 环境搭建步骤 + Python 自测题 + 环境自检脚本 |
| `PROGRESS.md` | 每周进度与复盘记录（**面试可展示**，务必认真写） |
| `RESOURCES.md` | 论文/文档/课程链接清单（按周索引） |
| `projects/` | 旗舰项目代码目录 |
| `notes/` | 学习笔记与原理图解 |
| `labs/` | 手写实验（BPE、mini-GPT、ReAct、BM25、RRF 等） |

---

**最后一句**：2 个月的时间不够你成为"AI 算法专家"，但足够你成为"**能把 Agent 做进生产系统的工程师**"——而市面上真正缺的，正是后者。你的 6 年工程经验不是需要藏起来的过去，而是这个新身份最硬的底牌。
