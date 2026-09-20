# notes —— 学习笔记与原理图解

> **建议先读**：[W1概念补充.md](W1概念补充.md) —— 从 Java 后端视角理解大模型，
> 把 LLM 概念映射到你已有的知识上。读完再看代码。

## 推荐顺序

1. **`W1概念补充.md`** —— 大模型是什么、概念映射表、读代码主线
2. 跑实验：`cd labs/w1_llm && uv run python -m w1_llm.demo`
3. 再回来看 `labs/w1_llm/` 的源码

## 建议文件（每学完一个主题写一页，A4 一页为限）

- `w1_attention.md` —— 手画 Attention 推导 + KV Cache 显存估算公式
- `w1_sampling.md` —— 采样参数对输出的影响实验记录
- `w1_finetune_decision.md` —— RAG vs 微调 vs Prompt 决策树
- `w2_chunking.md` —— 分块策略对比表
- `w2_rag_failure.md` —— RAG 失败四类归因与对策
- `w3_react_loop.md` —— ReAct 完整流程图（手绘扫描）
- `w3_paradigm_compare.md` —— ReAct / Plan-Execute / Reflexion 代价矩阵
- `w3_context_engineering.md` —— 上下文预算分配方案
- `w4_mcp_protocol.md` —— MCP 时序图 + 与 Function Calling 边界
- `w4_multiagent.md` —— 多 Agent 拓扑与失败模式

**要求**：每个笔记结尾必须有「面试官可能追问的 3 个问题及我的答案」。
