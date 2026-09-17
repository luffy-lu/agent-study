# labs —— 手写实验区（面试的核心弹药）

> 纪律：**核心机制先手写，再用框架**。每个实验都要能在面试时从零默写出来。

| 目录 | 对应周 | 内容 | 验收 |
|---|---|---|---|
| `w0_env/` | W0 | `check_env.py` 环境自检、30 道 Python 自测题 | 全绿 |
| `w1_llm/` | W1 | `my_bpe.py`、`attention.py`、`mini_gpt.py`、采样参数对比实验 | mini-GPT 能训练收敛 |
| `w2_rag/` | W2 | `chunking.py`、`my_bm25.py`、`my_rrf.py`、`retriever.py`、`rerank.py`、`eval_rag.py` | 评测报告有 before/after 数字 |
| `w3_agent/` | W3 | `my_react.py`、`tools.py`、`memory.py`、`langgraph_version.py`、`api_safety.py` | 手写 ReAct ≤200 行可跑通 |
| `w4_mcp/` | W4 | `my_mcp_server.py`、`mcp_client.py`、`multi_agent.py`、`trace_demo.py` | Server 被 2 个客户端调通 |
