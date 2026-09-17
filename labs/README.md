# labs —— 手写实验区（面试的核心弹药）

> 纪律：**核心机制先手写，再用框架**。每个实验都要能在面试时从零默写出来。

| 目录 | 对应周 | 内容 | 验收 | 状态 |
|---|---|---|---|---|
| `w0_env/` | W0 | `check_env.py` 环境自检、30 道 Python 自测题 | 全绿 | 🟡 脚本就绪 |
| `w1_llm/` | W1 | 手写 BPE、Attention、RoPE、KV Cache、mini-GPT（纯 NumPy，含手写反向传播 + 数值梯度校验） | 梯度校验全过 / loss 5.995→0.002 / 30 项测试通过 | ✅ 原理层完成，待补 API 层 |
| `w2_rag/` | W2 | `chunking.py`、`my_bm25.py`、`my_rrf.py`、`retriever.py`、`rerank.py`、`eval_rag.py` | 评测报告有 before/after 数字 | ⬜ |
| `w3_agent/` | W3 | `my_react.py`、`tools.py`、`memory.py`、`langgraph_version.py`、`api_safety.py` | 手写 ReAct ≤200 行可跑通 | ⬜ |
| `w4_mcp/` | W4 | `my_mcp_server.py`、`mcp_client.py`、`multi_agent.py`、`trace_demo.py` | Server 被 2 个客户端调通 | ⬜ |

## 运行 W1 实验

```bash
cd labs/w1_llm
source env.sh
uv sync
uv run pytest -v
uv run python -m w1_llm.mini_gpt --check-grad
```

`w1_llm/README.md` 记录了调试中抓到的 **3 个真实 bug**（dV 需转置 / 多层 KV Cache 位置漂移 /
参数命名不一致）及其定位方法 —— 这是面试可直接讲的深度素材，比"我会用 LangChain"值钱得多。
