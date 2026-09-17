# W1 实验包：手写大模型底层（BPE / Attention / KV Cache / mini-GPT）

> 纯 NumPy 实现，**不使用 PyTorch**。目的不是训练出好模型，而是让每一层都透明：
> 面试时能白板画出形状变化、能推导梯度、能写出 KV Cache 显存公式。

对应 `PLAN.md` 的 **W1：大模型底层原理 + API 层彻底打通**。

---

## 快速开始

```bash
cd labs/w1_llm
source env.sh            # 固定 only-managed Python + 工作区缓存
uv sync                  # 安装依赖（numpy / pytest / ruff）

uv run python -m w1_llm.bpe                 # 1. BPE 分词器演示
uv run python -m w1_llm.attention           # 2. 注意力 + KV Cache 自检
uv run python -m w1_llm.mini_gpt            # 3. 训练 + 生成（约 10 秒）
uv run python -m w1_llm.mini_gpt --check-grad   # 4. 数值梯度校验（必跑）
uv run python -m w1_llm.diag_grad           # 5. 组件级梯度诊断

uv run pytest -v                            # 全部测试（30 项）
uv run pytest -m "not slow"                 # 跳过梯度校验，快速跑
```

**环境要求**：Python ≥ 3.12（3.9 跑不了，`TaskGroup`/新 typing 语法需要）。本包已用 `3.12.14` 验证通过。

---

## 文件说明

| 文件 | 内容 | 关键点 |
|---|---|---|
| `src/w1_llm/bpe.py` | 字节级 BPE 分词器（训练/编码/解码/持久化） | byte-level 保证零 `<UNK>`；编码按 merge rank 贪心 |
| `src/w1_llm/attention.py` | softmax、因果掩码、scaled dot-product attention、RoPE、`KVCache`、`MultiHeadAttention` | 含 KV Cache 显存公式实现 |
| `src/w1_llm/mini_gpt.py` | RMSNorm、SwiGLU、Block、采样、GPT、Adam、**手写反向传播 + 数值梯度校验** | 完整可训练，loss 5.995 → 0.002 |
| `src/w1_llm/diag_grad.py` | 组件级有限差分诊断 | 定位反向传播错源用 |
| `tests/test_w1_llm.py` | 30 项测试 | 含 2 个针对真实 bug 的回归测试 |

---

## 实测结果（本机验证）

### 1. mini-GPT 训练收敛

```
参数量 851,072
初始 loss = 5.995      （理论值 ln(V) = ln(400) = 5.991  ✅ 初始化正确）
step  25  loss = 0.1039
step  50  loss = 0.0107
step 200  loss = 0.0022
耗时 1.4s   loss: 5.995 -> 0.002（下降 100%）
```

贪心解码逐字复现训练语料 —— 说明模型真的学到了，而不是随机输出。

### 2. 数值梯度校验（证明手写反向正确）

21 个参数张量随机抽查，解析梯度 vs 中心差分，相对误差 **~1e-9**：

```
b1.w_up       0.000379   0.000379   7.11e-08  ✅
b0.norm1.g    0.011656   0.011656   3.28e-09  ✅
wte          -0.079391  -0.079391   1.97e-08  ✅
...
结论：✅ 所有抽查元素的解析梯度与数值梯度一致，反向传播实现正确
```

### 3. KV Cache 正确性与收益

```
全量前向 vs 增量缓存 最大误差 = 4.44e-15     （机器精度级一致）
生成 40 token：带缓存 11 ms | 不带 31 ms | 加速 2.86x
```

> 配置小、序列短时差距有限；真实模型（32 层 / 4K 上下文）下
> 投影量从 O(T²) 降到 O(T)，加上显存带宽瓶颈，加速可达 10x 以上。

---

## 本包调试过程中抓到的 3 个真实 bug（**W1 最值钱的部分**）

这三个 bug 都不是"打错字"，而是**对原理理解不到位才会犯**的错。把它们记住，面试时能讲出深度。

### Bug 1：`dV = W @ d_out` 应为 `dV = Wᵀ @ d_out`

注意力输出是 `out[t,:] = Σ_j W[t,j]·V[j,:]`。对 V 求梯度时，W 是"按列"加权：

```
dV[j,:] = Σ_t W[t,j] · dout[t,:] = (Wᵀ @ dout)[j,:]      ← 要转置
dW[t,j] = dout[t,:] · V[j,:]ᵀ                            ← 这一路不转置
```

**踩坑表现**：`w_proj/b_proj` 梯度全对（因为都在下游），但 `w_qkv` 的 V 块梯度完全错，
`dx` 也跟着错。诊断手段：`diag_grad.py` 逐模块定位 + 有限差分。

**教训**：写成 `@` 之前，先在纸上把维度对齐写出来。

### Bug 2：多层模型下每层各自读 `cache.seq_len` 导致位置漂移 ⭐ 最隐蔽

```
for i, blk in enumerate(self.blocks):
    x = blk.forward(x, cache, i)     # 每层内部各自读 cache.seq_len 算 RoPE 位置
```

第 0 层追加 KV 后 `seq_len` 就变了，第 1 层拿到**偏移一位**的 position：

| | layer 0 的 positions | layer 1 的 positions |
|---|---|---|
| 期望 | `[3.]` | `[3.]` |
| 实际 | `[3.]` ✅ | `[4.]` ❌ |

**踩坑表现**：`T=1` 增量解码时 logits 与全量前向差 **1e-2 量级**（远超浮点误差），
贪心生成会在第 7 个 token 分叉；`T=1` 单层测试却完全正常（1e-16），极易误判为"浮点噪声"。

**修复**：位置在 `GPT.forward` 开头**只算一次**，显式传给所有层。

**教训**：**任何"会被后续步骤修改的共享状态"都不能在循环内读取**。这和 Java 里
"在遍历中修改集合"、分布式里"读共享 offset"是同一类错误。回归测试见
`test_multilayer_cache_positions_not_drifted`。

### Bug 3：`params()` 与 `Block.backward()` 的键名不一致

`params()` 返回 `b0.w_up`，`backward()` 返回 `b0.ffn.w_up`，梯度校验直接 `KeyError`。

**教训**：参数命名要有单一权威来源，否则优化器会静默跳过参数（梯度为 None → 参数永不更新，
loss 只是下降变慢而不报错，最难查）。

---

## 刻意保留的"诚实"现象

**高温采样会生成乱码字符**（如 `�O��`）：

这是 byte-level BPE 的固有特性，不是 bug。词表里同时存在"半个汉字"的字节 token，
`temperature=1.5` 时模型会采样出这些不完整字节序列，UTF-8 解码失败。

正好解释了为什么**线上必须做采样参数调优**：

| 目标 | 建议 |
|---|---|
| 事实性问答 / 工具调用参数 | `temperature=0` 或 0.1，配 top_p=0.9 |
| 一般对话 | `temperature=0.7`, `top_p=0.95` |
| 创意生成 | `temperature=1.0~1.2`，但需接受质量波动 |
| 关键：**结构化输出** | 不要靠采样，要用 JSON Schema / 受限解码 |

---

## 面试自测题（能答上才算过关）

1. Attention 里为什么除以 `√d_k`？不除会怎样？（`test_attention_scores_scale` 有数值验证）
2. KV Cache 显存公式？为什么它随 T 和 batch 线性增长？PagedAttention 解决什么？
3. RoPE 相比"绝对位置相加"的优势？为什么 `QᵀK` 只依赖相对距离？
4. 为什么 `temperature=0` 也不能保证跨次完全一致？（本包 `sample_token` 的注释里有答案）
5. top_k 和 top_p 的区别？各自适合什么场景？
6. 为什么用 SwiGLU 而不是 ReLU？隐藏维为什么取 8/3·d？
7. Pre-Norm 为什么比 Post-Norm 好训？
8. weight tying（输出层复用输入 embedding）省多少参数？有什么副作用？
9. 为什么用 RMSNorm 而不用 LayerNorm？省了什么？
10. byte-level BPE 为什么能保证零 `<UNK>`？代价是什么？

---

## 下一步（W1 剩余任务）

本包覆盖了 W1 的"原理层"。按 `PLAN.md`，W1 还需完成 **API 层**：

- [ ] 用 `httpx` 裸调 Chat Completions（非流式 + 流式 SSE + 多轮上下文）
- [ ] 手写 Function Calling 完整链路：tools schema → 解析 tool_calls → 本地执行 → 回填 tool 消息 → 二次请求
- [ ] 采样参数与结构化输出的线上调优实验
- [ ] 精读《Attention Is All You Need》《LoRA》并写一页笔记
