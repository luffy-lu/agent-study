"""W1 引导实验 —— 从"看得见的现象"建立直觉。

这个文件**不需要你先懂 Transformer**。它的逻辑是：
    先让你看见现象 → 再解释为什么 → 最后才去看代码怎么实现。

十个实验各回答一个具体问题：
    E1  大模型到底在算什么？
    E2  「训练」是什么意思？
    E3  采样参数在干什么？
    E4  它真的「学会」了吗？
    E5  没见过的输入会怎样？（幻觉的雏形）
    E6  为什么它不会偷看后面的字？（因果掩码）
    E7  KV Cache 到底缓存了什么？
    E8  位置信息传错会怎样？（我修过的真实 bug）
    E9  这个玩具和 GPT-3 差多远？
    E10 BPE 到底是什么？

运行：
    cd labs/w1_llm
    uv run python -m w1_llm.demo
"""

from __future__ import annotations

import time

import numpy as np

from .attention import KVCache, apply_rope, softmax
from .bpe import BPETokenizer
from .mini_gpt import GPT, Adam, causal_mask_bool

LINE = "=" * 74

CORPUS = (
    "Agent 通过工具调用与外部世界交互。"
    "Agent 会先思考，再行动，然后观察结果。"
    "Agent 需要记忆来保持多轮对话的连贯性。"
    "一个 Agent 由模型、工具、记忆和循环控制组成。"
    "工具调用失败时，Agent 应该重试或换个思路。"
    "好的 Agent 知道什么时候该停下来问用户。"
) * 4

BLOCK = 96
TRAIN_STEPS = 150


def title(n: str, q: str) -> None:
    print(f"\n{LINE}\n{n}  {q}\n{LINE}")


def show_token(tok: BPETokenizer, i: int) -> str:
    """把 token 显示成人能读的样子。

    byte-level BPE 的 token 可能是"半个汉字"，直接 decode 会得到乱码替换符。
    所以先尝试正常解码，失败就显示原始字节 —— 顺便让你看清它的本质。
    """
    raw = tok.vocab[int(i)]
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<字节 {raw.hex()}>"
    if s == "\n":
        return "\\n"
    if s == "\t":
        return "\\t"
    if s == " ":
        return "␣"
    return s if s.strip() else "<空白>"


# ---------------------------------------------------------------------------
# 准备
# ---------------------------------------------------------------------------


def build_model(seed: int = 42) -> tuple[BPETokenizer, GPT, np.ndarray, np.ndarray, float, float]:
    """训练玩具模型。

    为什么是 150 步？
        步数太少时模型只会胡说，采样参数看不出区别；
        步数太多时它把语料背死，概率分布极度尖锐（top1 接近 100%），采样参数又失效。
        150 步左右 top1≈52%、熵≈2.45 bit —— 既有明确偏好又有不确定性，适合观察。
    """
    print("【准备】训练一个玩具模型（约 2 秒）")
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)

    data = np.array(tok.encode(CORPUS), dtype=np.int64)
    x = data[:BLOCK][None]
    y = data[1 : BLOCK + 1][None]

    model = GPT(tok.next_id, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK, seed=seed)
    print(f"  词表大小 {tok.next_id} | 训练语料 {len(data)} 个 token | 参数量 {model.n_params():,}")

    params = model.params()
    opt = Adam(lr=3e-3)
    _, loss0 = model.forward(x, y)
    loss = loss0
    for _ in range(TRAIN_STEPS):
        _, loss = model.forward(x, y)
        opt.step(params, model.backward())
    assert loss0 is not None and loss is not None
    print(f"  训练完成：loss {loss0:.3f} -> {loss:.4f}")
    return tok, model, x, y, float(loss0), float(loss)


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------


def e1_next_token(tok: BPETokenizer, model: GPT) -> None:
    title("E1", "大模型到底在算什么？—— 它在给「下一个字」打分")

    prompt = "Agent 通过工具"
    ids = tok.encode(prompt)
    logits, _ = model.forward(np.array([ids]))
    probs = softmax(logits[0, -1], axis=-1)

    print(f"输入：{prompt!r}")
    print("模型输出的**不是一句话**，而是：词表里每个 token 的分数（一个概率分布）\n")
    print(f"{'排名':<5}{'下一个 token':<20}{'概率':>9}   可视化")
    print("-" * 72)
    for rank, i in enumerate(np.argsort(-probs)[:8], 1):
        p = float(probs[i])
        print(f"{rank:<5}{show_token(tok, int(i)):<20}{p:>8.2%}   {'█' * max(1, int(p * 45))}")

    ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
    print(f"\n  概率总和 = {probs.sum():.4f}（必然为 1）")
    print(f"  熵 = {ent:.2f} bit —— 衡量「有多不确定」，0 表示百分百确定")
    print(f"  最高概率 = {probs.max():.2%} —— 注意它**不是** 100%")
    print()
    print(">>> 关键认知：")
    print("    模型没有「理解」这句话，也没有去数据库查。它只是算出一个概率分布。")
    print("    所谓「生成」，就是从这个分布里挑一个接到句尾，然后再算一次。")
    print("    **整个大模型 = 一个循环 + 一个猜下一个 token 的函数。**")
    print()
    print("    注意它是一个「分布」而不是确定答案：挑哪个由采样参数决定（见 E3），")
    print("    这就是同一个问题问两次、答案不一样的原因。")
    print()
    print("    这跟 Java 的 switch/if 完全不同：它没有分支逻辑，只有矩阵乘法。")


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------


def e2_training(tok: BPETokenizer, loss0: float) -> None:
    title("E2", "「训练」是什么意思？—— 就是调数字，让猜得更准")

    model = GPT(tok.next_id, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK, seed=7)
    data = np.array(tok.encode(CORPUS), dtype=np.int64)
    x, y = data[:BLOCK][None], data[1 : BLOCK + 1][None]
    params, opt = model.params(), Adam(lr=3e-3)

    _, first = model.forward(x, y)
    print(f"第 0 步（随机权重）loss = {first:.3f}")
    print(f"  理论值 ln(词表大小) = ln({tok.next_id}) = {np.log(tok.next_id):.3f}")
    print("  → 意思是：完全随机猜，等于在 400 个选项里瞎选，没有任何信息\n")

    marks = {1: "和瞎猜一样", 25: "迅速下降", 50: "学到了搭配规律", 100: "已经很准", 150: "接近背熟"}
    print(f"{'步数':<8}{'loss':>10}   说明")
    print("-" * 62)
    for step in range(1, TRAIN_STEPS + 1):
        _, loss = model.forward(x, y)
        opt.step(params, model.backward())
        if step in marks:
            print(f"{step:<8}{loss:>10.4f}   {marks[step]}")

    print()
    print(">>> loss 是什么？")
    print("    loss = -log(模型给正确答案的概率) 的平均值")
    print("    loss=5.99 → 正确 token 的概率只有 1/400（等于瞎猜）")
    print("    loss=0.02 → 概率已经很高（几乎必然猜对）")
    print()
    print(">>> 「训练」= 反复做两件事：")
    print("    1. 前向：算预测 → 和正确答案比 → 得到 loss")
    print("    2. 反向：算出「每个权重该往哪个方向调、调多少」（这就是梯度）")
    print("    然后微调所有权重。真实大模型要重复几百万次、用几千张 GPU。")
    print()
    print(">>> 所以「模型的知识」是什么？")
    print("    就是一坨训练后固定下来的数字（权重）。")
    print("    没有任何逻辑规则、没有 if-else，纯粹是统计规律被压进矩阵里。")
    print("    这也解释了为什么它有时会胡说：它优化的是「概率最大」，不是「说真话」。")


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------


def e3_sampling(tok: BPETokenizer, model: GPT) -> None:
    title("E3", "采样参数在干什么？—— 同一个模型，行为完全不同")

    prompt_text = "Agent 通过工具"
    prompt = np.array([tok.encode(prompt_text)])
    logits, _ = model.forward(prompt)
    base = softmax(logits[0, -1], axis=-1)
    print(f"先看清「温度」对分布做了什么（prompt = {prompt_text!r}）：\n")
    print(f"{'temperature':<14}{'处理后最高概率':>16}{'熵(bit)':>10}   直观感受")
    print("-" * 68)
    for t in (0.3, 1.0, 2.0):
        p = softmax(logits[0, -1] / t, axis=-1)
        ent = float(-np.sum(p * np.log2(p + 1e-12)))
        desc = "更尖锐、更保守" if t < 1 else ("原始分布" if t == 1 else "更平坦、更随机")
        print(f"{t:<14}{p.max():>15.2%}{ent:>10.2f}   {desc}")
    print(f"（原始分布 top1 = {base.max():.2%}，熵 = "
          f"{-np.sum(base * np.log2(base + 1e-12)):.2f} bit）")

    print(f"\n再用不同参数真的生成一段（prompt = {prompt_text!r}）：\n")
    configs = [
        ("temperature=0 贪心", dict(temperature=0.0)),
        ("temperature=0.5", dict(temperature=0.5, top_p=None)),
        ("temperature=1.5", dict(temperature=1.5, top_p=None)),
        ("top_k=3", dict(temperature=1.0, top_k=3, top_p=None)),
        ("top_p=0.9", dict(temperature=1.0, top_k=None, top_p=0.9)),
    ]
    for label, kw in configs:
        out = model.generate(prompt, max_new_tokens=30, repetition_penalty=1.0, seed=3, **kw)
        print(f"[{label:<20}] {tok.decode(out[0].tolist())[:52]!r}")

    print()
    print(">>> 三个参数怎么理解：")
    print("    temperature  把概率分布「压尖」还是「摊平」")
    print("                 0    → 永远选最高的（确定、可复现、但死板）")
    print("                 >1   → 摊平，小概率的也被选上（发散、易胡说）")
    print("    top_k        只在概率最高的 k 个里挑，砍掉长尾垃圾")
    print("    top_p        取累计概率达到 p 的最小集合（比 top_k 更自适应）")
    print()
    print(">>> 生产环境怎么设（面试会问）：")
    print("    · 事实问答 / 工具调用参数 / SQL 生成 → temperature=0 或 0.1")
    print("    · 一般对话                            → 0.7 + top_p=0.95")
    print("    · 创意写作                            → 1.0 ~ 1.2")
    print("    · 结构化输出（JSON）                  → 别靠采样，用 Schema 强约束")
    print()
    print("    ⚠️ 注意 t=1.5 那行可能出现乱码 —— 原因见 E5 和 E10。")


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------


def e4_memorized(tok: BPETokenizer, model: GPT) -> None:
    title("E4", "它真的「学会」了吗？—— 看它能不能接上原文")

    for prompt in ["Agent 会先", "一个 Agent 由", "Agent 需要", "工具调用失败时"]:
        ids = np.array([tok.encode(prompt)])
        out = model.generate(ids, max_new_tokens=40, temperature=0.0)
        print(f"输入 {prompt!r}")
        print(f"  → {tok.decode(out[0].tolist())}\n")

    print(">>> 现象：它能相当准确地接上训练语料里的句子。")
    print("    说明模型确实学到了统计规律（哪些字接在哪些字后面）。")
    print()
    print("    但注意：我们只喂了 548 个字符、训练 150 步。")
    print("    它学到的是这段小语料里的搭配，不是「语言」本身。")
    print()
    print("    真实大模型用几千亿 token 训练，学到的是通用语言规律，")
    print("    所以能处理它「没见过」的具体文章 —— 这叫泛化。")
    print("    我们这个小模型几乎没有泛化能力（见 E5）。")


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------


def e5_hallucination(tok: BPETokenizer, model: GPT) -> None:
    title("E5", "没见过的输入会怎样？—— 幻觉的雏形")

    for prompt in ["Agent 通过工具", "今天天气真不错", "请帮我写一个 Java 类", "1+1 等于几"]:
        ids = np.array([tok.encode(prompt)])
        out = model.generate(ids, max_new_tokens=22, temperature=0.7, seed=0)
        print(f"输入 {prompt!r}")
        print(f"  → {tok.decode(out[0].tolist())[:60]!r}\n")

    print(">>> 现象：对完全没见过的输入，它照样自信地输出一串东西 —— 但毫无意义。")
    print()
    print(">>> 这就是幻觉的根源之一：")
    print("    模型的训练目标是「让下一个 token 的概率最大」，不是「说真话」。")
    print("    它**没有「我不知道」这个选项**，除非专门训练过。")
    print("    在它眼里，任何输入都只是 token 序列，都得往下接。")
    print()
    print(">>> 顺便解释 E3 里的乱码字符：")
    print("    这是 byte-level BPE（见 E10），词表里存在「半个汉字」的字节 token。")
    print("    高温度时模型会采样到这些不完整字节，UTF-8 解码就失败了。")
    print()
    print(">>> 生产上对付幻觉的三大手段（面试高频）：")
    print("    1. RAG           把真实资料塞进上下文，让它「看着材料回答」（W2 核心）")
    print("    2. 引用溯源      要求它指出答案来自哪段材料，没依据就拒答")
    print("    3. 输出校验      工具调用 / 结构化输出用 Schema 严格校验")


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------


def e6_causal_mask() -> None:
    title("E6", "因果掩码：为什么它不会偷看后面的字？")

    n = 6
    m = causal_mask_bool(n, n)
    print(f"{n} 个位置的可见性表（✓ = 第 i 个位置能看见第 j 个位置）：\n")
    print("      " + "".join(f" j={j} " for j in range(n)))
    for i in range(n):
        print(f" i={i} " + "".join("  ✓  " if m[i, j] else "  ·  " for j in range(n)))

    print()
    print(">>> 为什么必须这样？")
    print("    生成是「从左往右」的：算第 3 个字时，第 4、5 个字还不存在。")
    print("    如果训练时允许偷看后面，模型会学会「抄答案」，")
    print("    但真正生成时后面没有答案可抄 —— 效果直接崩。")
    print()
    print(">>> 这也带来一个可利用的性质：")
    print("    每个位置只看自己左边，所以**改变后面的字不会影响前面字的输出**。")
    print("    这正是「KV Cache 能复用历史计算结果」的前提（见 E7）。")
    print("    我们的测试 test_attention_causality 验证的就是这一点。")
    print()
    print(">>> 名字来源：论文里叫 masked attention，")
    print("    因为语义是「不能看未来」，业界普遍叫 causal（因果）掩码。")


# ---------------------------------------------------------------------------
# E7
# ---------------------------------------------------------------------------


def e7_kv_cache(tok: BPETokenizer, model: GPT) -> None:
    title("E7", "KV Cache 到底缓存了什么？")

    prompt = "Agent 通过工具调用"
    out = np.array([tok.encode(prompt)])
    cache = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)

    print(f"逐字生成，观察缓存增长。prompt = {prompt!r}\n")
    print(f"{'步':<4}{'本次喂给模型':<16}{'缓存中token数':>14}{'缓存占用(字节)':>16}   预测出的字")
    print("-" * 74)
    for step in range(8):
        cond = out if cache.seq_len == 0 else out[:, -1:]
        logits, _ = model.forward(cond, cache=cache)
        nxt = int(logits[0, -1].argmax())
        fed = f"{cond.shape[1]} 个 token" if step == 0 else "1 个（增量）"
        print(f"{step:<4}{fed:<16}{cache.seq_len:>14}{cache.memory_bytes(4):>16}   "
              f"{show_token(tok, nxt)}")
        out = np.concatenate([out, np.array([[nxt]], dtype=np.int64)], axis=1)

    print()
    print(">>> 关键观察：")
    print("    第 0 步喂整个 prompt，之后每步只喂 1 个新 token。")
    print("    但结果和「每步都重新喂全部历史」完全一样（误差 1e-15，测试里有证明）。")
    print()
    print(">>> 为什么能这样？")
    print("    注意力需要历史所有位置的 K（键）和 V（值）。")
    print("    这些值算过一次就不会变 —— 因为因果掩码保证了历史位置不受新 token 影响。")
    print("    所以可以存起来复用。这就是 Cache。")
    print("    不存的话，每生成一个 token 都要把前面全部重算一遍。")
    print()
    print(">>> 这跟你熟悉的东西一模一样：")
    print("    · MySQL buffer pool 缓存数据页")
    print("    · Redis 缓存计算结果")
    print("    · 前端虚拟 DOM diff 复用节点")
    print("    都是「算过的别重算」。只不过这里缓存的是矩阵，不是对象。")
    print()
    print(">>> 面试为什么爱问它？因为它决定**显存**和**并发能力**：")
    print("    缓存大小 = 2 × 层数 × 批大小 × 序列长度 × 头数 × 每头维度 × 字节数")
    print("    以 32 层 / 4K 上下文 / fp16 算，单个会话就要约 2 GB。")
    print("    → 这就是长上下文和大并发很贵的根因，也是 PagedAttention 要解决的问题。")

    p = np.array([tok.encode("Agent 通过")])
    t0 = time.perf_counter()
    model.generate(p, max_new_tokens=60, temperature=0.0, use_cache=True)
    t_with = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.generate(p, max_new_tokens=60, temperature=0.0, use_cache=False)
    t_without = time.perf_counter() - t0
    print()
    print(f"    实测：生成 60 个 token，带缓存 {t_with * 1000:.0f} ms，"
          f"不带 {t_without * 1000:.0f} ms，加速 {t_without / t_with:.2f}x")
    print("    （玩具模型层数少、序列短，差距不大；32 层 + 数千 token 时可达 10x 以上）")


# ---------------------------------------------------------------------------
# E8
# ---------------------------------------------------------------------------


def e8_position_bug(tok: BPETokenizer, model: GPT) -> None:
    title("E8", "位置信息传错会怎样？（我踩过的真实 bug）")

    # 用固定 token 序列（而不是重新分词），保证"prompt 5 个 token + 1 个增量"清晰可控
    seq = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    prompt_len = 5
    print(f"序列 = {seq[0].tolist()}，prompt 取前 {prompt_len} 个，"
          f"增量喂第 {prompt_len + 1} 个（位置编号应为 {prompt_len}）\n")

    cache = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)
    model.forward(seq[:, :prompt_len], cache=cache)
    lg_ok, _ = model.forward(seq[:, prompt_len : prompt_len + 1], cache=cache)
    lg_full, _ = model.forward(seq[:, : prompt_len + 1])
    print("场景：先用前 5 个 token 预填充缓存，再增量喂第 6 个 token。")
    print("正确与错误做法的唯一区别，是**第 6 个 token 的位置编号**。\n")
    print(f"正确（位置={prompt_len}）    vs 全量前向 → logits 差异 "
          f"{np.abs(lg_ok[0, -1] - lg_full[0, -1]).max():.2e}   ✅")

    # 复现 bug：位置漂移一位
    cache2 = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)
    model.forward(seq[:, :prompt_len], cache=cache2)
    bad_pos = np.arange(cache2.seq_len + 1, cache2.seq_len + 2, dtype=np.float64)
    x = model.wte[seq[:, prompt_len : prompt_len + 1]] + model.wpe[bad_pos.astype(int)][None]
    for i, blk in enumerate(model.blocks):
        x = blk.forward(x, cache2, i, bad_pos)
    lg_bad = model.norm_f.forward(x) @ model.wte.T
    print(f"错误（位置漂移成 {int(bad_pos[0])}）vs 全量前向 → logits 差异 "
          f"{np.abs(lg_bad[0, -1] - lg_full[0, -1]).max():.2e}   ❌")

    print()
    print(">>> 为什么这个 bug 特别难查：")
    print("    · 差异量级很暧昧：短序列/单层时约 1e-2，长了能到 6e-1。")
    print("      小的时候看起来像「浮点误差」，很容易放过。")
    print("    · 单层模型测试完全正常（1e-16），只有多层才暴露")
    print("    · 表现是「生成到中间开始跑偏」，而不是直接崩溃报错")
    print()
    print(">>> 根因：位置编号必须**整趟前向只算一次**，然后传给所有层。")
    print("    如果每层内部各读一次 cache.seq_len，第 0 层追加缓存后")
    print("    seq_len 就变了，第 1 层会读到偏移一位的位置。")
    print()
    print(">>> 你熟悉的同类错误：")
    print("    · Java 遍历集合时修改集合 → ConcurrentModificationException")
    print("    · 分页查询先查总数再查列表，中间有写入 → 数据错位")
    print("    · 分布式多节点各自读共享 offset → 重复消费")
    print("    本质相同：**在循环内部读取会被循环修改的共享状态**。")
    print()
    print(">>> 为什么位置错了影响这么大？RoPE 会把同一向量按位置旋转：")
    v = np.ones((1, 1, 1, 4))
    for pos in (0, 1, 3, 4):
        print(f"    位置 {pos}: {np.round(apply_rope(v, np.array([pos], dtype=np.float64))[0, 0, 0], 4)}")
    print("    → 同一向量在不同位置被旋转到不同方向，")
    print("      注意力靠这个方向差判断「谁在谁前面、隔多远」。")


# ---------------------------------------------------------------------------
# E9
# ---------------------------------------------------------------------------


def e9_scale(model: GPT, tok: BPETokenizer) -> None:
    title("E9", "这个玩具和 GPT-3 差多远？—— 诚实认识规模")

    ours_p = model.n_params()
    ours_d = len(tok.encode(CORPUS))
    rows = [
        ("我们的玩具 mini-GPT", ours_p, ours_d),
        ("GPT-2  (2019)", 1_500_000_000, 40_000_000_000),
        ("GPT-3  (2020)", 175_000_000_000, 300_000_000_000),
        ("Llama-3-70B (2024)", 70_000_000_000, 15_000_000_000_000),
    ]
    print(f"{'模型':<24}{'参数量':>20}{'训练 token 数':>24}")
    print("-" * 70)
    for name, p, d in rows:
        print(f"{name:<24}{p:>20,}{d:>24,}")

    print()
    print(f">>> 我们 vs GPT-3：参数差 {175_000_000_000 / ours_p:,.0f} 倍，"
          f"数据差 {300_000_000_000 / ours_d:,.0f} 倍")
    print()
    print(">>> 那为什么还要手写这个玩具？")
    print("    因为我们学的不是「怎么造大模型」—— 那是算法岗，要几千张 GPU。")
    print("    我们学的是**机制**：token 怎么切、注意力怎么算、")
    print("    缓存为什么省时间、位置为什么必须编码、梯度怎么传。")
    print()
    print("    这些机制在 1750 亿参数的模型里和在我们的 85 万参数模型里")
    print("    **完全一样**，只是矩阵更大。面试官问「KV Cache 原理」，")
    print("    不是问你能训多大的模型。")
    print()
    print(">>> 这也是你简历定位的关键：")
    print("    Agent 开发工程师要的是**理解模型的行为边界** ——")
    print("    它为什么会胡说、上下文为什么有限、成本怎么算、")
    print("    什么时候该用 RAG、什么时候该微调。这些在玩具模型上就能建立直觉。")


# ---------------------------------------------------------------------------
# E10
# ---------------------------------------------------------------------------


def e10_tokenization() -> None:
    title("E10", "BPE 到底是什么？—— 为什么中文比英文贵")

    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)

    print("同一个分词器，看不同文本被切成什么样：\n")
    print(f"{'文本':<26}{'字符':>6}{'字节':>6}{'token':>7}   实际切分")
    print("-" * 74)
    for s in ["人工智能", "Agent", "Agent 通过工具", "Hello, world!",
              "tokenization", "1234567890", "🚀🎯"]:
        ids = tok.encode(s)
        pieces = [show_token(tok, i) for i in ids]
        shown = "|".join(pieces)
        if len(shown) > 26:
            shown = shown[:26] + "…"
        print(f"{s!r:<26}{len(s):>6}{len(s.encode('utf-8')):>6}{len(ids):>7}   {shown}")

    print()
    print(">>> 三个关键认知：")
    print("    1. 中文一个字 = 3 个 UTF-8 字节。训练不足时会被切成多个 token，")
    print("       所以**中文的 token 消耗比英文高**，成本也更高。")
    print("    2. 高频词（如 'tokenization'）训练后会被合并成少数 token，")
    print("       这就是 BPE 的「贪心合并」效果。")
    print("    3. 因为它基于字节，**任何字符都能编码，永远不会出现 <UNK>**。")
    print("       代价是可能出现「半个汉字」的 token —— 这就是高温度乱码的来源。")
    print()
    print(">>> 为什么你必须关心 token 数？")
    print("    · 计费按 token 算（不是按字、也不是按词）")
    print("    · 上下文窗口按 token 算（128K 是 128K 个 token）")
    print("    · RAG 塞的资料越多，token 越贵，而且会稀释注意力")
    print("    → W2 学分块策略时，本质就是在权衡 token 预算。")
    print()
    raw = "tokenization"
    print(f">>> 训练前后对比 {raw!r}：")
    print(f"    未训练（纯字节）: {len(raw.encode('utf-8'))} 个 token")
    print(f"    训练后          : {len(tok.encode(raw))} 个 token")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    print(LINE)
    print("W1 引导实验：从现象建立直觉")
    print("每个实验先让你看见「发生了什么」，再解释「为什么」。")
    print(LINE)

    tok, model, _x, _y, loss0, _loss = build_model()

    e1_next_token(tok, model)
    e2_training(tok, loss0)
    e3_sampling(tok, model)
    e4_memorized(tok, model)
    e5_hallucination(tok, model)
    e6_causal_mask()
    e7_kv_cache(tok, model)
    e8_position_bug(tok, model)
    e9_scale(model, tok)
    e10_tokenization()

    print(f"\n{LINE}")
    print("做完这 10 个实验，你应该能回答这些问题了：")
    print(LINE)
    questions = [
        "大模型生成一句话的完整过程是什么？",
        "loss 是什么？为什么初始值是 ln(词表大小)？",
        "temperature / top_k / top_p 分别控制什么？生产环境怎么设？",
        "为什么模型会有幻觉？三种缓解手段是什么？",
        "因果掩码为什么必要？它带来什么可利用的性质？",
        "KV Cache 缓存了什么？显存怎么估算？为什么能加速？",
        "位置编码传错会导致什么现象？为什么难查？",
        "为什么中文比英文的 token 成本高？",
    ]
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print()
    print("下一步：带着这些直觉去读源码，按这个顺序：")
    print("  1. bpe.py       看 token 怎么切出来      （对应 E10）")
    print("  2. attention.py 看注意力与缓存怎么算      （对应 E6 / E7）")
    print("  3. mini_gpt.py  看整个模型与训练循环      （对应 E1 ~ E5）")


if __name__ == "__main__":
    main()
