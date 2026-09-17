"""W1 测试：验证 BPE / Attention / KV Cache / 采样 / 反向传播。

运行：
    uv run pytest -v
快速跑（跳过较慢的梯度校验）：
    uv run pytest -v -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from w1_llm.attention import (
    KVCache,
    MultiHeadAttention,
    apply_rope,
    causal_mask,
    scaled_dot_product_attention,
    softmax,
)
from w1_llm.bpe import BPETokenizer, get_pair_counts, merge_pair
from w1_llm.mini_gpt import GPT, Adam, check_gradients, sample_token

CORPUS = "人工智能正在改变世界。Agent 会调用工具并推理。The quick brown fox. " * 6


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------


def test_merge_pair_basic() -> None:
    assert merge_pair([1, 2, 1, 2], (1, 2), 9) == [9, 9]
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1], "重叠时应左边优先、不重叠"
    assert merge_pair([5, 6, 7], (1, 2), 9) == [5, 6, 7], "无匹配应原样返回"


def test_get_pair_counts() -> None:
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert get_pair_counts([7]) == {}


def test_bpe_roundtrip() -> None:
    """byte-level 的核心保证：任何文本都能无损往返，不存在 <UNK>。"""
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    for s in [CORPUS, "emoji 🚀🎯", "Mixed 中英 123", "\n\t 空格  保留", ""]:
        assert tok.decode(tok.encode(s)) == s, f"round-trip 失败: {s!r}"


def test_bpe_vocab_size_respected() -> None:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=350)
    assert tok.next_id <= 350
    assert len(tok.merges) == tok.next_id - 256


def test_bpe_compression() -> None:
    """训练后 token 数应显著少于字节数，否则说明没学到任何合并。"""
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    n_tok = len(tok.encode(CORPUS))
    n_byte = len(CORPUS.encode("utf-8"))
    assert n_tok < n_byte, f"无压缩效果: {n_tok} tokens vs {n_byte} bytes"


def test_bpe_save_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=320)
    p = tmp_path / "bpe.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(CORPUS) == tok.encode(CORPUS), "save/load 后编码必须一致"
    assert tok2.decode(tok2.encode(CORPUS)) == CORPUS


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def test_softmax_properties() -> None:
    x = np.array([1000.0, 1001.0, 1002.0])
    p = softmax(x)
    assert np.all(np.isfinite(p)), "必须数值稳定，不能溢出"
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]


def test_causal_mask() -> None:
    m = causal_mask(4)
    assert m[0].tolist() == [True, False, False, False]
    assert m[3].all()
    assert np.tril(m).sum() == 10


def test_attention_causality() -> None:
    """改动未来 token 不得影响过去的输出 —— 因果掩码的根本约束。"""
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(32, 4, rng)
    x1 = rng.normal(0, 1, (1, 6, 32))
    x2 = x1.copy()
    x2[:, 4:, :] += 10.0
    o1, o2 = mha.forward(x1), mha.forward(x2)
    assert np.allclose(o1[:, :4], o2[:, :4], atol=1e-10), "过去位置被未来的改动影响了"
    assert not np.allclose(o1[:, 4:], o2[:, 4:], atol=1e-6), "未来位置应受影响"


def test_attention_scores_scale() -> None:
    """除以 √d_k 的原因：不缩放时点积方差会随 d_k 线性增长。"""
    rng = np.random.default_rng(0)
    d = 256
    q = rng.normal(0, 1, (1, d))
    k = rng.normal(0, 1, (4000, d))
    raw = (q @ k.T)[0]
    assert 12 < raw.std() < 20, f"未缩放标准差应≈√d={np.sqrt(d):.0f}，实测 {raw.std():.2f}"
    assert 0.8 < (raw / np.sqrt(d)).std() < 1.3, "缩放后标准差应回到 ≈1"


def test_sdpa_matches_manual() -> None:
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(0, 1, (1, 2, 5, 8)) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    manual = softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(8), -1)
    masked = np.where(causal_mask(5), manual, 0.0)
    masked = masked / masked.sum(-1, keepdims=True)
    assert np.allclose(w, masked, atol=1e-12)
    assert out.shape == (1, 2, 5, 8)


def test_rope_is_norm_preserving() -> None:
    """RoPE 是旋转：只改方向不改模长（这是它有外推能力的原因之一）。"""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (1, 2, 6, 8))
    y = apply_rope(x, np.arange(6, dtype=np.float64))
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), atol=1e-10)


def test_rope_relative_position() -> None:
    """QᵀK 只依赖相对距离：平移两个序列的位置，内积不变。"""
    rng = np.random.default_rng(3)
    q = rng.normal(0, 1, (1, 1, 3, 8))
    k = rng.normal(0, 1, (1, 1, 3, 8))
    q1 = apply_rope(q, np.arange(3, dtype=np.float64))
    k1 = apply_rope(k, np.arange(3, dtype=np.float64))
    q2 = apply_rope(q, np.arange(3, dtype=np.float64) + 10)
    k2 = apply_rope(k, np.arange(3, dtype=np.float64) + 10)
    assert np.allclose(q1 @ np.swapaxes(k1, -1, -2), q2 @ np.swapaxes(k2, -1, -2), atol=1e-9)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 2, 5, 8])
def test_kv_cache_matches_full_forward(T: int) -> None:
    """带缓存逐步解码的结果必须与一次性全量前向逐元素一致。"""
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(32, 4, rng)
    x = rng.normal(0, 1, (1, T, 32))
    full = mha.forward(x)

    cache = KVCache(n_layers=1, batch=1, n_heads=4, d_head=8)
    steps = [mha.forward(x[:, i : i + 1, :], cache=cache, layer_idx=0) for i in range(T)]
    inc = np.concatenate(steps, axis=1)
    assert np.allclose(full, inc, atol=1e-10), f"T={T} 时缓存解码与全量前向不一致"


def test_kv_cache_memory_formula() -> None:
    """显存 = 2 × L × B × T × H × d_head × dtype_bytes。"""
    L, B, T, H, d = 4, 2, 64, 8, 16
    c = KVCache(L, B, H, d)
    c.k = [np.zeros((B, H, T, d)) for _ in range(L)]
    c.v = [np.zeros((B, H, T, d)) for _ in range(L)]
    expected = 2 * L * B * T * H * d * 2  # fp16
    assert c.memory_bytes(2) == expected
    assert c.seq_len == T
    c.reset()
    assert c.seq_len == 0 and c.memory_bytes() == 0


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


def test_sample_temperature_zero_is_greedy() -> None:
    logits = np.array([0.1, 5.0, 0.2, 3.0])
    assert sample_token(logits, temperature=0.0) == 1
    for _ in range(20):
        assert sample_token(logits, temperature=0.0) == 1


def test_sample_top_k_restricts_support() -> None:
    logits = np.array([0.0, 10.0, 9.0, -10.0])
    seen = {sample_token(logits, temperature=1.0, top_k=2, rng=np.random.default_rng(i)) for i in range(50)}
    assert seen <= {1, 2}, f"top_k=2 只允许 {1,2}，实际出现 {seen}"


def test_sample_repetition_penalty_lowers_logit() -> None:
    """对已出现 token 加惩罚后，它被选中的比例应显著下降。"""
    logits = np.array([5.0, 1.0, 1.0])
    n = 400
    without = sum(
        sample_token(logits, temperature=1.0, rng=np.random.default_rng(i)) == 0
        for i in range(n)
    )
    with_pen = sum(
        sample_token(logits, temperature=1.0, repetition_penalty=3.0,
                     recent_tokens=[0], rng=np.random.default_rng(i)) == 0
        for i in range(n)
    )
    assert without - with_pen >= 100, f"惩罚前 {without}/{n}，惩罚后 {with_pen}/{n}，下降不足"


def test_sample_seed_reproducible() -> None:
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    a = [sample_token(logits, temperature=1.0, rng=np.random.default_rng(7)) for _ in range(5)]
    b = [sample_token(logits, temperature=1.0, rng=np.random.default_rng(7)) for _ in range(5)]
    assert a == b


# ---------------------------------------------------------------------------
# 模型与梯度
# ---------------------------------------------------------------------------


def _tiny_model() -> GPT:
    # block_size 需容纳 prompt(3) + max_new_tokens(8) = 11，留余量取 32
    return GPT(23, n_layer=2, n_head=2, n_embd=16, block_size=32, seed=0)


def test_initial_loss_near_log_vocab() -> None:
    """随机初始化时 loss 应接近 ln(V) —— 这是判断模型/损失实现正确性的第一道关。"""
    model = _tiny_model()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 23, (2, 6))
    _, loss = model.forward(idx, idx)
    assert loss is not None
    assert abs(loss - np.log(23)) < 0.6, f"初始 loss={loss:.3f}，期望≈{np.log(23):.3f}"


@pytest.mark.slow
def test_gradients_match_numerical() -> None:
    """数值梯度校验：手写反向传播的正确性证明。"""
    assert check_gradients(vocab_size=13, n_layer=2, n_embd=16, n_head=2,
                           block_size=5, n_checks=8), "反向传播与数值梯度不一致"


def test_training_reduces_loss() -> None:
    """在玩具语料上训练几十步，loss 必须明显下降。"""
    model = _tiny_model()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 23, (1, 6))
    y = rng.integers(0, 23, (1, 6))
    params = model.params()
    opt = Adam(lr=3e-3)

    _, loss0 = model.forward(x, y)
    for _ in range(60):
        _, loss = model.forward(x, y)
        opt.step(params, model.backward())
    assert loss0 is not None and loss is not None
    assert loss < loss0 * 0.6, f"loss 未下降: {loss0:.3f} -> {loss:.3f}"


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_shapes_and_cache_equivalence(use_cache: bool) -> None:
    """KV Cache 版与全量版的贪心生成必须给出完全相同的 token 序列。"""
    model = _tiny_model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=use_cache)
    assert out.shape == (1, 9)
    assert out[0, :3].tolist() == [1, 2, 3], "prompt 必须原样保留"

    other = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=not use_cache)
    assert out.tolist() == other.tolist(), "缓存与否不应改变贪心生成结果"


def test_multilayer_cache_positions_not_drifted() -> None:
    """回归测试：多层模型下，增量前向的 logits 必须与全量前向一致。

    这里锁定的是一个非常隐蔽的 bug：如果每层各自读 `cache.seq_len` 来决定 RoPE 位置，
    那么第 0 层追加 KV 后 seq_len 就变了，第 1 层会拿到偏移一位的位置，
    在 T=1 增量时表现为"位置错位"，logits 相差 ~1e-2 量级。
    正确做法是整趟前向只算一次 positions 再传给所有层。
    """
    model = _tiny_model()
    seq = np.array([[4, 5, 6, 7, 8]], dtype=np.int64)
    prompt_len = 4

    cache = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)
    model.forward(seq[:, :prompt_len], cache=cache)          # 预填充 prompt
    incremental, _ = model.forward(seq[:, prompt_len:], cache=cache)
    full, _ = model.forward(seq)

    diff = float(np.abs(incremental[0, -1] - full[0, -1]).max())
    assert diff < 1e-10, f"多层增量位置错位：logits 差 {diff:.3e}"


def test_cache_does_not_mutate_weights() -> None:
    """带缓存前向不得改动任何参数（防止把状态误写进权重）。"""
    model = _tiny_model()
    before = {k: v.copy() for k, v in model.params().items()}
    cache = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)
    seq = np.array([[1, 2, 3, 4]], dtype=np.int64)
    for i in range(seq.shape[1]):
        model.forward(seq[:, i : i + 1], cache=cache)
    assert cache.seq_len == seq.shape[1]
    for k, v in model.params().items():
        assert np.array_equal(before[k], v), f"参数 {k} 被缓存前向改动了"


def test_param_count_and_weight_tying() -> None:
    model = _tiny_model()
    assert model.n_params() > 0
    assert model.wte.shape == (23, 16)
    # weight tying：输出层复用 wte，故 logits 的最后一维必须是 vocab_size
    logits, _ = model.forward(np.array([[1, 2]], dtype=np.int64))
    assert logits.shape == (1, 2, 23)
