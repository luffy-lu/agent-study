"""手写注意力机制与 KV Cache —— 纯 NumPy，不用任何深度学习框架。

为什么不用 PyTorch？
    框架会把因果掩码、多头 reshape、KV Cache 的拼接都藏起来。W1 的目标是让这些
    东西在你手里"透明"：面试时能白板画出 Q/K/V 的形状变化、能写出 KV Cache 的
    显存公式，才算真的懂。

本文件包含（建议按顺序读）：
    1. softmax / 因果掩码
    2. scaled_dot_product_attention —— 注意力的最小内核
    3. apply_rope —— RoPE 旋转位置编码
    4. MultiHeadAttention.forward —— 带 KV Cache 的多头注意力
    5. KVCache —— 缓存结构与显存公式

运行：
    uv run python -m w1_llm.attention
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


# ---------------------------------------------------------------------------
# 1. softmax 与掩码
# ---------------------------------------------------------------------------


def softmax(x: Array, axis: int = -1) -> Array:
    """数值稳定的 softmax：先减去最大值，防止 exp 溢出。

    为什么要减 max？
        exp(1000) = inf；exp(1000-1000)=1。结果数学等价，数值安全。
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(n: int) -> NDArray[np.bool_]:
    """下三角布尔掩码，True 表示"允许看见"。

    例 n=4：
        [[T,F,F,F],
         [T,T,F,F],
         [T,T,T,F],
         [T,T,T,T]]
    第 i 行只能看 j <= i，因为生成时未来 token 还不存在。
    """
    return np.tril(np.ones((n, n), dtype=bool))


# ---------------------------------------------------------------------------
# 2. 注意力的最小内核
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Array, Array]:
    """Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask) V

    形状：
        q: (..., T_q, d_k)
        k: (..., T_k, d_k)
        v: (..., T_k, d_v)
        返回: (输出 (..., T_q, d_v), 注意力权重 (..., T_q, T_k))

    **为什么除以 √d_k（面试第一问）**：
        假设 q、k 各分量独立、均值 0 方差 1，则点积 q·k 的方差是 d_k。
        d_k = 64 时点积标准差 ≈ 8，softmax 输入跨度大会导致输出接近 one-hot，
        梯度趋近 0（饱和）。除以 √d_k 把方差拉回 1，保持梯度健康。
    """
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)

    if mask is not None:
        # 用极小值而非 0：softmax 后 ≈ 0，且不会把概率质量分摊给被屏蔽位置
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    return weights @ v, weights


# ---------------------------------------------------------------------------
# 3. RoPE（旋转位置编码）
# ---------------------------------------------------------------------------


def _rope_tables(dim: int, positions: Array, base: float = 10000.0) -> tuple[Array, Array]:
    """预计算每个位置的 cos/sin 表。dim 必须是偶数。"""
    if dim % 2 != 0:
        raise ValueError(f"RoPE 需要偶数维度，收到 {dim}")
    half = dim // 2
    # 不同维度对用不同频率：低频维度编码长距离，高频维度编码近距离
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    angles = positions[:, None] * inv_freq[None, :]  # (T, half)
    return np.cos(angles), np.sin(angles)


def _rotate_half(x: Array) -> Array:
    """把最后一维前后两半交换：(x1, x2) -> (-x2, x1)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: Array, positions: Array, base: float = 10000.0) -> Array:
    """对 q 或 k 施加 RoPE。

    形状: x (B, H, T, d_head)，positions (T,) 为该序列的绝对位置。

    为什么位置编码要"旋转"而不是"相加"？
        相加会改变向量本身的内容（污染语义）；旋转只改变方向、保持模长，
        且 QᵀK 的内积只依赖 (i-j) 相对距离 —— 这是 RoPE 外推能力的来源。
    """
    dim = x.shape[-1]
    cos, sin = _rope_tables(dim, positions, base)  # (T, half)
    # 拼成与最后一维等长的 (T, dim)
    cos = np.concatenate([cos, cos], axis=-1)[None, None, :, :]
    sin = np.concatenate([sin, sin], axis=-1)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# 4. KV Cache
# ---------------------------------------------------------------------------


class KVCache:
    """自回归解码时的键值缓存。

    **核心问题：为什么需要它？**
        生成第 t 个 token 时，注意力要算 q_t @ Kᵀ，其中 K 是前 t-1 个 token 的键。
        而这些 K 在第 t-1 步已经算过了。没有缓存 → 每步重算全部历史 → O(T²) 次投影。
        有缓存 → 每步只投影 1 个新 token → 总投影量 O(T)。

    **显存公式（面试常问）**：
        bytes = 2 (K和V) × L (层数) × B (batch) × T (序列长度)
                × H (头数) × d_head (每头维度) × bytes_per_element
        例：L=32, B=1, T=4096, H=32, d_head=128, fp16(2字节)
            = 2×32×1×4096×32×128×2 = 2,147,483,648 B ≈ 2 GB
        注意它与 T 线性增长，与 batch 线性增长 —— 这就是长上下文/大并发爆显存的根源，
        也是 PagedAttention 要解决的问题（用分页管理避免预留连续显存、减少碎片）。
    """

    def __init__(self, n_layers: int, batch: int, n_heads: int, d_head: int) -> None:
        self.n_layers = n_layers
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.k: list[Array | None] = [None] * n_layers
        self.v: list[Array | None] = [None] * n_layers

    @property
    def seq_len(self) -> int:
        """已缓存的 token 数（即下一个 token 的起始位置）。"""
        return 0 if self.k[0] is None else int(self.k[0].shape[2])

    def append(self, layer: int, k_new: Array, v_new: Array) -> tuple[Array, Array]:
        """追加本步的 K/V，返回完整（历史 + 新）的 K/V。

        k_new/v_new: (B, H, T_new, d_head)，增量解码时 T_new=1
        """
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            assert self.k[layer] is not None
            self.k[layer] = np.concatenate([self.k[layer], k_new], axis=2)  # type: ignore[arg-type]
            self.v[layer] = np.concatenate([self.v[layer], v_new], axis=2)  # type: ignore[arg-type]
        return self.k[layer], self.v[layer]  # type: ignore[return-value]

    def memory_bytes(self, dtype_bytes: int = 2) -> int:
        """当前缓存占用字节数。"""
        if self.k[0] is None:
            return 0
        elems = sum(
            int(c.shape[2] * c.shape[1] * c.shape[0] * c.shape[3])
            for layer in range(self.n_layers)
            for c in (self.k[layer], self.v[layer])
            if c is not None
        )
        return elems * dtype_bytes

    def reset(self) -> None:
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers


# ---------------------------------------------------------------------------
# 5. 多头注意力（带 KV Cache）
# ---------------------------------------------------------------------------


class MultiHeadAttention:
    """因果多头自注意力。

    形状流转（B=batch, T=序列长, C=模型维度, H=头数, d=C/H）：
        x:      (B, T, C)
        qkv:    (B, T, 3C)          —— 一次矩阵乘出 Q/K/V
        拆头:    (B, H, T, d)
        注意力:  (B, H, T, d)
        合并:    (B, T, C)

    **为什么要多头（面试常问）**：
        单头只能在一种"关系模式"下做加权平均；多头让不同子空间分别关注
        语法、指代、位置等不同模式。多头不增加参数量（总维度 C 被拆成 H 份）。
    """

    def __init__(self, n_embd: int, n_head: int, rng: np.random.Generator) -> None:
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_head = n_embd // n_head
        scale = 1.0 / np.sqrt(n_embd)
        # GPT-2 风格：合并的 QKV 投影 + 输出投影
        self.w_qkv = rng.normal(0, scale, (n_embd, 3 * n_embd))
        self.b_qkv = np.zeros(3 * n_embd)
        self.w_proj = rng.normal(0, scale, (n_embd, n_embd))
        self.b_proj = np.zeros(n_embd)

    def forward(
        self,
        x: Array,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        use_rope: bool = True,
    ) -> Array:
        B, T, C = x.shape
        H, d = self.n_head, self.d_head

        qkv = x @ self.w_qkv + self.b_qkv                      # (B, T, 3C)
        q, k, v = np.split(qkv, 3, axis=-1)                    # each (B, T, C)

        # 拆头：(B, T, C) -> (B, T, H, d) -> (B, H, T, d)
        def split_heads(t: Array) -> Array:
            return np.transpose(t.reshape(B, T, H, d), (0, 2, 1, 3))

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if use_rope:
            # 有缓存时，新 token 的位置从 cache.seq_len 开始
            start = cache.seq_len if cache is not None else 0
            positions = np.arange(start, start + T, dtype=np.float64)
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        if cache is not None:
            k, v = cache.append(layer_idx, k, v)               # 历史 + 新

        T_k = k.shape[2]
        # 因果掩码：只有当 q 的长度等于 k 的长度时才是方阵；
        # 增量解码（T=1, T_k=start+1）时，单个 query 可以看见全部历史
        mask = None
        if T > 1:
            mask = causal_mask(T) if T == T_k else causal_mask(T_k)[-T:, :]

        out, _ = scaled_dot_product_attention(q, k, v, mask)   # (B, H, T, d)

        # 合并头：(B, H, T, d) -> (B, T, H, d) -> (B, T, C)
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)
        return out @ self.w_proj + self.b_proj


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def _demo() -> None:
    rng = np.random.default_rng(0)

    print("=== 1. softmax 数值稳定性 ===")
    big = np.array([1000.0, 1001.0, 1002.0])
    print("直接 exp 会溢出；稳定版结果:", softmax(big))

    print("\n=== 2. 因果掩码 ===")
    print(causal_mask(4).astype(int))

    print("\n=== 3. 缩放为什么必要 ===")
    d = 64
    q = rng.normal(0, 1, (1, d))
    k = rng.normal(0, 1, (1000, d))
    raw = (q @ k.T)[0]
    print(f"d_k={d}: 未缩放点积 标准差={raw.std():.2f}（理论 √d_k={np.sqrt(d):.2f}）")
    print(f"         缩放后   标准差={(raw / np.sqrt(d)).std():.2f}")

    print("\n=== 4. 因果性验证（改未来 token 不应影响过去输出）===")
    mha = MultiHeadAttention(n_embd=32, n_head=4, rng=rng)
    x1 = rng.normal(0, 1, (1, 6, 32))
    x2 = x1.copy()
    x2[:, 4:, :] += 10.0  # 只改第 4 个位置之后
    o1, o2 = mha.forward(x1), mha.forward(x2)
    same_past = np.allclose(o1[:, :4], o2[:, :4], atol=1e-8)
    diff_future = not np.allclose(o1[:, 4:], o2[:, 4:], atol=1e-6)
    print("过去位置输出不变:", same_past, "| 未来位置输出改变:", diff_future)
    assert same_past and diff_future, "因果掩码失效！"

    print("\n=== 5. KV Cache 正确性（缓存解码 vs 全量前向 必须一致）===")
    x = rng.normal(0, 1, (1, 8, 32))
    full = mha.forward(x)

    cache = KVCache(n_layers=1, batch=1, n_heads=4, d_head=8)
    steps = [mha.forward(x[:, i : i + 1, :], cache=cache, layer_idx=0) for i in range(8)]
    incremental = np.concatenate(steps, axis=1)
    max_diff = float(np.abs(full - incremental).max())
    print(f"全量前向 vs 增量缓存 最大误差 = {max_diff:.2e}")
    assert max_diff < 1e-8, "KV Cache 实现有误！"

    print("\n=== 6. KV Cache 显存公式 ===")
    for L, T, H, d in [(12, 512, 8, 32), (32, 4096, 32, 128)]:
        c = KVCache(n_layers=L, batch=1, n_heads=H, d_head=d)
        c.k = [np.zeros((1, H, T, d)) for _ in range(L)]
        c.v = [np.zeros((1, H, T, d)) for _ in range(L)]
        formula = 2 * L * 1 * T * H * d * 2
        print(f"L={L:>2} T={T:>4} H={H:>2} d={d:>3} -> "
              f"{c.memory_bytes() / 1024**2:>8.1f} MB（公式 {formula / 1024**2:.1f} MB）")


if __name__ == "__main__":
    _demo()
