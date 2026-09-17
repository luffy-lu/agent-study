"""mini-GPT —— 用纯 NumPy 拼一个**真正可训练**的 decoder-only Transformer。

设计取舍（重要，先读这段）：
    本文件手写完整反向传播，因此每一块的梯度都是透明的。为保证正确性，文件末尾
    提供了**数值梯度校验**（finite difference），它会把每个解析梯度和数值梯度对比。
    这是 W1 最有价值的一课：**当你能证明自己的梯度是对的，你才真的懂了这个模型。**

结构（Pre-Norm，与 LLaMA 系一致）：
    Embedding(wte + wpe) -> N × [ RMSNorm -> MHA(RoPE, KV Cache) -> +残差
                                  RMSNorm -> SwiGLU           -> +残差 ]
    -> RMSNorm -> LM Head（与 wte 权重绑定 / weight tying）

运行（约 1–2 分钟）：
    uv run python -m w1_llm.mini_gpt
校验梯度（强烈建议跑）：
    uv run python -m w1_llm.mini_gpt --check-grad
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numpy.typing import NDArray

from .attention import KVCache, _rope_tables, _rotate_half, apply_rope, softmax  # noqa: F401
from .bpe import BPETokenizer

Array = NDArray[np.floating]
INT = np.int64


# ---------------------------------------------------------------------------
# 前向/反向辅助函数
# ---------------------------------------------------------------------------


def rope_backward(dout: Array, positions: Array, base: float = 10000.0) -> Array:
    """RoPE 的反向：因为该变换是正交的，逆变换 = 用 -sin 再旋转一次。

    y = x·cos + rot(x)·sin
    dx = dout·cos - rot(dout)·sin      （rot 是反对称矩阵，转置即取负）
    """
    dim = dout.shape[-1]
    cos, sin = _rope_tables(dim, positions, base)
    cos = np.concatenate([cos, cos], axis=-1)[None, None, :, :]
    sin = np.concatenate([sin, sin], axis=-1)[None, None, :, :]
    return dout * cos - _rotate_half(dout) * sin


def sdpa_backward(
    dout: Array,
    q: Array,
    k: Array,
    v: Array,
    weights: Array,
    mask: NDArray[np.bool_] | None,
) -> tuple[Array, Array, Array]:
    """scaled_dot_product_attention 的反向。"""
    d_k = q.shape[-1]
    dv = dout @ np.swapaxes(weights, -1, -2)                     # (…,T_q,T_k)
    dweights = dout @ np.swapaxes(v, -1, -2)                     # (…,T_q,T_k)

    # softmax 反向：dscore = w * (dw - sum(dw*w, axis=-1, keepdim))
    dscore = weights * (dweights - np.sum(dweights * weights, axis=-1, keepdims=True))
    if mask is not None:
        dscore = np.where(mask, dscore, 0.0)                     # 被屏蔽位置不回流梯度

    dscore = dscore / np.sqrt(d_k)
    dq = dscore @ k
    dk = np.swapaxes(dscore, -1, -2) @ q
    return dq, dk, dv


def causal_mask_bool(t_q: int, t_k: int) -> NDArray[np.bool_]:
    """(T_q, T_k) 因果掩码：第 i 个 query 可看 j <= i + (T_k - T_q)。

    完整序列（T_q == T_k）时即标准下三角；增量解码（T_q=1）时全为 True。
    """
    offset = t_k - t_q
    i = np.arange(t_q)[:, None]
    j = np.arange(t_k)[None, :]
    return j <= (i + offset)


def softmax_backward(dout: Array, out: Array) -> Array:
    return out * (dout - np.sum(dout * out, axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm:
    """RMSNorm：只用均方根缩放，不减均值、不加偏置（LLaMA 采用）。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.dim = dim
        self.g = np.ones(dim)
        self.dg = np.zeros(dim)
        self._xhat: Array | None = None
        self._rms: Array | None = None

    def forward(self, x: Array) -> Array:
        self._rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self._xhat = x / self._rms
        return self._xhat * self.g

    def backward(self, dout: Array) -> Array:
        assert self._xhat is not None and self._rms is not None
        xhat, rms = self._xhat, self._rms
        axes = tuple(range(dout.ndim - 1))
        self.dg = np.sum(dout * xhat, axis=axes)
        d_xhat = dout * self.g
        # xhat = x/rms(x)，且 rms 依赖 x => 需要减掉投影到 xhat 方向的分量
        proj = np.mean(d_xhat * xhat, axis=-1, keepdims=True)
        return (d_xhat - xhat * proj) / rms


# ---------------------------------------------------------------------------
# FFN (SwiGLU)
# ---------------------------------------------------------------------------


class SwiGLU:
    """down(silu(gate(x)) * up(x))

    门控机制让网络能按输入动态筛选信息；隐藏维取 ~8/3·d 以对齐标准 4× FFN 的参数量。
    """

    def __init__(self, n_embd: int, rng: np.random.Generator) -> None:
        hidden = (int(8 * n_embd / 3) + 7) // 8 * 8
        self.hidden = hidden
        self.w_gate = rng.normal(0, 1 / np.sqrt(n_embd), (n_embd, hidden))
        self.w_up = rng.normal(0, 1 / np.sqrt(n_embd), (n_embd, hidden))
        self.w_down = rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_embd))
        self.grads: dict[str, Array] = {}

    @staticmethod
    def _silu(x: Array) -> Array:
        return x / (1.0 + np.exp(-x))

    @staticmethod
    def _silu_grad(x: Array) -> Array:
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 + x * (1.0 - s))

    def forward(self, x: Array) -> Array:
        self._x = x
        self._pre = x @ self.w_gate
        self._up = x @ self.w_up
        self._h = self._silu(self._pre) * self._up
        return self._h @ self.w_down

    def backward(self, dout: Array) -> Array:
        flat = lambda t: t.reshape(-1, t.shape[-1])  # noqa: E731
        xf, pre, up, hf, df = flat(self._x), flat(self._pre), flat(self._up), flat(self._h), flat(dout)

        self.grads = {
            "w_down": hf.T @ df,
            "w_gate": xf.T @ (df @ self.w_down.T * self._silu_grad(pre) * up),
            "w_up": xf.T @ (df @ self.w_down.T * self._silu(pre)),
        }
        d_pre = df @ self.w_down.T * self._silu_grad(pre) * up
        d_up = df @ self.w_down.T * self._silu(pre)
        return (d_pre @ self.w_gate.T + d_up @ self.w_up.T).reshape(dout.shape)


# ---------------------------------------------------------------------------
# 注意力（补上反向，复用 attention.py 的前向思想）
# ---------------------------------------------------------------------------


class Attention:
    """带 RoPE 与 KV Cache 的因果多头注意力，含完整反向。"""

    def __init__(self, n_embd: int, n_head: int, rng: np.random.Generator) -> None:
        assert n_embd % n_head == 0 and (n_embd // n_head) % 2 == 0, "头维度需为偶数以支持 RoPE"
        self.n_embd, self.n_head = n_embd, n_head
        self.d_head = n_embd // n_head
        self.w_qkv = rng.normal(0, 1 / np.sqrt(n_embd), (n_embd, 3 * n_embd))
        self.b_qkv = np.zeros(3 * n_embd)
        self.w_proj = rng.normal(0, 1 / np.sqrt(n_embd), (n_embd, n_embd))
        self.b_proj = np.zeros(n_embd)
        self.grads: dict[str, Array] = {}

    def forward(self, x: Array, cache: KVCache | None = None, layer_idx: int = 0,
                positions: Array | None = None) -> Array:
        B, T, C = x.shape
        H, d = self.n_head, self.d_head
        self._x, self._B, self._T, self._C = x, B, T, C

        self._qkv = x @ self.w_qkv + self.b_qkv
        q, k, v = np.split(self._qkv, 3, axis=-1)

        def heads(t: Array) -> Array:
            return np.transpose(t.reshape(B, T, H, d), (0, 2, 1, 3))

        qh, kh, vh = heads(q), heads(k), heads(v)

        # ⚠️ 易错点（作者在此处踩过坑）：
        #   位置**必须由调用方（GPT.forward）在整趟前向开始时算一次**，不能在这里读
        #   cache.seq_len —— 因为每个 block 追加 KV 后 seq_len 就变了，
        #   会导致第 2 层的 position 偏移第 1 层一个位置，RoPE 全错。
        if positions is None:
            start = cache.seq_len if cache is not None else 0
            positions = np.arange(start, start + T, dtype=np.float64)
        self._positions = positions

        # RoPE 是正交变换，反向时再转一次（见 rope_backward）即可
        self._qh, self._kh = qh, kh
        qh_rope = apply_rope(qh, positions)
        kh_rope = apply_rope(kh, positions)

        if cache is not None:
            kh_rope, vh = cache.append(layer_idx, kh_rope, vh)
        self._kh_full, self._vh_full = kh_rope, vh

        T_k = kh_rope.shape[2]
        mask = causal_mask_bool(T, T_k) if T > 1 else None
        self._mask = mask

        # 掩码不用 -1e9 魔法值，而是"先算 softmax 再把被屏蔽位置置零并重新归一化"，
        # 这样不会污染 max 的数值范围（-1e9 会把 exp 全部压到 0，虽能工作但不干净）
        scores = qh_rope @ np.swapaxes(kh_rope, -1, -2) / np.sqrt(d)
        self._scores = scores
        if mask is not None:
            weights = softmax(scores, axis=-1) * mask
            weights = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-12)
        else:
            weights = softmax(scores, axis=-1)
        self._weights = weights
        self._qh_rope = qh_rope

        out = self._weights @ vh
        self._out_heads = out
        merged = np.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)
        self._merged = merged
        return merged @ self.w_proj + self.b_proj

    def backward(self, dout: Array) -> Array:
        B, T, C, H, d = self._B, self._T, self._C, self.n_head, self.d_head

        self.grads = {"w_proj": self._merged.reshape(-1, C).T @ dout.reshape(-1, C),
                      "b_proj": dout.sum(axis=(0, 1))}
        d_merged = dout @ self.w_proj.T

        d_out = np.transpose(d_merged.reshape(B, T, H, d), (0, 2, 1, 3))

        # ⚠️ 易错点（作者在此处踩过坑，值得记住）：
        #   out[t,:] = Σ_j W[t,j] · V[j,:]
        #   => dV[j,:] = Σ_t W[t,j] · dout[t,:] = (Wᵀ @ dout)[j,:]
        #   是 Wᵀ 而不是 W！W 是 (T_q, T_k) 的注意力权重矩阵，它对 V 的作用方式是
        #   "按列"加权求和，所以求 V 的梯度要做转置。
        #   同理 dW[t,j] = dout[t,:] · V[j,:]ᵀ（这一路确实用 W 不转置，即下行的 d_weights）。
        dv = np.swapaxes(self._weights, -1, -2) @ d_out
        d_weights = d_out @ np.swapaxes(self._vh_full, -1, -2)

        d_scores = softmax_backward(d_weights, self._weights)
        if self._mask is not None:
            d_scores = np.where(self._mask, d_scores, 0.0)
        d_scores = d_scores / np.sqrt(d)

        d_q_rope = d_scores @ self._kh_full
        d_k_rope = np.swapaxes(d_scores, -1, -2) @ self._qh_rope

        # 只把 K/V 的梯度回传给"本步新增"的那部分（缓存的历史部分不属于本次计算图）
        T_new = T
        d_k_rope = d_k_rope[:, :, -T_new:, :]
        d_v = dv[:, :, -T_new:, :]

        # RoPE 反向（正交变换的转置）
        d_q = rope_backward(d_q_rope, self._positions)
        d_k = rope_backward(d_k_rope, self._positions)

        def unheads(t: Array) -> Array:
            return np.transpose(t, (0, 2, 1, 3)).reshape(B, T, C)

        d_qkv = np.concatenate([unheads(d_q), unheads(d_k), unheads(d_v)], axis=-1)
        self.grads["w_qkv"] = self._x.reshape(-1, C).T @ d_qkv.reshape(-1, 3 * C)
        self.grads["b_qkv"] = d_qkv.sum(axis=(0, 1))
        return d_qkv @ self.w_qkv.T


def _inv_freq(dim: int) -> Array:
    half = dim // 2
    return 1.0 / (10000.0 ** (np.arange(half, dtype=np.float64) / half))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class Block:
    """x = x + Attn(Norm(x));  x = x + FFN(Norm(x))"""

    def __init__(self, n_embd: int, n_head: int, rng: np.random.Generator) -> None:
        self.norm1, self.attn = RMSNorm(n_embd), Attention(n_embd, n_head, rng)
        self.norm2, self.ffn = RMSNorm(n_embd), SwiGLU(n_embd, rng)

    def forward(self, x: Array, cache: KVCache | None = None, layer_idx: int = 0,
                positions: Array | None = None) -> Array:
        self._n1 = self.norm1.forward(x)
        x = x + self.attn.forward(self._n1, cache, layer_idx, positions)
        self._n2 = self.norm2.forward(x)
        return x + self.ffn.forward(self._n2)

    def backward(self, dout: Array) -> tuple[Array, dict[str, Array]]:
        """返回 (对输入的梯度, 参数字典)。键名与 GPT.params() 保持一致（去掉 ffn./attn. 前缀）。"""
        g: dict[str, Array] = {}
        # FFN 分支
        d_ffn = self.ffn.backward(dout)
        for k, v in self.ffn.grads.items():
            g[k] = v
        d_norm2 = self.norm2.backward(d_ffn)
        g["norm2.g"] = self.norm2.dg
        d_in = dout + d_norm2
        # Attention 分支
        d_attn = self.attn.backward(d_in)
        for k, v in self.attn.grads.items():
            g[k] = v
        d_norm1 = self.norm1.backward(d_attn)
        g["norm1.g"] = self.norm1.dg
        return d_in + d_norm1, g


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


def sample_token(
    logits: Array,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    recent_tokens: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """从 logits 采样下一个 token（logits 形状 (V,)）。

    参数（面试高频）：
        temperature   <1 更确定保守；>1 更随机易跑偏；=0 等价贪心
        top_k         只在最高 k 个里采样，砍长尾
        top_p         核采样：累计概率达 p 的最小集合，比 top_k 更自适应
        repetition_penalty  对已出现 token 的 logit 打折，抑制复读

    ⚠️ temperature=0 也不能保证跨次完全一致：浮点累加顺序、同 batch 其他请求、
       kernel 实现、MoE 路由都会造成差异。要可复现就固定 seed 并 pin 模型版本。
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    rng = rng or np.random.default_rng()

    if repetition_penalty != 1.0 and recent_tokens:
        for t in set(recent_tokens):
            if 0 <= t < logits.shape[0]:
                # 正 logit 除以惩罚（压低），负 logit 乘以惩罚（更负）—— 都朝"更不可能"方向
                logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty

    if temperature <= 0:
        return int(np.argmax(logits))

    probs = softmax(logits / temperature, axis=-1)

    if top_k is not None and 0 < top_k < probs.shape[0]:
        kth = np.sort(probs)[-top_k]
        probs = np.where(probs >= kth, probs, 0.0)

    if top_p is not None and 0 < top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cum, top_p, side="left")) + 1
        keep = order[:cutoff]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked

    total = probs.sum()
    if total <= 0:
        return int(np.argmax(logits))
    return int(rng.choice(probs.shape[0], p=probs / total))


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------


class GPT:
    def __init__(
        self,
        vocab_size: int,
        *,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.vocab_size, self.n_layer, self.n_head = vocab_size, n_layer, n_head
        self.n_embd, self.block_size = n_embd, block_size
        rng = np.random.default_rng(seed)
        self.wte = rng.normal(0, 0.02, (vocab_size, n_embd))
        self.wpe = rng.normal(0, 0.01, (block_size, n_embd))
        self.blocks = [Block(n_embd, n_head, rng) for _ in range(n_layer)]
        self.norm_f = RMSNorm(n_embd)
        self._targets: Array | None = None

    # ------- 参数视图（供优化器使用）-------

    def params(self) -> dict[str, Array]:
        p: dict[str, Array] = {"wte": self.wte, "wpe": self.wpe, "norm_f.g": self.norm_f.g}
        for i, b in enumerate(self.blocks):
            p[f"b{i}.norm1.g"] = b.norm1.g
            p[f"b{i}.norm2.g"] = b.norm2.g
            for k, v in (("w_qkv", b.attn.w_qkv), ("b_qkv", b.attn.b_qkv),
                         ("w_proj", b.attn.w_proj), ("b_proj", b.attn.b_proj),
                         ("w_gate", b.ffn.w_gate), ("w_up", b.ffn.w_up),
                         ("w_down", b.ffn.w_down)):
                p[f"b{i}.{k}"] = v
        return p

    # ------- 前向 -------

    def forward(self, idx: Array, targets: Array | None = None,
                cache: KVCache | None = None) -> tuple[Array, float | None]:
        B, T = idx.shape
        start = cache.seq_len if cache is not None else 0
        assert start + T <= self.block_size, f"超长: {start + T} > {self.block_size}"
        self._idx, self._T = idx, T

        # 整趟前向只算一次位置，传给所有层（关键：不能每层各读一次 cache.seq_len）
        positions = np.arange(start, start + T, dtype=np.float64)

        x = self.wte[idx] + self.wpe[start : start + T][None]
        for i, blk in enumerate(self.blocks):
            x = blk.forward(x, cache, i, positions)
        self._x_final = x
        x = self.norm_f.forward(x)
        self._x_normed = x
        logits = x @ self.wte.T                       # weight tying：输出层复用输入 embedding

        loss = None
        if targets is not None:
            flat_t = targets.reshape(-1)
            probs = softmax(logits.reshape(-1, self.vocab_size), axis=-1)
            self._probs, self._targets = probs, flat_t
            loss = float(-np.mean(np.log(probs[np.arange(flat_t.size), flat_t] + 1e-12)))
        return logits, loss

    # ------- 反向 -------

    def backward(self) -> dict[str, Array]:
        assert self._targets is not None, "backward 需要 forward 时传入 targets"
        B, T, V, C = *self._idx.shape, self.vocab_size, self.n_embd
        n = self._targets.size

        dlogits = self._probs.copy()
        dlogits[np.arange(n), self._targets] -= 1.0
        dlogits = (dlogits / n).reshape(B, T, V)

        # LM head 与 wte 共享：两路梯度相加
        d_x_normed = dlogits @ self.wte
        g_wte = dlogits.reshape(-1, V).T @ self._x_normed.reshape(-1, C)

        d_x = self.norm_f.backward(d_x_normed)
        grads: dict[str, Array] = {"norm_f.g": self.norm_f.dg}

        for i in reversed(range(self.n_layer)):
            d_x, bg = self.blocks[i].backward(d_x)
            for k, v in bg.items():
                grads[f"b{i}.{k}"] = v

        # embedding 反向：token 位置累加
        g_wte_bp = np.zeros_like(self.wte)
        np.add.at(g_wte_bp, self._idx.reshape(-1), d_x.reshape(-1, C))
        grads["wte"] = g_wte + g_wte_bp

        g_wpe = np.zeros_like(self.wpe)
        np.add.at(g_wpe[:T], np.arange(T), d_x.sum(axis=0))
        grads["wpe"] = g_wpe
        return grads

    # ------- 生成 -------

    def generate(self, idx: Array, max_new_tokens: int = 100, *, temperature: float = 0.8,
                 top_k: int | None = 20, top_p: float | None = 0.95,
                 repetition_penalty: float = 1.1, use_cache: bool = True,
                 seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        B = idx.shape[0]
        cache = KVCache(self.n_layer, B, self.n_head, self.n_embd // self.n_head) if use_cache else None
        out = idx
        for _ in range(max_new_tokens):
            if cache is not None:
                idx_cond = out if cache.seq_len == 0 else out[:, -1:]
            else:
                idx_cond = out[:, -self.block_size :]
            logits, _ = self.forward(idx_cond, cache=cache)
            nxt = sample_token(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                               repetition_penalty=repetition_penalty,
                               recent_tokens=out[0].tolist(), rng=rng)
            out = np.concatenate([out, np.array([[nxt]], dtype=INT)], axis=1)
        return out

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params().values()))


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


class Adam:
    """一阶动量 + 二阶动量 + 偏差校正。

    二阶动量 v 让每个参数有自适应学习率：除以 sqrt(v) 后，梯度大的参数步长被压小、
    梯度小的被放大，避免"统一 lr 要么震荡要么不动"。
    """

    def __init__(self, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: dict[str, Array] = {}
        self.v: dict[str, Array] = {}
        self.t = 0

    def step(self, params: dict[str, Array], grads: dict[str, Array]) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            if k not in self.m:
                self.m[k], self.v[k] = np.zeros_like(p), np.zeros_like(p)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# 数值梯度校验 —— 证明手写反向是对的
# ---------------------------------------------------------------------------


def check_gradients(
    vocab_size: int = 17,
    n_layer: int = 2,
    n_embd: int = 16,
    n_head: int = 2,
    block_size: int = 6,
    n_checks: int = 12,
    eps: float = 1e-5,
    tol: float = 2e-4,
    seed: int = 0,
) -> bool:
    """有限差分校验：解析梯度 vs 数值梯度。"""
    model = GPT(vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size, seed=seed)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, vocab_size, (2, block_size))
    tgt = rng.integers(0, vocab_size, (2, block_size))

    params = model.params()
    _, _ = model.forward(idx, tgt)
    grads = model.backward()

    names = sorted(params)
    print(f"校验 {len(names)} 个参数张量中的 {n_checks} 个随机元素（eps={eps}）\n")
    print(f"{'参数':<16}{'解析':>14}{'数值':>14}{'相对误差':>12}  结果")
    print("-" * 62)

    ok_all = True
    for _ in range(n_checks):
        name = names[rng.integers(0, len(names))]
        p = params[name]
        if p.size == 0:
            continue
        flat_i = int(rng.integers(0, p.size))
        unrav = np.unravel_index(flat_i, p.shape)

        orig = p[unrav]
        p[unrav] = orig + eps
        _, lp = model.forward(idx, tgt)
        p[unrav] = orig - eps
        _, lm = model.forward(idx, tgt)
        p[unrav] = orig

        num = (float(lp) - float(lm)) / (2 * eps)
        ana = float(grads[name][unrav])
        denom = max(abs(num), abs(ana), 1e-8)
        rel = abs(num - ana) / denom
        ok = rel < tol
        ok_all &= ok
        print(f"{name:<16}{ana:>14.6f}{num:>14.6f}{rel:>12.2e}  {'✅' if ok else '❌'}")

    print("-" * 62)
    print("结论：", "✅ 所有抽查元素的解析梯度与数值梯度一致，反向传播实现正确"
          if ok_all else "❌ 存在不一致，反向传播有 bug")
    return ok_all


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

TOY_TEXT = (
    "Agent 通过工具调用与外部世界交互。"
    "Agent 会先思考，再行动，然后观察结果。"
    "Agent 需要记忆来保持多轮对话的连贯性。"
    "一个 Agent 由模型、工具、记忆和循环控制组成。"
) * 10


def _demo() -> None:
    print("=== 1. 训练字节级 BPE ===")
    tok = BPETokenizer()
    tok.train(TOY_TEXT, vocab_size=400)

    data = np.array(tok.encode(TOY_TEXT), dtype=INT)
    block_size = 48
    print(f"语料 {len(data)} tokens，词表 {tok.next_id}\n")

    print("=== 2. 初始化模型 ===")
    model = GPT(tok.next_id, n_layer=4, n_head=4, n_embd=128, block_size=block_size, seed=42)
    x = data[:block_size][None]
    y = data[1 : block_size + 1][None]
    _, loss0 = model.forward(x, y)
    print(f"参数量 {model.n_params():,}")
    print(f"初始 loss = {loss0:.3f}　（随机初始化理论值 ln(V) = {np.log(tok.next_id):.3f}）\n")

    print("=== 3. 训练：loss 是否下降（这是 W1 的验收标准）===")
    params = model.params()
    opt = Adam(lr=3e-3)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, 201):
        _, loss = model.forward(x, y)
        assert loss is not None
        losses.append(loss)
        opt.step(params, model.backward())
        if step % 25 == 0 or step == 1:
            print(f"step {step:>3}  loss = {loss:.4f}")
    dt = time.perf_counter() - t0
    print(f"\n耗时 {dt:.1f}s　loss: {losses[0]:.3f} -> {losses[-1]:.3f}  "
          f"（下降 {(1 - losses[-1] / losses[0]) * 100:.1f}%）")
    assert losses[-1] < losses[0] * 0.5, "loss 未显著下降，反向传播可能有问题！"
    print("✅ loss 显著下降，反向传播有效\n")

    print("=== 4. 采样参数对比（同一 prompt，不同策略）===")
    prompt = np.array([tok.encode("Agent ")])
    for label, kw in [
        ("贪心 t=0", dict(temperature=0.0)),
        ("保守 t=0.3 top_k=5", dict(temperature=0.3, top_k=5, top_p=None)),
        ("平衡 t=0.8 top_p=0.95", dict(temperature=0.8, top_k=None, top_p=0.95)),
        ("发散 t=1.5", dict(temperature=1.5, top_k=None, top_p=None)),
    ]:
        out = model.generate(prompt, max_new_tokens=40, repetition_penalty=1.05, seed=0, **kw)
        print(f"  [{label:<22}] {tok.decode(out[0].tolist())}")

    print("\n=== 5. KV Cache 的正确性与收益 ===")
    full = model.forward(np.array([tok.encode("Agent 通过工具")]))[0]
    c = KVCache(model.n_layer, 1, model.n_head, model.n_embd // model.n_head)
    ids = tok.encode("Agent 通过工具")
    inc = np.concatenate(
        [model.forward(np.array([[t]], dtype=INT), cache=c)[0] for t in ids], axis=1
    )
    print(f"全量前向 vs 增量缓存 最大误差 = {np.abs(full - inc).max():.2e}")

    prompt = np.array([tok.encode("Agent 通过")])
    t0 = time.perf_counter()
    model.generate(prompt, max_new_tokens=40, temperature=0.0, use_cache=True)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.generate(prompt, max_new_tokens=40, temperature=0.0, use_cache=False)
    tn = time.perf_counter() - t0
    print(f"生成 40 token：带缓存 {tc * 1000:.0f} ms | 不带 {tn * 1000:.0f} ms | 加速 {tn / tc:.2f}x")
    print("（配置小、序列短时差距不明显；层数与长度增大后 O(T²)→O(T) 会急剧放大）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mini-GPT 演示")
    ap.add_argument("--check-grad", action="store_true", help="运行数值梯度校验")
    args = ap.parse_args()
    if args.check_grad:
        raise SystemExit(0 if check_gradients() else 1)
    _demo()
