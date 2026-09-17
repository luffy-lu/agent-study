"""组件级梯度诊断：逐个模块做有限差分校验，定位反向传播的错源。

用法：
    uv run python -m w1_llm.diag_grad
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .attention import KVCache  # noqa: F401  （保留：便于在本文件里手工扩展诊断）
from .mini_gpt import Attention, Block, RMSNorm, SwiGLU

Array = NDArray[np.floating]


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def check_tensor(
    forward_fn,
    backward_fn,
    params: dict[str, Array],
    *,
    n_checks: int = 5,
    eps: float = 1e-6,
    tol: float = 1e-5,
    seed: int = 0,
) -> tuple[bool, list[str]]:
    """对给定前向/反向函数做数值梯度校验。

    forward_fn: () -> float   （标量损失）
    backward_fn: () -> dict[str, Array]   （解析梯度）
    """
    grads = backward_fn()
    rng = np.random.default_rng(seed)
    msgs: list[str] = []
    ok_all = True
    for name in sorted(params):
        p = params[name]
        for _ in range(n_checks):
            i = int(rng.integers(0, p.size))
            u = np.unravel_index(i, p.shape)
            orig = p[u]
            p[u] = orig + eps
            lp = forward_fn()
            p[u] = orig - eps
            lm = forward_fn()
            p[u] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(grads[name][u])
            e = rel_err(ana, num)
            ok = e < tol
            ok_all &= ok
            msgs.append(f"    {name:<12} 解析={ana:>12.7f} 数值={num:>12.7f} 误差={e:.2e} {'✅' if ok else '❌'}")
    return ok_all, msgs


def main() -> None:
    rng = np.random.default_rng(0)

    # ---------------- RMSNorm ----------------
    print("=" * 70)
    print("1. RMSNorm")
    x = rng.normal(0, 1, (2, 3, 8))
    norm = RMSNorm(8)
    y = norm.forward(x)
    dout = rng.normal(0, 1, y.shape)
    dx_ana = norm.backward(dout)
    dg_ana = norm.dg
    params = {"g": norm.g}

    def fwd() -> float:
        return float(np.sum(norm.forward(x) * dout))

    ok, msgs = check_tensor(fwd, lambda: {"g": dg_ana}, params)

    # 同时校验 dx（用随机方向的方向导数）
    d = rng.normal(0, 1, x.shape)
    eps = 1e-6
    num_dx = (float(np.sum(norm.forward(x + eps * d) * dout)) - float(np.sum(norm.forward(x - eps * d) * dout))) / (2 * eps)
    ana_dx = float(np.sum(dx_ana * d))
    print(f"    dx 方向导数: 解析={ana_dx:.8f} 数值={num_dx:.8f} 误差={rel_err(ana_dx, num_dx):.2e} "
          f"{'✅' if rel_err(ana_dx, num_dx) < 1e-5 else '❌'}")
    print("\n".join(msgs))

    # ---------------- SwiGLU ----------------
    print("=" * 70)
    print("2. SwiGLU")
    x2 = rng.normal(0, 1, (2, 3, 8))
    ffn = SwiGLU(8, np.random.default_rng(1))
    dout2 = rng.normal(0, 1, (2, 3, 8))
    ffn.forward(x2)                 # 先前向，backward 依赖缓存
    dx2 = ffn.backward(dout2)
    p2 = {"w_gate": ffn.w_gate, "w_up": ffn.w_up, "w_down": ffn.w_down}
    ok2, msgs2 = check_tensor(lambda: float(np.sum(ffn.forward(x2) * dout2)),
                              lambda: ffn.grads, p2)
    eps = 1e-6
    d2 = rng.normal(0, 1, x2.shape)
    num2 = (float(np.sum(ffn.forward(x2 + eps * d2) * dout2)) - float(np.sum(ffn.forward(x2 - eps * d2) * dout2))) / (2 * eps)
    ana2 = float(np.sum(dx2 * d2))
    print(f"    dx 方向导数: 解析={ana2:.8f} 数值={num2:.8f} 误差={rel_err(ana2, num2):.2e} "
          f"{'✅' if rel_err(ana2, num2) < 1e-5 else '❌'}")
    print("\n".join(msgs2))

    # ---------------- Attention ----------------
    print("=" * 70)
    print("3. Attention（含因果掩码 + RoPE，无缓存）")
    att = Attention(8, 2, np.random.default_rng(2))
    x3 = rng.normal(0, 1, (1, 4, 8))
    dout3 = rng.normal(0, 1, (1, 4, 8))
    att.forward(x3)                 # 先前向
    dx3 = att.backward(dout3)
    p3 = {"w_qkv": att.w_qkv, "b_qkv": att.b_qkv, "w_proj": att.w_proj, "b_proj": att.b_proj}
    ok3, msgs3 = check_tensor(lambda: float(np.sum(att.forward(x3) * dout3)),
                              lambda: att.grads, p3)
    eps = 1e-6
    d3 = rng.normal(0, 1, x3.shape)
    num3 = (float(np.sum(att.forward(x3 + eps * d3) * dout3)) - float(np.sum(att.forward(x3 - eps * d3) * dout3))) / (2 * eps)
    ana3 = float(np.sum(dx3 * d3))
    print(f"    dx 方向导数: 解析={ana3:.8f} 数值={num3:.8f} 误差={rel_err(ana3, num3):.2e} "
          f"{'✅' if rel_err(ana3, num3) < 1e-5 else '❌'}")
    print("\n".join(msgs3))

    # ---------------- Block ----------------
    print("=" * 70)
    print("4. Block（单个 Transformer 层）")
    blk = Block(8, 2, np.random.default_rng(3))
    x4 = rng.normal(0, 1, (1, 4, 8))
    dout4 = rng.normal(0, 1, (1, 4, 8))
    blk.forward(x4)                 # 先前向
    dx4, g4 = blk.backward(dout4)
    p4 = {
        "norm1.g": blk.norm1.g, "norm2.g": blk.norm2.g,
        "w_qkv": blk.attn.w_qkv, "b_qkv": blk.attn.b_qkv,
        "w_proj": blk.attn.w_proj, "b_proj": blk.attn.b_proj,
        "w_gate": blk.ffn.w_gate, "w_up": blk.ffn.w_up, "w_down": blk.ffn.w_down,
    }
    ok4, msgs4 = check_tensor(lambda: float(np.sum(blk.forward(x4) * dout4)),
                              lambda: g4, p4, n_checks=3)
    eps = 1e-6
    d4 = rng.normal(0, 1, x4.shape)
    num4 = (float(np.sum(blk.forward(x4 + eps * d4) * dout4)) - float(np.sum(blk.forward(x4 - eps * d4) * dout4))) / (2 * eps)
    ana4 = float(np.sum(dx4 * d4))
    print(f"    dx 方向导数: 解析={ana4:.8f} 数值={num4:.8f} 误差={rel_err(ana4, num4):.2e} "
          f"{'✅' if rel_err(ana4, num4) < 1e-5 else '❌'}")
    print("\n".join(msgs4))

    print("=" * 70)
    print("结论：哪个模块报 ❌，就是哪个模块的反向有问题")


if __name__ == "__main__":
    main()
