"""字节级 BPE 分词器 —— 从零实现，无第三方依赖。

为什么 W1 第一天就要手写 BPE？
    因为 Token 是 LLM 的"原子单位"：它决定成本、上下文长度、以及模型能不能看见
    中文/代码/数字。不理解分词，后面所有关于"上下文窗口""Token 计费""为什么模型
    数不清字母"的问题都只能背结论。

实现路线（与 GPT-2 一致的 byte-level BPE）：
    1. 文本 → UTF-8 字节序列（0..255），这一步就保证了"任何字符都能编码"，
       不会出现 <UNK>，中文一个字 = 3 个字节。
    2. 统计相邻 token 对的频次，反复合并最高频的一对，每次合并产生一个新 token id。
    3. 训练结束后，把每个 token 的字节序列存下来，编码时按合并"优先级"（rank）贪心应用。

关键认知：
    - 词表大小每 +1，就多一个"更长的字节片段"，压缩率更高但 embedding 表更大。
    - 编码时用 rank 而非重新统计频次，是为了 O(#merges) 而不是 O(语料) 复杂度。

运行：
    uv run python -m w1_llm.bpe
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def get_pair_counts(token_ids: Iterable[int]) -> Counter[tuple[int, int]]:
    """统计序列中所有相邻 token 对的频次。

    例：[1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}
    """
    ids = list(token_ids)
    return Counter(zip(ids, ids[1:], strict=False))


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把序列中所有出现的 pair 合并为 new_id（从左到右、不重叠）。

    例：merge([1,2,1,2], (1,2)=9) -> [9,9]
    例：merge([1,1,1], (1,1)=9)   -> [9,1]  （左边优先，不重叠）
    """
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i] == a and i + 1 < n and token_ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(token_ids[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------


class BPETokenizer:
    """byte-level BPE 分词器。

    属性：
        merges:      (id_a, id_b) -> 新 token id，按训练顺序（顺序即优先级）
        vocab:       token id -> 对应的原始字节（bytes）
        next_id:     下一个可用的 token id
    """

    def __init__(self) -> None:
        # 0..255 为单字节 token，字节值即 id
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.next_id: int = 256
        # 编码时用：pair -> (rank, new_id)，rank 越小优先级越高
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}

    # ---------------- 训练 ----------------

    def train(
        self,
        text: str,
        vocab_size: int = 1024,
        *,
        verbose: bool = False,
        verbose_every: int = 100,
    ) -> None:
        """在 text 上学习 BPE 合并规则，直到词表达到 vocab_size。

        vocab_size 含 256 个基础字节 token。
        """
        if vocab_size <= 256:
            raise ValueError("vocab_size 必须 > 256（前 256 个是单字节 token）")

        ids = list(text.encode("utf-8"))
        print(f"[train] 语料 {len(text)} 字符 -> {len(ids)} 字节；目标词表 {vocab_size}")

        while self.next_id < vocab_size:
            counts = get_pair_counts(ids)
            if not counts:
                print("[train] 序列已合并为单个 token，提前停止")
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                print(f"[train] 最高频 pair 频次={freq} < 2，继续合并无收益，提前停止")
                break

            new_id = self.next_id
            ids = merge_pair(ids, pair, new_id)

            self.merges[pair] = new_id
            self._ranks[pair] = (len(self.merges) - 1, new_id)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.next_id += 1

            n_merges = len(self.merges)
            if verbose and n_merges % verbose_every == 0:
                piece = self.vocab[new_id]
                print(
                    f"[train] merges={n_merges:>4}  "
                    f"new_id={new_id:>4}  freq={freq:>5}  "
                    f"token={piece!r}  当前序列长度={len(ids)}"
                )

        print(
            f"[train] 完成：merges={len(self.merges)}  词表={self.next_id}  "
            f"压缩率={len(ids) / len(text.encode('utf-8')):.3f} token/byte"
        )

    # ---------------- 编码 / 解码 ----------------

    def _encode_bytes(self, bs: bytes) -> list[int]:
        """对一段字节序列按 rank 贪心应用合并规则。"""
        ids = list(bs)
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # 在所有"存在合并规则"的 pair 中，选 rank 最小（即训练最早）的那个
            best_pair: tuple[int, int] | None = None
            best_new_id = 0
            best_rank = len(self.merges) + 1
            for pair in counts:
                entry = self._ranks.get(pair)
                if entry is not None and entry[0] < best_rank:
                    best_rank, best_pair, best_new_id = entry[0], pair, entry[1]
            if best_pair is None:
                break
            ids = merge_pair(ids, best_pair, best_new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表。无 <UNK>：任何字符都能被编码。"""
        return self._encode_bytes(text.encode("utf-8"))

    def encode_iter(self, texts: Iterable[str]) -> Iterator[int]:
        """流式编码多个文本，逐个吐出 token id（后面做数据集时用）。"""
        for t in texts:
            yield from self.encode(t)

    def decode(self, ids: Iterable[int]) -> str:
        """token id 列表 -> 文本。"""
        bs = b"".join(self.vocab[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    # ---------------- 持久化 ----------------

    def _rebuild_ranks(self) -> None:
        self._ranks = {pair: (r, new_id) for r, (pair, new_id) in enumerate(self.merges.items())}

    def save(self, path: str | Path) -> None:
        """保存为 JSON（字节用 latin-1 单射映射成 str，保证可逆）。"""
        path = Path(path)
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[save] {path}（{path.stat().st_size / 1024:.1f} KB）")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = {(int(a), int(b)): int(i) for a, b, i in data["merges"]}
        tok.vocab = {int(k): v.encode("latin-1") for k, v in data["vocab"].items()}
        tok.next_id = max(tok.vocab) + 1
        tok._rebuild_ranks()
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab={self.next_id}, merges={len(self.merges)})"


# ---------------------------------------------------------------------------
# 自检 / 演示
# ---------------------------------------------------------------------------

TOY_CORPUS = (
    "人工智能正在改变世界。AI agents 可以调用工具、进行推理并完成复杂任务。"
    "大语言模型通过 next token prediction 学习语言的统计规律。"
    "The quick brown fox jumps over the lazy dog. "
    "Agent 开发需要理解 tokenization, attention, KV cache 与 tool calling。"
    "北京是中国的首都。上海是最大的城市。" * 4
)


def _demo() -> None:
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=600, verbose=True, verbose_every=50)

    print("\n=== 编码结果（Token 视角） ===")
    for s in ["人工智能", "AI agents", "北京是中国的首都", "tokenization", "1234567890"]:
        ids = tok.encode(s)
        pieces = [tok.vocab[i] for i in ids]
        print(f"{s!r:28} -> {len(ids):>2} tokens  {pieces}")

    print("\n=== 压缩率对比（token 数 / 字符数）===")
    for s in ["人工智能", "Agent 开发需要理解 tokenization", "Hello, world!"]:
        n_tok, n_chr, n_byte = len(tok.encode(s)), len(s), len(s.encode("utf-8"))
        print(f"{s!r:38} chars={n_chr:>2} bytes={n_byte:>2} tokens={n_tok:>2}  "
              f"字节压缩={n_byte / n_tok:.2f}x")

    print("\n=== round-trip 无损性验证 ===")
    ok = all(tok.decode(tok.encode(s)) == s for s in [TOY_CORPUS, "emoji 🚀🎯", "混合 mixed 文本"])
    print("decode(encode(x)) == x :", ok)
    assert ok, "round-trip 失败！"

    print("\n=== 思考题（面试会问）===")
    print("1. 为什么中文一句话的 token 数 ≈ 字符数 × 1.5~2？")
    print("2. 词表 600 vs 50000 分别有什么代价？")
    print("3. 为什么模型算不清 'strawberry 里有几个 r'？")


if __name__ == "__main__":
    _demo()
