# W0 预热周：环境搭建 + Python 突击 + 自检清单

> 目标：一次性清掉所有环境与语言障碍，让 W1 之后每天都能"打开就写"。
> 预计耗时：3 天（每天 3h）；若你 Python 已熟练，可压缩到 1 天。

---

## 一、环境搭建（约 2h）

### 1.1 基础运行时

```bash
# macOS（你是 macOS 环境）
# 1. Homebrew（若未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Python 3.11+（不要用系统自带 3.9）
brew install python@3.12 uv

# 3. Docker Desktop（跑向量库/Redis/PG/Langfuse 用）
brew install --cask docker

# 4. Node（前端可视化用）
brew install node pnpm

# 5. 验证
python3 --version    # >= 3.11
uv --version
docker --version
node --version
```

### 1.2 Python 工程习惯（**这是面试会看的细节**）

```bash
cd ~/ai/agent-study
uv init --package labs          # 或 mkdir labs && cd labs && uv init
cd labs
uv add httpx openai pydantic python-dotenv rich
uv add --dev pytest ruff mypy ipython
uv run python -c "import httpx, pydantic; print('ok')"
```

约定：
- 每个实验目录独立 `pyproject.toml`，依赖用 `uv` 管（不用全局 pip install）
- 统一用 `ruff format` + `ruff check --fix`，写代码时启用类型标注
- `.env` 存 API Key，**绝不提交到 Git**；提交 `.env.example`

```bash
# .gitignore 至少要包含
.env
.venv/
__pycache__/
*.pyc
```

### 1.3 VS Code 扩展（按重要性）

| 扩展 | 作用 |
|---|---|
| Python + Pylance | 类型提示与跳转 |
| Ruff | 格式化 + lint |
| Docker | 容器管理 |
| Jupyter | 跑实验室脚本 |
| GitLens | 代码历史 |
| Error Lens | 行内报错 |

`settings.json` 建议：

```json
{
  "python.analysis.typeCheckingMode": "basic",
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff", "editor.formatOnSave": true },
  "editor.rulers": [100]
}
```

---

## 二、模型与可观测性账号（约 1h）

### 2.1 模型 API（至少配两路，一主一备）

| 用途 | 推荐 | 说明 |
|---|---|---|
| 主力对话/工具调用 | DeepSeek / Qwen（阿里百炼）/ 智谱 | 便宜、Function Calling 支持好 |
| 强推理备选 | Claude / GPT 系列 | 用于对照评测、难题 |
| Embedding | 通义 text-embedding / BGE 系列 | 也可本地跑 BGE-M3 |
| Rerank | BGE-reranker（本地或 API） | W2 需要 |
| 本地备胎 | Ollama + qwen2.5:7b | 断网/省钱/开发调试 |

```bash
# 本地备胎
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
curl http://localhost:11434/api/tags   # 验证
```

统一配置方式（`.env`）：

```dotenv
PRIMARY_BASE_URL=https://api.deepseek.com/v1
PRIMARY_API_KEY=sk-xxx
PRIMARY_MODEL=deepseek-chat
BACKUP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BACKUP_API_KEY=sk-xxx
BACKUP_MODEL=qwen-plus
EMBEDDING_BASE_URL=...
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=...
LOCAL_BASE_URL=http://localhost:11434/v1
```

> 大部分国产模型兼容 OpenAI 协议，所以**统一用 OpenAI 兼容格式 + 可切换 base_url** 是最省事的做法，也正好练出"模型路由"的雏形。

### 2.2 可观测性

- 选一：Langfuse Cloud 免费版 / 本地 `docker compose` 自托管
- 拿到 `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` / `HOST` 写入 `.env`
- W4 之前不必接入，先把账号建好

### 2.3 数据库与向量库（Docker）

```bash
docker run -d --name pg     -e POSTGRES_PASSWORD=dev -p 5432:5432 postgres:16
docker run -d --name redis  -p 6379:6379 redis:7-alpine
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
# Milvus 用官方 docker compose（W2 需要时再起）
```

---

## 三、Python 突击：30 道自测题

> 规则：**先不看答案自己写，跑通为止**。全部通过才算 W0 结束。这些题覆盖后面所有实验会用到的语法。

### 基础与数据建模（1–8）
1. 用 `dataclass` 定义一个 `Message`（role、content、tool_calls 可选），并实现 `to_dict()`。
2. 用 `pydantic` 定义 `ToolCallArgs`，要求 `city: str`、`days: int = 1`，并对非法输入抛出可读错误。
3. 写一个函数，接受任意关键字参数并生成 JSON Schema 风格的 dict。
4. 用 `typing.Protocol` 定义一个 `LLMClient` 协议（有 `async chat()` 方法）。
5. 用 `Literal` 限定 role 只能是 `"system"|"user"|"assistant"|"tool"`。
6. 用 `@dataclass(frozen=True)` 实现不可变的 `ToolResult`，并说明为什么 Agent 里推荐不可变状态。
7. 用 `enum.Enum` 定义工具权限等级（READ/WRITE/DANGEROUS）。
8. 写一个上下文管理器 `timer()`，用 `with` 打印代码块耗时。

### 异步与并发（9–16，**最关键**）
9. 用 `asyncio.gather` 并发调用 3 个模拟接口（`asyncio.sleep`），总耗时约等于最慢的一个。
10. 给任务加超时（`asyncio.wait_for`），超时后优雅降级返回默认值。
11. 用 `asyncio.Semaphore` 限制并发数为 3，处理 10 个任务。
12. 实现带指数退避的重试装饰器（最多 3 次，捕获特定异常）。
13. 用 `asyncio.Queue` 实现"生产者-消费者"：生产者读 SSE 流，消费者逐块处理。
14. 解释 `asyncio.create_task` 与 `await` 的区别，以及为什么 `create_task` 后必须持有引用（否则可能被 GC 回收）。
15. 用 `httpx.AsyncClient` 流式读取一个响应（`aiter_lines` / `aiter_bytes`）。
16. 实现 `asyncio.TaskGroup`（Python 3.11+）版本，并说明它与 `gather` 在异常传播上的差异。

### 文本与数据（17–24）
17. 读一个目录下所有 `.md` 文件，按标题层级切分成 dict 列表。
18. 实现一个滑窗函数：把 `list` 按 `window` 和 `step` 切成子列表。
19. 用 `regex` 从文本中抽取所有 `key: value` 对。
20. 手写余弦相似度（不用 numpy），并用它对 100 个向量做 Top-K 检索。
21. 用 `collections.Counter` 统计词频，手写一个"停用词过滤 + TF 计算"函数（为 BM25 打基础）。
22. 用 `json` 解析一个可能不合法（带 ```json 包裹）的模型输出，写一个健壮的 `extract_json()`。
23. 用 `hashlib` 对文本生成稳定 ID（用于去重与缓存键）。
24. 用 `pathlib` 遍历并统计一个目录代码行数（忽略 `.venv`）。

### 工程习惯（25–30）
25. 用 `argparse` 或 `typer` 给脚本加命令行参数（`--query`、`--top-k`）。
26. 用 `logging` 配置：控制台带颜色、文件记 DEBUG，并演示在异步环境中不乱序。
27. 写一个 `pytest` 用例，用 `monkeypatch` mock 掉 API 调用（**Agent 测试必备**）。
28. 用 `pytest.mark.asyncio`（或 `anyio`）测一个异步函数。
29. 实现一个简单的内存缓存装饰器（`functools.lru_cache` 不能用于 async，说明原因并手写异步版）。
30. 把上面的代码用 `ruff check` 和 `mypy` 跑通，0 error。

> 卡住的题记进 `PROGRESS.md`，W1 开始前必须清零。**这些不是"学习内容"，是"工具"，不熟练会拖累后面 7 周。**

---

## 四、环境自检脚本

保存为 `labs/check_env.py`，运行 `uv run python labs/check_env.py`，**全绿才进入 W1**。

```python
"""W0 环境自检：验证 Python 环境、依赖、API 连通性、本地服务。"""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OK, FAIL, WARN = "\033[92m✔\033[0m", "\033[91m✘\033[0m", "\033[93m!\033[0m"
results: list[tuple[bool, str]] = []


def check(name: str, passed: bool, detail: str = "", warn_only: bool = False) -> None:
    icon = OK if passed else (WARN if warn_only else FAIL)
    results.append((passed or warn_only, name))
    print(f"{icon} {name}" + (f"  — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    check("Python >= 3.11", v >= (3, 11), f"当前 {v.major}.{v.minor}.{v.micro}")


def check_imports() -> None:
    for mod in ("httpx", "openai", "pydantic", "dotenv", "pytest", "rich"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"import {mod}", False, repr(exc))


def check_binaries() -> None:
    for bin_ in ("uv", "docker", "node", "git"):
        check(f"bin {bin_}", shutil.which(bin_) is not None, shutil.which(bin_) or "未找到")


def check_port(name: str, host: str, port: int, warn_only: bool = True) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            check(f"{name} ({host}:{port})", True)
    except OSError:
        check(f"{name} ({host}:{port})", False, "未启动", warn_only=warn_only)


async def check_llm() -> None:
    import httpx

    base, key, model = (
        os.getenv("PRIMARY_BASE_URL"),
        os.getenv("PRIMARY_API_KEY"),
        os.getenv("PRIMARY_MODEL"),
    )
    if not (base and key and model):
        check("主模型 API", False, "缺少 PRIMARY_* 环境变量")
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": "回复：pong"}]},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            check("主模型 API", True, f"model={model} 回复={text[:20]!r}")
    except Exception as exc:  # noqa: BLE001
        check("主模型 API", False, repr(exc))


async def check_llm_stream() -> None:
    import httpx

    base, key, model = (
        os.getenv("PRIMARY_BASE_URL"),
        os.getenv("PRIMARY_API_KEY"),
        os.getenv("PRIMARY_MODEL"),
    )
    if not (base and key and model):
        check("流式输出", False, "跳过：无主模型配置")
        return
    try:
        chunks = 0
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "stream": True,
                    "messages": [{"role": "user", "content": "数到5"}],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks += 1
        check("流式输出", chunks > 1, f"收到 {chunks} 个分片")
    except Exception as exc:  # noqa: BLE001
        check("流式输出", False, repr(exc))


def check_dirs() -> None:
    for d in ("labs", "notes", "projects"):
        Path(d).mkdir(parents=True, exist_ok=True)
        check(f"目录 {d}/", True)


async def main() -> None:
    print("\n=== W0 环境自检 ===\n")
    check_python()
    check_imports()
    check_binaries()
    check_dirs()
    for name, host, port in (("PostgreSQL", "127.0.0.1", 5432), ("Redis", "127.0.0.1", 6379),
                             ("Qdrant", "127.0.0.1", 6333), ("Ollama", "127.0.0.1", 11434)):
        check_port(name, host, port)
    await check_llm()
    await check_llm_stream()

    failed = [n for ok, n in results if not ok]
    print("\n" + ("=" * 40))
    if failed:
        print(f"\033[91m未通过 {len(failed)} 项：\033[0m " + ", ".join(failed))
        print("→ 修完再进入 W1。标 ! 的本地服务可延后到 W2 再起。")
    else:
        print("\033[92m全部通过，可以进入 W1。\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 五、W0 结束检查表

- [ ] Python 3.11+ / uv / Docker / Node 均可用
- [ ] `labs`、`notes`、`projects` 三个目录已建，Git 仓库已初始化并提交
- [ ] `.env` 配好主/备模型 + Embedding，`.gitignore` 排除 `.env`
- [ ] 主模型**非流式**与**流式**调用均成功
- [ ] 本地 Ollama 可回复
- [ ] Postgres / Redis / Qdrant 容器可启动（允许 W2 前补齐）
- [ ] Langfuse 账号已建，Key 已备（W4 才接入）
- [ ] 30 道 Python 自测题**全部通过**
- [ ] `check_env.py` 输出全绿

---

## 六、给 Java 背景的加速建议

你 6 年的 Java 经验可以显著加速以下部分，**主动做映射，能省下 10+ 小时**：

| Java 世界 | Python / Agent 世界 | 迁移提示 |
|---|---|---|
| `CompletableFuture` / Reactor | `asyncio` / `TaskGroup` | 概念一致：`await` ≈ `thenCompose`；注意 Python 单线程事件循环，**不要写阻塞代码**（`time.sleep` 会卡死整个循环） |
| `@Transactional` / 状态机 | LangGraph Checkpointer | 编排就是状态机；Checkpointer 就是带版本的持久化快照 |
| Spring `@Bean` / 依赖注入 | 函数式注册表 / `Protocol` | Agent 工具就是"可被 LLM 名字调用的 Bean" |
| Maven/Gradle | uv/poetry | 锁文件思路相同 |
| JUnit + Mockito | pytest + monkeypatch | `monkeypatch` 就是轻量 Mockito |
| SLF4J + MDC TraceId | logging + Langfuse Trace | 全链路追踪思路完全一致，只是 Span 里多了 Token 与 Prompt 版本 |
| Spring Cloud Gateway 熔断限流 | 模型路由 + 熔断 + 限流 | 你的存量经验直接复用，这是你的差异化优势 |

**特别提醒**：Python 里最容易踩的坑是"用 Java 思维写同步阻塞代码"。Agent 系统全是 IO 密集（等模型、等检索、等工具），**异步写对了，延迟直接砍半**，这本身就是一个能在面试里讲的优化点。
