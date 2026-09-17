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
