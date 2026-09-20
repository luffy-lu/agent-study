cd ~/ai/agent-study/labs/w0_env      # 或 labs/w1_llm，每个工程独立
uv run python check_env.py           # ← 最常用：跑脚本
uv run python                        # ← 进交互式 REPL
uv run pytest -v                     # ← 跑测试
uv run python -m w1_llm.bpe          # ← 跑模块