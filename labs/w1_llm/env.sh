# W1 starter 运行配置（source 或复制到你的 shell 配置里）
#
# 用法：
#   cd labs/w1_llm
#   source env.sh
#   uv run python -m w1_llm.bpe

# 1) 只使用 uv 自己管理的 Python，避免 brew 版 / 系统版 3.9 混进来
export UV_PYTHON_PREFERENCE=only-managed

# 2) 缓存放在工作区内：可选，但好处是不污染全局 ~/.cache
#    若你希望用默认全局缓存，把下面这行注释掉即可（你自己的终端有权限）
export UV_CACHE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)/.uv-cache"

# 3) 让 `python` 指向当前 venv，避免误用系统 3.9.6
#    （uv run 会自动处理，这行只是为了你在 shell 里直接敲 python 时也安全）
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

echo "UV_PYTHON_PREFERENCE=$UV_PYTHON_PREFERENCE"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
python -V 2>/dev/null || echo "venv 未激活，请先运行 uv sync"
