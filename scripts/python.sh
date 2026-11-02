# Shared interpreter selection. Sourced by entrypoints; never installs packages.
if [[ -n "${ADAM_MF_PYTHON:-}" ]]; then
  adam_mf_python="$ADAM_MF_PYTHON"
  if [[ ! -x "$adam_mf_python" || "$adam_mf_python" != /* ]]; then
    echo 'ADAM_MF_PYTHON 必须指向可执行 Python 的绝对路径。' >&2
    exit 1
  fi
else
  if ! command -v conda >/dev/null 2>&1; then
    echo '未找到 Conda。请安装 Conda 并运行 bash scripts/setup_env.sh，或显式设置 ADAM_MF_PYTHON。' >&2
    exit 1
  fi
  adam_mf_python="$(conda run --no-capture-output -n adam-mf python -c 'import sys; print(sys.executable)' 2>/dev/null)" || {
    echo '未找到可用的 adam-mf 环境，请先运行 bash scripts/setup_env.sh。' >&2
    exit 1
  }
fi
export PYTHONPATH="$adam_mf_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export ARROW_DEFAULT_MEMORY_POOL=system CUBLAS_WORKSPACE_CONFIG=:4096:8
