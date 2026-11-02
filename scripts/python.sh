# Sourced by the three entrypoints. ADAM_MF_PYTHON can select an installed environment.
if [[ -n "${ADAM_MF_PYTHON:-}" ]]; then
  adam_mf_python="$ADAM_MF_PYTHON"
else
  adam_mf_python=""
  for candidate in python python3 "$HOME/miniconda3/envs/curve/bin/python" "$HOME/anaconda3/envs/curve/bin/python"; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -B -c 'import torch, yaml, opacus, timm, transformers' >/dev/null 2>&1; then
      adam_mf_python="$candidate"
      break
    fi
  done
fi
if [[ -z "$adam_mf_python" ]]; then
  echo '未找到所需 Python 环境。请按 README 安装环境，或设置 ADAM_MF_PYTHON。' >&2
  exit 1
fi
export PYTHONPATH="$adam_mf_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ARROW_DEFAULT_MEMORY_POOL=system CUBLAS_WORKSPACE_CONFIG=:4096:8
