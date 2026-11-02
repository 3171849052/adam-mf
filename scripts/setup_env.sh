#!/usr/bin/env bash
set -euo pipefail
adam_mf_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$adam_mf_root"
command -v conda >/dev/null 2>&1 || { echo '请先安装 Conda。' >&2; exit 1; }
if conda env list --json | python3 -c 'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name=="adam-mf" for p in json.load(sys.stdin)["envs"]))'; then
  echo 'adam-mf 已存在：只检查版本，不覆盖或升级现有依赖。'
else
  conda env create --file environment.yml --yes
fi
adam_mf_python="$(conda run --no-capture-output -n adam-mf python -c 'import sys; print(sys.executable)')"
# Fail before installing the project when core dependency versions differ.
"$adam_mf_python" scripts/check_env.py --versions-only
"$adam_mf_python" -m pip install --no-deps --no-build-isolation -e .
"$adam_mf_python" -m pip check
"$adam_mf_python" scripts/check_env.py --output logs/setup_environment_report.json
