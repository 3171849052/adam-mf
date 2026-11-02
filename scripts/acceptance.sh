#!/usr/bin/env bash
# Engineering acceptance only: never starts formal training.
set -euo pipefail
adam_mf_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$adam_mf_root/scripts/python.sh"
cd "$adam_mf_root"
"$adam_mf_python" scripts/check_config.py
./download_assets.sh
./prepare_data.sh
./verify_assets.sh
./check.sh
"$adam_mf_python" -m adam_mf.cli simulate
./start.sh --dry-run --wait
