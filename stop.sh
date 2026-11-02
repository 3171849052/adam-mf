#!/usr/bin/env bash
set -euo pipefail
adam_mf_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$adam_mf_root/scripts/python.sh"
cd "$adam_mf_root"
exec "$adam_mf_python" -B -m adam_mf.cli stop "$@"
