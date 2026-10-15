#!/usr/bin/env bash
# A separate frozen recovery; the original campaign remains unchanged.
set -euo pipefail
HP_RECOVERY_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: final_recovery.sh SOURCE_RUN NEW_RECOVERY_DIRECTORY [--dry-run]' >&2; exit 2; }
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_RECOVERY_SCRIPT_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_RECOVERY_SCRIPT_DIR/final_recovery.py" prepare "$@"
