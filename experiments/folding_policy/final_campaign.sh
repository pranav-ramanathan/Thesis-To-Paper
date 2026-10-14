#!/usr/bin/env bash
# Freeze and submit the final held-out comparison from the cluster terminal.
set -euo pipefail
HP_FINAL_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 1 ]] || { echo 'Usage: final_campaign.sh NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_FINAL_SCRIPT_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_FINAL_SCRIPT_DIR/final_launch.py" prepare "$@" --python "$HP_VENV_PATH/bin/python"
