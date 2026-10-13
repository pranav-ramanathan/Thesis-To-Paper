#!/usr/bin/env bash
# Freeze and submit only the validation pilot; no subsequent campaign is launched.
set -euo pipefail
HP_POLICY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 1 ]] || { echo 'Usage: pilot.sh NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_POLICY_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_POLICY_DIR/launch.py" prepare "$@" --python "$HP_VENV_PATH/bin/python"
