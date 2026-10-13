#!/usr/bin/env bash
# Submit only the frozen two-hour allocation pilot, with shared pretraining.
set -euo pipefail
HP_ALLOCATION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 1 ]] || { echo 'Usage: allocation_pilot.sh NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_ALLOCATION_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_ALLOCATION_DIR/allocation_launch.py" prepare "$@" --python "$HP_VENV_PATH/bin/python"
