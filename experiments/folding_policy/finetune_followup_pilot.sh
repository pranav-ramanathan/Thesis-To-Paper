#!/usr/bin/env bash
# Two low-learning-rate controls and longer inference of all saved endpoints.
set -euo pipefail
HP_FOLLOWUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: finetune_followup_pilot.sh SOURCE_FINETUNE NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
HP_FOLLOWUP_SOURCE=$1
HP_FOLLOWUP_OUTPUT=$2
shift 2
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_FOLLOWUP_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_FOLLOWUP_DIR/finetune_followup_launch.py" prepare "$HP_FOLLOWUP_OUTPUT" --source-finetune "$HP_FOLLOWUP_SOURCE" --python "$HP_VENV_PATH/bin/python" "$@"
