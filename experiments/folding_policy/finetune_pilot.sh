#!/usr/bin/env bash
# One bounded array: five fine-tuning branches and unchanged parent evaluations.
set -euo pipefail
HP_FINETUNE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: finetune_pilot.sh SOURCE_ALLOCATION NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
HP_FINETUNE_SOURCE=$1
HP_FINETUNE_OUTPUT=$2
shift 2
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_FINETUNE_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_FINETUNE_DIR/finetune_launch.py" prepare "$HP_FINETUNE_OUTPUT" --source-allocation "$HP_FINETUNE_SOURCE" --python "$HP_VENV_PATH/bin/python" "$@"
