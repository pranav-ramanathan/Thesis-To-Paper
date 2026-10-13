#!/usr/bin/env bash
# Reuse completed training endpoints; only validation inference is submitted.
set -euo pipefail
HP_INFERENCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: inference_pilot.sh SOURCE_ALLOCATION NEW_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
HP_INFERENCE_SOURCE=$1
HP_INFERENCE_OUTPUT=$2
shift 2
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_INFERENCE_DIR/../cluster/environment.sh"
exec "$HP_VENV_PATH/bin/python" "$HP_INFERENCE_DIR/inference_launch.py" prepare "$HP_INFERENCE_OUTPUT" --source-allocation "$HP_INFERENCE_SOURCE" --python "$HP_VENV_PATH/bin/python" "$@"
