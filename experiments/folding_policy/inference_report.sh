#!/usr/bin/env bash
set -euo pipefail
HP_INFERENCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 1 ]] || { echo 'Usage: inference_report.sh OUTPUT_DIR' >&2; exit 2; }
HP_INFERENCE_RUN=$(cd "$1" && pwd)
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_INFERENCE_DIR/../cluster/environment.sh"
"$HP_VENV_PATH/bin/python" "$HP_INFERENCE_DIR/inference_report.py" "$HP_INFERENCE_RUN"
cat "$HP_INFERENCE_RUN/report.md"
