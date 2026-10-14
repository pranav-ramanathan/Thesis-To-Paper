#!/usr/bin/env bash
set -euo pipefail
HP_FINETUNE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 1 ]] || { echo 'Usage: finetune_report.sh OUTPUT_DIR' >&2; exit 2; }
HP_FINETUNE_RUN=$(cd "$1" && pwd)
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_FINETUNE_DIR/../cluster/environment.sh"
"$HP_VENV_PATH/bin/python" "$HP_FINETUNE_DIR/finetune_report.py" "$HP_FINETUNE_RUN"
cat "$HP_FINETUNE_RUN/report.md"
