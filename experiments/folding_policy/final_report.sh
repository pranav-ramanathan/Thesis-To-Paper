#!/usr/bin/env bash
set -euo pipefail
HP_FINAL_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 1 || ( $# -eq 3 && $2 == --recovery ) ]] || {
    echo 'Usage: final_report.sh OUTPUT_DIR [--recovery RECOVERY_DIR]' >&2; exit 2;
}
HP_FINAL_REPORT_RUN=$(cd "$1" && pwd)
HP_FINAL_REPORT_ARGS=("$HP_FINAL_REPORT_RUN")
HP_FINAL_REPORT_OUTPUT="$HP_FINAL_REPORT_RUN/report.md"
if [[ $# -eq 3 ]]; then
    HP_FINAL_RECOVERY_RUN=$(cd "$3" && pwd)
    HP_FINAL_REPORT_ARGS+=(--recovery "$HP_FINAL_RECOVERY_RUN")
    HP_FINAL_REPORT_OUTPUT="$HP_FINAL_RECOVERY_RUN/report/report.md"
fi
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
source "$HP_FINAL_SCRIPT_DIR/../cluster/environment.sh"
"$HP_VENV_PATH/bin/python" "$HP_FINAL_SCRIPT_DIR/final_report.py" "${HP_FINAL_REPORT_ARGS[@]}"
cat "$HP_FINAL_REPORT_OUTPUT"
