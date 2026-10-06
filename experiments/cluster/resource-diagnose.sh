#!/usr/bin/env bash
# Read existing pilot records and logs; no benchmark or job submission.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 1 ]] || { echo 'Usage: resource-diagnose.sh PILOT_OUTPUT_DIR' >&2; exit 2; }
HP_CAMPAIGN_DIR=$(cd "$1" && pwd)
source "$HP_SCRIPT_DIR/environment.sh"
"$HP_VENV_PATH/bin/python" "$HP_SCRIPT_DIR/resource_pilot.py" --diagnose "$HP_CAMPAIGN_DIR"
