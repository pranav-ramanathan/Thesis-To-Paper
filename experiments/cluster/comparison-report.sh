#!/usr/bin/env bash
# Regenerate only derived reports; old and new raw artifacts remain separate.
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: comparison-report.sh CAMPAIGN' >&2; exit 2; }
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HP_SCRIPT_DIR/environment.sh"
"$HP_VENV_PATH/bin/python" "$HP_SCRIPT_DIR/analyse.py" "$1"
cat "$1/report.md"
