#!/usr/bin/env bash
# Submit only the one-hour resource pilot; never launch a campaign afterwards.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: resource-test.sh cp_sat|rl NEW_PERSISTENT_OUTPUT_DIR [--dry-run]' >&2; exit 2; }
HP_PILOT_ARM=$1; HP_PILOT_OUT=$2; shift 2
[[ "$HP_PILOT_ARM" == cp_sat || "$HP_PILOT_ARM" == rl ]] || { echo 'Choose cp_sat or rl.' >&2; exit 2; }
exec bash "$HP_SCRIPT_DIR/submit.sh" pilot "$HP_PILOT_OUT" --pilot-arm "$HP_PILOT_ARM" "$@"
