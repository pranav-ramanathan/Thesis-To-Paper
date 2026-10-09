#!/usr/bin/env bash
# Reuse verified completed seed-0 observations, submit only missing CP/RL tasks.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: rl-cp-comparison.sh COMPLETED_CAMPAIGN NEW_OUT [--dry-run]' >&2; exit 2; }
HP_REUSE_CAMPAIGN=$1
HP_COMPARISON_OUT=$2
shift 2
if [[ $# -eq 1 && "$1" != --dry-run ]]; then
    echo 'Only --dry-run is accepted; this comparison fixes 24 hours, 24 concurrent tasks and eight CPUs.' >&2
    exit 2
fi
exec bash "$HP_SCRIPT_DIR/submit.sh" campaign "$HP_COMPARISON_OUT" \
    --stage rl-vs-cp-sat --reuse-campaign "$HP_REUSE_CAMPAIGN" \
    --run-hours 24 --concurrency 24 --threads 8 "$@"
