#!/usr/bin/env bash
# Nine fresh primary learning/search runs; no resource profiling or follow-up campaign.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: seed0-test.sh NEW_OUT [--dry-run]' >&2; exit 2; }
HP_SEED0_OUT=$1
shift
if [[ $# -eq 1 && "$1" != --dry-run ]]; then
    echo 'Only --dry-run is accepted; this stage fixes 24 hours, nine tasks and eight CPUs.' >&2
    exit 2
fi
exec bash "$HP_SCRIPT_DIR/submit.sh" campaign "$HP_SEED0_OUT" \
    --stage seed0-feasibility --run-hours 24 --concurrency 9 --threads 8 "$@"
