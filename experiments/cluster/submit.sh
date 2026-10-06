#!/usr/bin/env bash
# Freeze source/configurations before submitting. No hidden or automatic follow-up jobs.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'Usage: submit.sh pilot|campaign OUT [--pilot-arm cp_sat|rl] [--run-hours H] [--concurrency N] [--threads N] [--include-decisionboost --model-dir DIR] [--dry-run]' >&2; exit 2; }
HP_MODE=$1; HP_OUT=$2; shift 2
source "$HP_SCRIPT_DIR/environment.sh"
HP_ARGS=(); HP_DRY_RUN=0
for HP_ARG in "$@"; do
    if [[ "$HP_ARG" == --dry-run ]]; then HP_DRY_RUN=1; else HP_ARGS+=("$HP_ARG"); fi
done
if [[ $HP_DRY_RUN -eq 0 && -z "${HP_NODE_CONSTRAINT:-}" && -z "${HP_NODELIST:-}" ]]; then
    echo 'Set HP_NODE_CONSTRAINT or one verified EHC hostname in HP_NODELIST before submission.' >&2
    exit 2
fi
if [[ $HP_DRY_RUN -eq 0 && -n "${HP_NODELIST:-}" && -z "${HP_NODE_CONSTRAINT:-}" ]]; then
    HP_NODE_COUNT=$(scontrol show hostnames "$HP_NODELIST" | wc -l)
    [[ "$HP_NODE_COUNT" -eq 1 ]] || { echo 'HP_NODELIST must select one node for these single-node jobs. Use HP_NODE_CONSTRAINT for a homogeneous node pool.' >&2; exit 2; }
fi
"$HP_VENV_PATH/bin/python" "$HP_SCRIPT_DIR/campaign.py" "$HP_MODE" "$HP_OUT" "${HP_ARGS[@]}"
HP_OUT=$(cd "$HP_OUT" && pwd)
HP_VALUES=$("$HP_VENV_PATH/bin/python" - "$HP_OUT" <<'PY'
import json,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'campaign.json').read_text())
print(len(m['tasks']),m['concurrency'],m['threads'],m.get('pilot_arm') or '-')
PY
)
read -r HP_COUNT HP_CONCURRENCY HP_THREADS HP_PILOT_ARM <<< "$HP_VALUES"
HP_OPTIONS=(--parsable --chdir="$HP_OUT" --output="$HP_OUT/logs/%A_%a.out" --error="$HP_OUT/logs/%A_%a.err" --cpus-per-task="$HP_THREADS")
if [[ "$HP_MODE" == pilot && -n "${HP_PILOT_PARTITION:-}" ]]; then
    [[ "$HP_PILOT_PARTITION" == compute || "$HP_PILOT_PARTITION" == computeshort ]] || { echo 'HP_PILOT_PARTITION must be compute or computeshort.' >&2; exit 2; }
    HP_OPTIONS+=(--partition="$HP_PILOT_PARTITION")
fi
if [[ "$HP_MODE" == pilot ]]; then HP_OPTIONS+=(--job-name="hp-$HP_PILOT_ARM-resources"); fi
# A hardware restriction is compulsory for the primary comparison. Feature names
# must come from sinfo; the docs do not publish an EHC Slurm feature string.
if [[ -n "${HP_NODE_CONSTRAINT:-}" ]]; then
    HP_OPTIONS+=(--constraint="$HP_NODE_CONSTRAINT")
elif [[ -n "${HP_NODELIST:-}" ]]; then
    HP_OPTIONS+=(--nodelist="$HP_NODELIST")
elif [[ $HP_DRY_RUN -eq 0 ]]; then
    echo 'Set HP_NODE_CONSTRAINT or HP_NODELIST to a verified homogeneous EHC node selection; inspect sinfo -N -p compute -o "%N %f %c".' >&2
    exit 2
fi
if [[ "$HP_MODE" == campaign ]]; then HP_OPTIONS+=(--array="0-$((HP_COUNT-1))%$HP_CONCURRENCY"); fi
if [[ $HP_DRY_RUN -eq 1 ]]; then
    printf 'Prepared; submission command: '
    printf '%q ' sbatch "${HP_OPTIONS[@]}" "$HP_OUT/bundle/$HP_MODE.sbatch" "$HP_OUT"
    printf '\n'
    exit 0
fi
HP_JOB_ID=$(sbatch "${HP_OPTIONS[@]}" "$HP_OUT/bundle/$HP_MODE.sbatch" "$HP_OUT")
printf '%s\n' "$HP_JOB_ID" > "$HP_OUT/job_id.txt"
echo "Submitted job $HP_JOB_ID. Deadline and all settings are frozen in $HP_OUT/campaign.json"
