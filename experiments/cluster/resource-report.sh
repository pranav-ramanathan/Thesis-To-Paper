#!/usr/bin/env bash
# Run on the login node after the pilot finishes: reporting only, no models.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 1 ]] || { echo 'Usage: resource-report.sh PILOT_OUTPUT_DIR' >&2; exit 2; }
HP_CAMPAIGN_DIR=$(cd "$1" && pwd)
source "$HP_SCRIPT_DIR/environment.sh"
# The current reader verifies frozen experiment sources, then regenerates derived
# reports. It never edits the frozen bundle or its original measurement records.
"$HP_VENV_PATH/bin/python" "$HP_SCRIPT_DIR/resource_pilot.py" --report "$HP_CAMPAIGN_DIR"
if [[ -r "$HP_CAMPAIGN_DIR/job_id.txt" ]]; then
    HP_JOB_ID=$(cat "$HP_CAMPAIGN_DIR/job_id.txt")
    # Step rows matter: allocation-level MaxRSS is often blank. Keep .batch/.0.
    if sacct -j "$HP_JOB_ID" --units=G --parsable2 \
        --format=JobID,State,ExitCode,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem,NodeList \
        > "$HP_CAMPAIGN_DIR/accounting.txt"; then
        cat "$HP_CAMPAIGN_DIR/accounting.txt"
    else
        echo 'Slurm accounting unavailable; retry this report command later.' >&2
    fi
fi
