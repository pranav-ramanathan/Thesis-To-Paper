#!/usr/bin/env bash
# Source from the batch script after loading the same Python module as setup.
set -euo pipefail
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load "${HP_PYTHON_MODULE:-python}"
fi
HP_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -n "${HP_CAMPAIGN_DIR:-}" && -z "${HP_VENV_PATH:-}" ]]; then
    HP_CAMPAIGN_PYTHON=$(python3 - "$HP_CAMPAIGN_DIR/campaign.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['python_executable'])
PY
)
    HP_VENV_PATH=$(dirname "$(dirname "$HP_CAMPAIGN_PYTHON")")
fi
HP_VENV_PATH=${HP_VENV_PATH:-$HP_REPO_ROOT/.venv-cluster}
[[ -x "$HP_VENV_PATH/bin/python" ]] || { echo 'Run experiments/cluster/setup.sh first.' >&2; return 1; }
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export HP_VENV_PATH
