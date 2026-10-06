#!/usr/bin/env bash
set -euo pipefail
HP_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HP_VENV_PATH=${HP_VENV_PATH:-$HP_REPO_ROOT/.venv-cluster}
if type module >/dev/null 2>&1; then module load "${HP_PYTHON_MODULE:-python}"; fi
HP_SETUP_PYTHON=${HP_SETUP_PYTHON:-python3}
"$HP_SETUP_PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Load a Python 3.11–3.13 module (3.12 recommended)"'
"$HP_SETUP_PYTHON" -m venv "$HP_VENV_PATH"
"$HP_VENV_PATH/bin/python" -m pip install --upgrade pip
"$HP_VENV_PATH/bin/python" -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
"$HP_VENV_PATH/bin/python" -m pip install -r "$HP_REPO_ROOT/experiments/cluster/requirements.txt"
if [[ "${1:-}" == --decisionboost ]]; then
    "$HP_VENV_PATH/bin/python" -m pip install -r "$HP_REPO_ROOT/experiments/cluster/requirements-decisionboost.txt"
elif [[ $# -gt 0 ]]; then
    echo 'Usage: setup.sh [--decisionboost]' >&2; exit 2
fi
"$HP_VENV_PATH/bin/python" -m pip check
"$HP_VENV_PATH/bin/python" -m pip freeze > "$HP_VENV_PATH/installed.txt"
echo "Environment ready: $HP_VENV_PATH"
