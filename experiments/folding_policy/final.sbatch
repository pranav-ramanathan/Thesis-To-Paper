#!/usr/bin/env bash
#SBATCH --job-name=hp-final
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
# Array indices, dependencies and group wall limits are frozen by final_launch.py.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: final.sbatch RUN_DIRECTORY GROUP' >&2; exit 2; }
HP_FINAL_RUN=$1
HP_FINAL_GROUP=$2
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_FINAL_PYTHON=$(python3 - "$HP_FINAL_RUN/campaign.json" <<'PYTHON'
import json
import sys

with open(sys.argv[1]) as handle:
    print(json.load(handle)['python_executable'])
PYTHON
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_FINAL_PYTHON" \
    "$HP_FINAL_RUN/bundle/folding_policy/final_launch.py" run "$HP_FINAL_RUN" \
    --group "$HP_FINAL_GROUP" --index "$SLURM_ARRAY_TASK_ID"
