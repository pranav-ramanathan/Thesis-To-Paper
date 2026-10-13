#!/usr/bin/env bash
#SBATCH --job-name=hp-policy-alloc
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%3
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
# Wall time and dependencies are set separately for each arm by the launcher.
set -euo pipefail
HP_ALLOCATION_RUN=$1
HP_ALLOCATION_ARM=$2
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_ALLOCATION_PYTHON=$(python3 - "$HP_ALLOCATION_RUN/allocation.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['python_executable'])
PY
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_ALLOCATION_PYTHON" "$HP_ALLOCATION_RUN/bundle/folding_policy/allocation_launch.py" run "$HP_ALLOCATION_RUN" --arm "$HP_ALLOCATION_ARM" --seed "$SLURM_ARRAY_TASK_ID"
