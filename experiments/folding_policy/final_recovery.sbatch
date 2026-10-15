#!/usr/bin/env bash
#SBATCH --job-name=hp-final-recover
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: final_recovery.sbatch RECOVERY_DIRECTORY' >&2; exit 2; }
HP_RECOVERY_DIR=$1
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_RECOVERY_PYTHON=$(python3 - "$HP_RECOVERY_DIR/recovery.json" <<'PYTHON'
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
srun --ntasks=1 --cpu-bind=cores "$HP_RECOVERY_PYTHON" \
    "$HP_RECOVERY_DIR/bundle/final_recovery.py" run "$HP_RECOVERY_DIR" --index "$SLURM_ARRAY_TASK_ID"
