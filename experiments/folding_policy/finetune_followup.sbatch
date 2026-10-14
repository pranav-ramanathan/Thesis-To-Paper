#!/usr/bin/env bash
#SBATCH --job-name=hp-ft-followup
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
# Array and four-hour scheduler limit are frozen by finetune_followup_launch.py.
set -euo pipefail
HP_FOLLOWUP_RUN=$1
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_FOLLOWUP_PYTHON=$(python3 - "$HP_FOLLOWUP_RUN/finetune_followup.json" <<'PYTHON'
import json, sys
print(json.load(open(sys.argv[1]))['python_executable'])
PYTHON
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_FOLLOWUP_PYTHON" "$HP_FOLLOWUP_RUN/bundle/folding_policy/finetune_followup_launch.py" run "$HP_FOLLOWUP_RUN" --task "$SLURM_ARRAY_TASK_ID"
