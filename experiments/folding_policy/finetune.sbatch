#!/usr/bin/env bash
#SBATCH --job-name=hp-finetune
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
# Array indexes and wall time are frozen by finetune_launch.py.
set -euo pipefail
HP_FINETUNE_RUN=$1
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_FINETUNE_PYTHON=$(python3 - "$HP_FINETUNE_RUN/finetune.json" <<'PYTHON'
import json, sys
print(json.load(open(sys.argv[1]))['python_executable'])
PYTHON
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_FINETUNE_PYTHON" "$HP_FINETUNE_RUN/bundle/folding_policy/finetune_launch.py" run "$HP_FINETUNE_RUN" --task "$SLURM_ARRAY_TASK_ID"
