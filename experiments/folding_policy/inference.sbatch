#!/usr/bin/env bash
#SBATCH --job-name=hp-policy-infer
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
# Array indexes and wall times are separately frozen for models and baselines.
set -euo pipefail
HP_INFERENCE_RUN=$1
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_INFERENCE_PYTHON=$(python3 - "$HP_INFERENCE_RUN/inference.json" <<'PYTHON'
import json, sys
print(json.load(open(sys.argv[1]))['python_executable'])
PYTHON
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_INFERENCE_PYTHON" "$HP_INFERENCE_RUN/bundle/folding_policy/inference_launch.py" run "$HP_INFERENCE_RUN" --task "$SLURM_ARRAY_TASK_ID"
