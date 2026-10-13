#!/usr/bin/env bash
#SBATCH --job-name=hp-policy-pilot
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9%10
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
set -euo pipefail
HP_POLICY_RUN=$1
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load "${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
HP_POLICY_PYTHON=$(python3 - "$HP_POLICY_RUN/pilot.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['python_executable'])
PY
)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpu-bind=cores "$HP_POLICY_PYTHON" "$HP_POLICY_RUN/bundle/folding_policy/launch.py" run "$HP_POLICY_RUN" --task "$SLURM_ARRAY_TASK_ID"
