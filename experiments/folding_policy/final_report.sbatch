#!/usr/bin/env bash
# Submit from the repository root; analysis only, no training or search.
#SBATCH --job-name=hp-final-report
#SBATCH --partition=compute
#SBATCH --constraint=ehc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
set -euo pipefail
[[ $# -eq 1 || ( $# -eq 3 && $2 == --recovery ) ]] || {
    echo 'Usage: sbatch final_report.sbatch RUN_DIRECTORY [--recovery RECOVERY_DIR]' >&2; exit 2;
}
HP_FINAL_REPORT_SCRIPT="${SLURM_SUBMIT_DIR:?Submit from the repository root}/experiments/folding_policy/final_report.sh"
[[ -f "$HP_FINAL_REPORT_SCRIPT" ]] || { echo 'Submit this job from the repository root.' >&2; exit 2; }
export HP_PYTHON_MODULE=${HP_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun --ntasks=1 --cpu-bind=cores bash "$HP_FINAL_REPORT_SCRIPT" "$@"
