#!/usr/bin/env bash
#SBATCH --job-name=hp-cpu-campaign
#SBATCH --partition=compute
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=USR1@600
#SBATCH --no-requeue
set -euo pipefail
HP_CAMPAIGN_DIR=$1
source "$HP_CAMPAIGN_DIR/bundle/environment.sh"
srun --cpu-bind=cores "$HP_VENV_PATH/bin/python" "$HP_CAMPAIGN_DIR/bundle/launch.py" --campaign "$HP_CAMPAIGN_DIR" --task "${SLURM_ARRAY_TASK_ID:?Missing array task id}" "${@:2}"
