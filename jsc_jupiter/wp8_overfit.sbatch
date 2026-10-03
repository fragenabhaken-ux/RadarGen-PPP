#!/bin/bash
#SBATCH --job-name=ppp_wp8_overfit
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=wp8_overfit_%j.out
#SBATCH --error=wp8_overfit_%j.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
WP8_WORK_DIR="output/RadarGen_PPP_WP8_overfit_${SLURM_JOB_ID}"
srun --cpus-per-task=72 --cpu-bind=verbose python -u scripts/train_ppp.py \
  --config_path configs/RadarGen_PPP_WP8_overfit.yaml \
  --work_dir "$WP8_WORK_DIR" --max-updates 200
python scripts/summarize_ppp_overfit.py "$WP8_WORK_DIR"
