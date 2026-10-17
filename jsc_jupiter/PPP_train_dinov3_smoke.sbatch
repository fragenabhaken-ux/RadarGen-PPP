#!/bin/bash
#SBATCH --job-name=PPP_dinov3_smoke
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PPP_CONFIG=configs/PPP_train_dinov3_smoke.yaml
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

# Exercise 200 updates with the real model/data/losses and GPU memory logging.
srun --ntasks=1 --kill-on-bad-exit=1 python -u scripts/train_ppp.py \
  --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" \
  --max-updates 200 --smoke-check

# A second process resumes optimizer, scheduler, progress and RNG state.
srun --ntasks=1 --kill-on-bad-exit=1 python -u scripts/train_ppp.py \
  --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" \
  --max-updates 5 --smoke-check --resume latest
