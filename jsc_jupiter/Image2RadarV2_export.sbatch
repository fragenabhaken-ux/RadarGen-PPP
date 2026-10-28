#!/bin/bash
#SBATCH --job-name=Image2RadarV2_export
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
V2_RUN=/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2557477_Image2RadarV2_overfit
srun --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
  python -u -m scripts.export_camera_overfit \
  --config "$V2_RUN/submitted_config.yaml" \
  --checkpoint "$V2_RUN/checkpoints/epoch_58_step_59.pth" \
  --output-dir "$V2_RUN"
