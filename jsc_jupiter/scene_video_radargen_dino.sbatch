#!/bin/bash
#SBATCH --job-name=scene_video_radargen_dino
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# Existing RadarGen export is reused after provenance checks.
module load FFmpeg/.7.0.2
ffmpeg -version
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/scene_comparison.py predict \
    --config evaluation/configs/scene_comparison_traffic2.yaml \
    --models radargen dino_1024_2658284
srun --ntasks=1 --cpus-per-task=288 --kill-on-bad-exit=1 \
  python -u evaluation/scripts/scene_comparison.py render \
    --scene-dir /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/044c648ac12345f1aedf33c9f91cdc5a \
    --models radargen dino_1024_2658284 \
    --marks doppler rcs heatmap --formats mp4 --fps 10 \
    --output-dir /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/044c648ac12345f1aedf33c9f91cdc5a/video_radargen_dino_2658284
