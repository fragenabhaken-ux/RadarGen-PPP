#!/bin/bash
#SBATCH --job-name=scene_overlay_checkpoint_images
#SBATCH --array=1-3
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# Reuse matching existing exports; add the two intermediate checkpoints.
case "$SLURM_ARRAY_TASK_ID" in
  1) SCENE_TOKEN=bfe5157ddf0a47648a60d3e2737e8868 ;;
  2) SCENE_TOKEN=044c648ac12345f1aedf33c9f91cdc5a ;;
  3) SCENE_TOKEN=9d1180d52b6a46328063042b8fb325de ;;
  *) echo "Unexpected scene index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
SCENE_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/$SCENE_TOKEN"
module load FFmpeg/.7.0.2
ffmpeg -version
srun --ntasks=1 --cpus-per-task=288 --kill-on-bad-exit=1 \
  python -u - "$SCENE_DIR" <<'PYTHON'
import logging
import sys
from evaluation.scene_overlay import render_overlays
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
render_overlays(sys.argv[1], ['dino_2684031_step1000', 'dino_2773475_step15000'],
                sys.argv[1], formats=['mp4', 'png'], fps=10, dpi=110, gamma=.2,
                doppler_limit=30., rcs_limits=[-20., 66.])
PYTHON
