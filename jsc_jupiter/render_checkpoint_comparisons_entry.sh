#!/bin/bash
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
module load FFmpeg/.7.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
for SCENE_TOKEN in bfe5157ddf0a47648a60d3e2737e8868 044c648ac12345f1aedf33c9f91cdc5a 9d1180d52b6a46328063042b8fb325de; do
SCENE_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/$SCENE_TOKEN"
# Render all three comparisons from saved exports; this also renders each DINO overlay.
for MODEL_ID in dino_2773475 dino_2684031_step1000 dino_2773475_step15000; do
  python -u evaluation/scripts/scene_comparison.py render \
      --scene-dir "$SCENE_DIR" --models radargen "$MODEL_ID" \
      --marks doppler rcs heatmap --formats mp4 png --fps 10 \
      --output-dir "$SCENE_DIR/video_radargen_${MODEL_ID}"
done
done
