#!/bin/bash
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
module load FFmpeg/.7.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
EXPORT_JOB="${1:?Supply the scene-export array job ID}"
SCENE_INDEX=0
while read -r SCENE_TOKEN; do
  SCENE_INDEX=$((SCENE_INDEX+1))
  SCENE_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/$SCENE_TOKEN"
  echo "Waiting for scene $SCENE_INDEX: $SCENE_TOKEN"
  while [[ ! -f "$SCENE_DIR/predictions/radargen.h5" || ! -f "$SCENE_DIR/predictions/dino_2744861_step1000.h5" || ! -f "$SCENE_DIR/predictions/dino_2744861_step1000_dense.h5" ]]; do
    EXPORT_STATE="$(sacct -n -X -j "${EXPORT_JOB}_${SCENE_INDEX}" --format=State --parsable2 | head -1)"
    case "$EXPORT_STATE" in
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|DEPENDENCY*)
        echo "Scene export failed: ${EXPORT_JOB}_${SCENE_INDEX}: $EXPORT_STATE" >&2; exit 1 ;;
      COMPLETED*) echo 'Export completed without required files' >&2; exit 1 ;;
    esac
    sleep 20
  done
  python -u evaluation/scripts/scene_comparison.py render \
    --scene-dir "$SCENE_DIR" --models radargen dino_2744861_step1000 \
    --marks doppler rcs heatmap --formats mp4 --fps 10 \
    --output-dir "$SCENE_DIR/video_radargen_dino_2744861_step1000"
done < jsc_jupiter/scene_extra10_step15000_tokens.txt
echo 'All ten comparisons and overlays completed.'
