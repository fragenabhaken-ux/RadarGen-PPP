#!/bin/bash
#SBATCH --job-name=FULL_validation
#SBATCH --array=0-4
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# Each task evaluates one checkpoint; no inference is run on entry nodes.
MODELS=(radargen dino_2684031 dino_2744861 dit_2684032 dit_2744862)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108"
REPORT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_metrics_comparison.md"
mkdir -p "$ROOT"
update_report() {
  flock "$ROOT/comparison.lock" python scripts/compare_full_validation.py --root "$ROOT" --output "$REPORT"
}
trap update_report EXIT
ARGS=()
if [[ -f "$ROOT/$MODEL/validation_config.json" ]]; then
  ARGS+=(--resume)
fi
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/validate.py \
    --config "evaluation/configs/FULL_validation_${MODEL}.yaml" \
    --output-dir "$ROOT/$MODEL" "${ARGS[@]}"
