#!/bin/bash
#SBATCH --job-name=SS50_dino_checkpoint_sweep
#SBATCH --array=0-9
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# Ten unevaluated checkpoints; final step 19914 and baseline reuse matching SS50 results.
STEPS=(10000 11000 12000 13000 14000 15000 16000 17000 18000 19000)
printf -v STEP '%06d' "${STEPS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep"
MANIFEST="evaluation/configs/SS50_dino_2773475_checkpoint_sweep/manifest.json"
REPORT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_comparison.md"
mkdir -p "$ROOT"
update_report() {
  (
    flock 9
    python scripts/compare_ss50_checkpoint_sweep.py --manifest "$MANIFEST" --output "$REPORT"
    cp "$REPORT" docs/SS50_dino_2773475_checkpoint_comparison.md
  ) 9>"$ROOT/comparison.lock"
}
trap update_report EXIT
ARGS=()
if [[ -f "$ROOT/step_$STEP/validation_config.json" ]]; then
  ARGS+=(--resume)
fi
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/validate.py \
    --config "evaluation/configs/SS50_dino_2773475_checkpoint_sweep/step_${STEP}.yaml" \
    --output-dir "$ROOT/step_$STEP" "${ARGS[@]}"
