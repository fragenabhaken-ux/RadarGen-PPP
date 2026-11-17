#!/bin/bash
#SBATCH --job-name=FULL_validation_finetune_step1000
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

VALIDATION_DIR=/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2744861_step1000
ARGS=()
if [[ -f "$VALIDATION_DIR/validation_config.json" ]]; then ARGS+=(--resume); fi
update_report() {
  python3 scripts/compare_finetune_step1000_full.py
}
trap update_report EXIT
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/validate.py \
    --config evaluation/configs/FULL_validation_dino_2744861_step1000.yaml \
    --output-dir "$VALIDATION_DIR" "${ARGS[@]}"
