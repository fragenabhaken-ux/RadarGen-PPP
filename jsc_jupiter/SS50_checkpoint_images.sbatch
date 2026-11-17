#!/bin/bash
#SBATCH --job-name=SS50_checkpoint_images
#SBATCH --array=1-2
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

case "$SLURM_ARRAY_TASK_ID" in
  1) CONFIG=evaluation/configs/PPP_validation_ss50_4node_step1000_images.yaml ;;
  2) CONFIG=evaluation/configs/PPP_validation_ss50_finetune_step15000_images.yaml ;;
  *) exit 2 ;;
esac
VALIDATION_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/validate.py \
    --config "$CONFIG" \
    --output-dir "$VALIDATION_DIR" "$@"

