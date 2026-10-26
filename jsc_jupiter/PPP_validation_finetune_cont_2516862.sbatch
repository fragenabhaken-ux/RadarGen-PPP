#!/bin/bash
#SBATCH --job-name=PPP_validation_finetune_cont_2516862
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# CLI overrides: --checkpoint /path/to/model.pth; --max-samples 3 for a smoke run.
VALIDATION_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
srun --ntasks=1 --cpus-per-task=72 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u evaluation/scripts/validate.py \
    --config evaluation/configs/PPP_validation_finetune_cont_2516862.yaml \
    --output-dir "$VALIDATION_DIR" "$@"
