#!/bin/bash
#SBATCH --job-name=PPP_train_frozen_decoder
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

# Walltime includes startup and prediction rendering; duration is not measured.

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Six-epoch frozen-decoder training run; uses the existing full training manifest and local assets.
PPP_CONFIG="${1:-configs/PPP_train_frozen_decoder.yaml}"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
test -f "$PPP_CONFIG" || { echo "Missing configuration: $PPP_CONFIG" >&2; exit 1; }

# One Slurm task sees all four allocated GPUs; torchrun starts four DDP workers.
# Do not add --gpus-per-task=1: that would restrict this launcher to one GPU.
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py \
    --config_path "$PPP_CONFIG" \
    --work_dir "$PPP_WORK_DIR"
