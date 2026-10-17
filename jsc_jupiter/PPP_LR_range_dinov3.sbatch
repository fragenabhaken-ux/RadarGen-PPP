#!/bin/bash
#SBATCH --job-name=PPP_LR_range_dinov3
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

PPP_CONFIG="${1:-configs/PPP_LR_range_dinov3.yaml}"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
PPP_LOG="jsc_jupiter/logs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"

# The config supplies the 2000-update limit; there is no smoke/resume phase.
PPP_TRAIN_STATUS=0
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" \
  || PPP_TRAIN_STATUS=$?

# Keep a loss-vs-LR plot even when the high-LR end stops with a numerical error.
PPP_PLOT_STATUS=0
python -u scripts/plot_ppp_lr_range.py \
  --log "$PPP_LOG" \
  --output "$PPP_WORK_DIR/lr_range/lr_loss.png" \
  --smoothing-beta 0.95 \
  || PPP_PLOT_STATUS=$?

if (( PPP_TRAIN_STATUS != 0 )); then
  exit "$PPP_TRAIN_STATUS"
fi
exit "$PPP_PLOT_STATUS"
