#!/bin/bash
#SBATCH --job-name=PPP_LR_dit_joint_1024
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:50:00
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
DIT_LR_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
DIT_LR_BACKBONE="${1:-/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2516862_PPP_train_finetune_cont_2438928/checkpoints/epoch_6_step_12534.pth}"
test -f "$DIT_LR_BACKBONE"
mkdir -p "$DIT_LR_WORK_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv | tee "$DIT_LR_WORK_DIR/gpu_inventory.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$DIT_LR_WORK_DIR/gpu_memory.csv" &
DIT_LR_MONITOR_PID=$!
trap 'kill "$DIT_LR_MONITOR_PID" 2>/dev/null || true' EXIT
python -u scripts/run_dit_joint_lr_range.py \
  --config configs/PPP_LR_range_dit_joint_dino_1024.yaml \
  --work-dir "$DIT_LR_WORK_DIR" --backbone-checkpoint "$DIT_LR_BACKBONE"
