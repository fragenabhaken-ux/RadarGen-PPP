#!/bin/bash
#SBATCH --job-name=PPP_LR_dit_joint_512
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
DIT_LR_CONFIG="configs/PPP_LR_range_dit_joint_dino_512.yaml"
cp "$DIT_LR_CONFIG" "$DIT_LR_WORK_DIR/submitted_config.yaml"
DIT_LR_STATUS=0
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_ppp.py --config_path "$DIT_LR_CONFIG" \
  --work_dir "$DIT_LR_WORK_DIR" --init-backbone-from "$DIT_LR_BACKBONE" \
  > "$DIT_LR_WORK_DIR/lr_range.out" 2> "$DIT_LR_WORK_DIR/lr_range.err" || DIT_LR_STATUS=$?
python scripts/plot_ppp_lr_range.py --log "$DIT_LR_WORK_DIR/lr_range.err" \
  --output "$DIT_LR_WORK_DIR/lr_range/lr_loss.png" --smoothing-beta 0.95 || true
exit "$DIT_LR_STATUS"
