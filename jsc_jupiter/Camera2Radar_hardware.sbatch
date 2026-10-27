#!/bin/bash
#SBATCH --job-name=Camera2Radar_hardware
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
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

PPP_CONFIG="${1:-configs/Camera2Radar_hardware.yaml}"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
RAW_ROOT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/cam2radar_raw"
mkdir -p "$PPP_WORK_DIR"

# Prepare all inventory scenes; completed scene fingerprints make this resumable.
python -u -m scripts.prepare_cam2radar_storage --output-root "$RAW_ROOT"

# Device-wide usage includes CUDA/NCCL overhead absent from PyTorch allocations.
nvidia-smi --query-gpu=timestamp,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv -l 15 > "$PPP_WORK_DIR/hardware_gpu.csv" &
CAM2RADAR_MONITOR_PID=$!
trap 'kill "$CAM2RADAR_MONITOR_PID" 2>/dev/null || true' EXIT

srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" --max-updates 200
