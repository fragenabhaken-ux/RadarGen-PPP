#!/bin/bash
#SBATCH --job-name=ablation_resnet50_backbone_lr_range
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:45:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ABLATION_RECOMMENDATION="${1:?Pass the entry-probe batch_recommendation.json}"
ABLATION_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$ABLATION_WORK_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$ABLATION_WORK_DIR/gpu_inventory.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv --loop=1 > "$ABLATION_WORK_DIR/gpu_memory.csv" &
ABLATION_MONITOR_PID=$!
trap 'kill "$ABLATION_MONITOR_PID" 2>/dev/null || true' EXIT
python -u scripts/run_ablation_lr_range.py \
    --config configs/ablation_resnet50_backbone_lr_range.yaml \
    --recommendation "$ABLATION_RECOMMENDATION" --work-dir "$ABLATION_WORK_DIR"
