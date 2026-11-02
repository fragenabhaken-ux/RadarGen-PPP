#!/bin/bash
#SBATCH --job-name=PPP_overfit_dino_attention_1024
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
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
DINO_CONFIG=configs/PPP_overfit_dinov3_attention_1024.yaml
DINO_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
DINO_SOURCE="${1:-/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2391378_PPP_train_dinov3_finetune_2382473/checkpoints/epoch_65_step_6305.pth}"
mkdir -p "$DINO_WORK_DIR"
if [[ ! -f output/ppp_manifests/truckscenes_dino_attention_overfit.json ]]; then
    python scripts/prepare_ppp_subset_manifest.py --config_path "$DINO_CONFIG" \
        --source output/ppp_manifests/truckscenes_train.json
fi
# Driver measurements complement PyTorch allocated/reserved per-update peaks.
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv | tee "$DINO_WORK_DIR/gpu_inventory.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv \
    --loop=5 > "$DINO_WORK_DIR/gpu_memory.csv" &
DINO_MONITOR_PID=$!
trap 'kill "$DINO_MONITOR_PID" 2>/dev/null || true' EXIT
srun --ntasks=1 --cpus-per-task=72 --cpu-bind=verbose --kill-on-bad-exit=1 \
    python -u scripts/train_ppp.py --config_path "$DINO_CONFIG" --work_dir "$DINO_WORK_DIR" \
    --init-dino-from "$DINO_SOURCE" --max-updates 1000 --smoke-check
python scripts/summarize_ppp_overfit.py "$DINO_WORK_DIR" | tee "$DINO_WORK_DIR/overfit_summary.txt"
