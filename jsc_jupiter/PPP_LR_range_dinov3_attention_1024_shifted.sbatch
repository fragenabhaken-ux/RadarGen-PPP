#!/bin/bash
#SBATCH --job-name=PPP_LR_dino_shifted_b24
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
DINO_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
DINO_SOURCE="${1:-/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2391378_PPP_train_dinov3_finetune_2382473/checkpoints/epoch_65_step_6305.pth}"
mkdir -p "$DINO_WORK_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv | tee "$DINO_WORK_DIR/gpu_inventory.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv \
    --loop=5 > "$DINO_WORK_DIR/gpu_memory.csv" &
DINO_MONITOR_PID=$!
trap 'kill "$DINO_MONITOR_PID" 2>/dev/null || true' EXIT
# Separate process: probe updates never enter the LR-range model or optimizer.
echo "Validating batch 24/GPU with five DDP optimizer updates; logs: $DINO_WORK_DIR/batch_probe.err"
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py --config_path configs/PPP_batch_probe_dinov3_attention_1024_shifted.yaml \
    --work_dir "$DINO_WORK_DIR/batch_probe" --init-dino-from "$DINO_SOURCE" --max-updates 5 \
    > "$DINO_WORK_DIR/batch_probe.out" 2> "$DINO_WORK_DIR/batch_probe.err"
python - "$DINO_WORK_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads(s) for s in (root / 'batch_probe/gpu_memory.jsonl').read_text().splitlines() if s]
rows = [r for r in rows if r['phase'] == 'train']
if len(rows) != 5:
    raise ValueError(f'Expected five successful probe updates, got {len(rows)}')
summary = dict(batch_per_gpu=24, world_size=4, effective_batch=96,
               peak_allocated_gib=max(r['gpu_peak_allocated_gib'] for r in rows),
               peak_reserved_gib=max(r['gpu_peak_reserved_gib'] for r in rows),
               gpu_total_gib=min(r['gpu_total_gib'] for r in rows))
(root / 'batch_probe_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('DDP batch probe passed:', json.dumps(summary), flush=True)
PY
echo "Starting fresh 600-update LR sweep, 1e-7 -> 1e-2; logs: $DINO_WORK_DIR/lr_range.err"
DINO_TRAIN_STATUS=0
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py --config_path configs/PPP_LR_range_dinov3_attention_1024_shifted.yaml \
    --work_dir "$DINO_WORK_DIR/sweep" --init-dino-from "$DINO_SOURCE" \
    > "$DINO_WORK_DIR/lr_range.out" 2> "$DINO_WORK_DIR/lr_range.err" || DINO_TRAIN_STATUS=$?
DINO_PLOT_STATUS=0
python scripts/plot_ppp_lr_range.py --log "$DINO_WORK_DIR/lr_range.err" \
    --output "$DINO_WORK_DIR/lr_range/lr_loss.png" --smoothing-beta 0.95 || DINO_PLOT_STATUS=$?
if (( DINO_TRAIN_STATUS != 0 )); then
    exit "$DINO_TRAIN_STATUS"
fi
exit "$DINO_PLOT_STATUS"
