#!/bin/bash
#SBATCH --job-name=PPP_overfit_wide_pixelshuffle_5k
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
PPP_CONFIG="configs/PPP_overfit_wide_pixelshuffle_5k.yaml"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
PPP_RESUME_CHECKPOINT="${1:-/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2653347_PPP_overfit_wide_pixelshuffle_1k/checkpoints/epoch_999_step_1000.pth}"
test -f "$PPP_RESUME_CHECKPOINT"
mkdir -p "$PPP_WORK_DIR"
cp "$PPP_CONFIG" "$PPP_WORK_DIR/submitted_config.yaml"
if [[ ! -f output/ppp_manifests/truckscenes_overfit_wide_pixelshuffle.json ]]; then
  python scripts/prepare_ppp_subset_manifest.py --config_path "$PPP_CONFIG" \
    --source output/ppp_manifests/truckscenes_train.json
fi
export PPP_CONFIG PPP_WORK_DIR PPP_RESUME_CHECKPOINT
PPP_STATUS=0
srun --ntasks=1 --cpus-per-task=72 --cpu-bind=verbose --kill-on-bad-exit=1 bash -c '
  nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv -l 2 > "$PPP_WORK_DIR/gpu_usage.csv" &
  monitor_pid=$!
  trap "kill $monitor_pid 2>/dev/null || true" EXIT
  python -u scripts/train_ppp.py --config_path "$PPP_CONFIG" \
    --work_dir "$PPP_WORK_DIR" --resume "$PPP_RESUME_CHECKPOINT" --max-updates 4000
' || PPP_STATUS=$?
# Keep summaries even when training stops early; preserve its exit status.
if [[ -f "$PPP_WORK_DIR/gpu_memory.jsonl" ]]; then
  python scripts/summarize_ppp_memory.py "$PPP_WORK_DIR" | tee "$PPP_WORK_DIR/gpu_memory_summary.txt" || true
fi
if [[ -f "$PPP_WORK_DIR/overfit_diagnostics/metrics.jsonl" ]]; then
  python scripts/summarize_ppp_overfit.py "$PPP_WORK_DIR" | tee "$PPP_WORK_DIR/overfit_summary.txt" || true
fi
exit "$PPP_STATUS"
