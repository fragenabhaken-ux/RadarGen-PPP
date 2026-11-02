#!/bin/bash
#SBATCH --job-name=NeckPPP_hardware_b160
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
NECK_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$NECK_WORK_DIR"
export NECK_WORK_DIR
srun --cpus-per-task=72 --cpu-bind=verbose bash -c '
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv -l 1 > "$NECK_WORK_DIR/gpu_memory.csv" &
  monitor_pid=$!
  trap "kill $monitor_pid 2>/dev/null || true" EXIT
  for workers in 4 16 32; do
    echo "Hardware phase: batch=160 workers=$workers"
    python -u scripts/train_ppp.py --config_path "configs/PPP_hardware_neck_b160_w${workers}.yaml" \
      --work_dir "$NECK_WORK_DIR/workers_${workers}" --max-updates 60
  done
'
