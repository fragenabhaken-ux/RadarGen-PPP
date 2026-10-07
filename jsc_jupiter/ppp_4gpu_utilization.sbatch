#!/bin/bash
#SBATCH --job-name=ppp_4gpu_util
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=ppp_4gpu_util_%j.out
#SBATCH --error=ppp_4gpu_util_%j.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Save the utilization YAML at this path, or pass its path to sbatch.
PPP_CONFIG="${1:-configs/RadarGen_PPP_4GPU_utilization.yaml}"
PPP_WORK_DIR="output/RadarGen_PPP_4GPU_utilization_${SLURM_JOB_ID}"
test -f "$PPP_CONFIG" || { echo "Missing configuration: $PPP_CONFIG" >&2; exit 1; }

# One Slurm task sees all four allocated GPUs; torchrun starts four DDP workers.
# Do not add --gpus-per-task=1: that would restrict this launcher to one GPU.
srun --ntasks=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train_ppp.py \
    --config_path "$PPP_CONFIG" \
    --work_dir "$PPP_WORK_DIR" \
    --max-updates 1000
