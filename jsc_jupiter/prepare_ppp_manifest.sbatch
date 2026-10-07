#!/bin/bash
#SBATCH --job-name=ppp_manifest
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=ppp_manifest_%j.out
#SBATCH --error=ppp_manifest_%j.err

# Use the known booster allocation/environment, not an assumed CPU partition.
# This allocates GPUs but performs only CPU validation with CUDA hidden.
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
export CUDA_VISIBLE_DEVICES=''
# Match the existing module/venv setup without the GPU-detection helper.
source jsc_jupiter/config_jupiter.sh
source jsc_jupiter/modules_jupiter.sh
source "$ENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ENV_DIR/lib/python3.12/site-packages:$REPO_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PPP_CONFIG="${1:-configs/RadarGen_PPP_4GPU_utilization.yaml}"
shift "$(( $# > 0 ? 1 : 0 ))"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --kill-on-bad-exit=1 \
  python -u scripts/prepare_ppp_manifest.py --config_path "$PPP_CONFIG" "$@"
