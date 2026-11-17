#!/bin/bash
#SBATCH --job-name=SS50_dit_2684032_sweep
#SBATCH --array=0-4%2
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=00:20:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%A_%a_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  bash jsc_jupiter/SS50_dit_2684032_sweep_worker.sh
