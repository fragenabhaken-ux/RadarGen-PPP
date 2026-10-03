#!/bin/bash
#SBATCH --job-name=ppp_wp6
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=wp6_inference_%j.out
#SBATCH --error=wp6_inference_%j.err
set -e
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun --cpus-per-task=72 --cpu-bind=verbose python -u scripts/check_ppp_inference.py
