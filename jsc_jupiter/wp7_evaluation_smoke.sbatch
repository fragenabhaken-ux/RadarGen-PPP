#!/bin/bash
#SBATCH --job-name=ppp_wp7
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=wp7_evaluation_%j.out
#SBATCH --error=wp7_evaluation_%j.err
set -e
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun --cpus-per-task=72 --cpu-bind=verbose python -u evaluation/scripts/check_ppp_evaluation.py
