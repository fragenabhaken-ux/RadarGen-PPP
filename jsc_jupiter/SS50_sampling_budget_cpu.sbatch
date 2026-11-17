#!/bin/bash
#SBATCH --job-name=SS50_sampling_budget_cpu
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ss50_sampling_budget_%j.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ss50_sampling_budget_%j.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/ss50_sampling_budget.py --output output/ss50_sampling_budget --threads 1 --workers 48
