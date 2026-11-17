#!/bin/bash
#SBATCH --job-name=EVAL_all_random
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/radargen-eval-programme-${SLURM_JOB_ID}
mkdir -p "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/random"
cp "evaluation/configs/all_ablation_checkpoint_evaluations_20261114.json" "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/random/submitted_manifest_${SLURM_JOB_ID}.json"
cp "$0" "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/random/submitted_launcher_${SLURM_JOB_ID}.sbatch"
srun --ntasks=4 --cpus-per-task=72 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 \
 --output="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/random/job_${SLURM_JOB_ID}_worker_%t.out" --error="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/random/job_${SLURM_JOB_ID}_worker_%t.err" \
 python -u scripts/run_ablation_checkpoint_programme.py --manifest "evaluation/configs/all_ablation_checkpoint_evaluations_20261114.json" --model-index 2
