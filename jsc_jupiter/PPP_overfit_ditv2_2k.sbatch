#!/bin/bash
#SBATCH --job-name=DiTV2_overfit_2k
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DITV2_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$DITV2_WORK_DIR"
cp configs/PPP_overfit_ditv2_2k.yaml "$DITV2_WORK_DIR/submitted_config.yaml"
srun --cpus-per-task=72 --cpu-bind=verbose python -u scripts/train_ppp.py \
  --config_path configs/PPP_overfit_ditv2_2k.yaml \
  --work_dir "$DITV2_WORK_DIR" --max-updates 1800 --final-pngs-only \
  --resume /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2570275_DiTV2_overfit/checkpoints/epoch_199_step_200.pth
