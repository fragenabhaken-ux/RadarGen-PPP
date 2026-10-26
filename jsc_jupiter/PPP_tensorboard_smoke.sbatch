#!/bin/bash
#SBATCH --job-name=PPP_tb_smoke
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
case "${1:-}" in
  radargen) PPP_SMOKE_MODEL="$1"; PPP_SMOKE_UPDATES=20 ;;
  dinov3) PPP_SMOKE_MODEL="$1"; PPP_SMOKE_UPDATES=300 ;;
  *) echo 'Usage: sbatch [options] jsc_jupiter/PPP_tensorboard_smoke.sbatch {radargen|dinov3}' >&2; exit 2 ;;
esac
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PPP_CONFIG="configs/PPP_tensorboard_smoke_${PPP_SMOKE_MODEL}.yaml"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${PPP_SMOKE_MODEL}"
# Fresh initialization with the existing pretrained assets; no old training resume.
# The smoke check verifies parameter updates and checkpoint restoration.
srun --ntasks=1 --kill-on-bad-exit=1 python -u scripts/train_ppp.py \
  --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" \
  --max-updates "$PPP_SMOKE_UPDATES" --smoke-check

# Training has closed/flushed its writer before event readback.
python scripts/check_ppp_tensorboard.py "$PPP_WORK_DIR" --expected-step "$PPP_SMOKE_UPDATES"
