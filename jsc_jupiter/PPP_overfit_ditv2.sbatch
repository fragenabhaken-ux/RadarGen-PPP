#!/bin/bash
#SBATCH --job-name=DiTV2_overfit
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DITV2_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
python scripts/prepare_ppp_subset_manifest.py --config_path configs/PPP_overfit_ditv2.yaml \
  --source output/ppp_manifests/truckscenes_train.json
mkdir -p "$DITV2_WORK_DIR"
cp configs/PPP_overfit_ditv2.yaml "$DITV2_WORK_DIR/submitted_config.yaml"
srun --cpus-per-task=72 --cpu-bind=verbose python -u scripts/train_ppp.py \
  --config_path configs/PPP_overfit_ditv2.yaml \
  --work_dir "$DITV2_WORK_DIR" --max-updates 200
python scripts/summarize_ppp_overfit.py "$DITV2_WORK_DIR"
