#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=radargen_wp5
#SBATCH --output=wp5_training_%j.out
#SBATCH --error=wp5_training_%j.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
# Separate output; three updates (six microbatches), then three resumed updates.
# Preserves optimizer/schedule types, loss weights and conditioning dropout.
# Shorten warmup to two updates only for this three-update diagnostic.
WP5_WORK_DIR="output/RadarGen_PPP_WP5_smoke_${SLURM_JOB_ID}"
srun python scripts/train_ppp.py \
  --config_path configs/RadarGen_600M_512px_TS_PPP_training.yaml \
  --work_dir "$WP5_WORK_DIR" --report_to none \
  --train.train_batch_size 1 --train.num_workers 0 \
  --train.log_interval 1 --train.lr_schedule_args '{"num_warmup_steps": 2}' \
  --max-updates 3 --smoke-check
srun python scripts/train_ppp.py \
  --config_path configs/RadarGen_600M_512px_TS_PPP_training.yaml \
  --work_dir "$WP5_WORK_DIR" --report_to none \
  --train.train_batch_size 1 --train.num_workers 0 \
  --train.log_interval 1 --train.lr_schedule_args '{"num_warmup_steps": 2}' \
  --max-updates 3 --smoke-check --resume latest
