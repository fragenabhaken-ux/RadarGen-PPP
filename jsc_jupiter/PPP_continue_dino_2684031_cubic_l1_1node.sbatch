#!/bin/bash
#SBATCH --job-name=PPP_continue_dino_2684031_cubic_l1_1node
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PPP_CONFIG="configs/PPP_continue_dino_2684031_cubic_l1_1node.yaml"
PPP_CHECKPOINT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_45_step_9971.pth"
PPP_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
test -f "$PPP_CHECKPOINT"
mkdir -p "$PPP_WORK_DIR"
cp "$PPP_CONFIG" "$PPP_WORK_DIR/submitted_config.yaml"
# One node, four GPU workers; load model weights only.
export PPP_CONFIG PPP_CHECKPOINT PPP_WORK_DIR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  bash -c 'exec python -u -m torch.distributed.run \
    --nnodes=1 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    scripts/train_ppp.py --config_path "$PPP_CONFIG" --work_dir "$PPP_WORK_DIR" \
    --init-from "$PPP_CHECKPOINT"'
