#!/bin/bash
#SBATCH --job-name=ablation_dino_neck_compact_resume_4nodes
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ABLATION_CONFIG="configs/ablation_dino_neck_compact_train_4nodes.yaml"
ABLATION_WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$ABLATION_WORK_DIR"
cp "$ABLATION_CONFIG" "$ABLATION_WORK_DIR/submitted_config.yaml"
cp "$0" "$ABLATION_WORK_DIR/submitted_launcher.sbatch"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$ABLATION_WORK_DIR/nodes.txt"
python - "$ABLATION_WORK_DIR" <<'PY'
import json, os, sys
from pathlib import Path
root=Path(sys.argv[1])
(root/'launch_manifest.json').write_text(json.dumps(dict(
    slurm_job_id=os.environ['SLURM_JOB_ID'],model='ablation_dino_neck_compact',
    phase='second_12h_allocation',nodes=4,gpus_per_node=4,world_size=16,
    batch_per_gpu=24,effective_batch=384,accumulation=1,
    max_new_optimizer_updates=15000,warmup_optimizer_updates=0,
    constant_lr=1e-5,training_guard_hours=11.8,allocation_hours=12,
    checkpoint_interval_updates=1000,keep_all_checkpoints=True,
    initialization='exact_resume',source_job=2849062,source_global_step=10372,
    resume_checkpoint='/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2849062_ablation_dino_neck_compact_train_4nodes/checkpoints/epoch_47_step_10372.pth',restore_optimizer=True,restore_scheduler=True,
    automatic_second_submission=False),indent=2)+'\n')
PY
export ABLATION_CONFIG ABLATION_WORK_DIR
MASTER_ADDR=$(head -n 1 "$ABLATION_WORK_DIR/nodes.txt")
export MASTER_ADDR MASTER_PORT=29500
# Production allocation on booster, without the short-test develbooster reservation.
# Exact resume: restore optimizer/scheduler/progress/RNG; warm-up already completed.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 --cpu-bind=verbose --kill-on-bad-exit=1 \
  bash -c 'exec python -u -m torch.distributed.run \
    --nnodes=4 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    scripts/train_ppp.py --config_path "$ABLATION_CONFIG" --work_dir "$ABLATION_WORK_DIR" \
    --resume "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2849062_ablation_dino_neck_compact_train_4nodes/checkpoints/epoch_47_step_10372.pth" --max-updates 15000'
# Intentionally no chained second allocation. Inspect before deciding whether/how to resume.
