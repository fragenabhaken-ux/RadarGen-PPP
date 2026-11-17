#!/bin/bash
#SBATCH --job-name=FULL_ablation_2867524_scenes
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
export MPLCONFIGDIR=/tmp/radargen-full-ablation-${SLURM_JOB_ID}
mkdir -p "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution"
cp "evaluation/configs/FULL_ablation_2867524_bundle.json" "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution/submitted_spec.json"
cp "evaluation/configs/FULL_ablation_2867524_step58057_png.yaml" "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution/submitted_validation.yaml"
cp "$0" "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution/submitted_launcher.sbatch"
srun --ntasks=4 --cpus-per-task=72 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 \
 --output="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution/worker_%t.out" --error="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/execution/worker_%t.err" \
 python -u scripts/run_ablation_full_validation_scenes.py --spec "evaluation/configs/FULL_ablation_2867524_bundle.json"
