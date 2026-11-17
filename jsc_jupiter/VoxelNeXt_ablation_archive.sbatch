#!/bin/bash
#SBATCH --job-name=VoxelNeXt_ablation
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/%j_%x.err
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen/radar_detection
source jsc_jupiter/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ "$#" -eq 2 ]] || { echo 'Usage: sbatch launcher /absolute/predictions.h5 model_label'; exit 2; }
[[ "$2" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
export BASELINE_ARCHIVE="$1"
export BASELINE_DATA_ROOT=/e/scratch/nxtaim-1/public/truckscenes/man-truckscenes
export BASELINE_OUTPUT="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/${SLURM_JOB_ID}_${2}_z1_flatup"
mkdir -p "$BASELINE_OUTPUT"
cp "$0" "$BASELINE_OUTPUT/submitted_launcher.sbatch"
printf '%s\n' "$BASELINE_ARCHIVE" > "$BASELINE_OUTPUT/input_archive.txt"
srun --ntasks=4 --cpus-per-task=8 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
 bash -c 'set -euo pipefail
 rank="$SLURM_LOCALID"
 IFS=, read -r -a allocated <<< "${SLURM_STEP_GPUS:-$CUDA_VISIBLE_DEVICES}"
 if (( ${#allocated[@]} > 1 )); then export CUDA_VISIBLE_DEVICES="${allocated[$rank]}"; fi
 echo "Detector rank $SLURM_PROCID on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
 exec python -m radar_eval.run --archive "$BASELINE_ARCHIVE" --data-root "$BASELINE_DATA_ROOT" \
 --side generated --height 1 --height-frame flat-up --seed 42 \
 --start "$((SLURM_PROCID * 477))" --max-frames 477 \
 --output "$BASELINE_OUTPUT/shard_${SLURM_PROCID}.h5"'
srun --ntasks=1 --gpus-per-task=1 python -m radar_eval.merge --archive "$BASELINE_ARCHIVE" \
 --output "$BASELINE_OUTPUT/detections.h5" "$BASELINE_OUTPUT"/shard_*.h5
srun --ntasks=1 --gpus-per-task=1 python -m radar_eval.score --archive "$BASELINE_ARCHIVE" \
 --detections "$BASELINE_OUTPUT/detections.h5" --data-root "$BASELINE_DATA_ROOT" \
 --output "$BASELINE_OUTPUT/metrics.json"
