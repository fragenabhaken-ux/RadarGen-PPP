#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=hdf5_launch_check
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_launch_check_%j.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_launch_check_%j.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset CUDA_VISIBLE_DEVICES
results=$(mktemp -d "$PWD/jsc_jupiter/hdf5_launch_check_${SLURM_JOB_ID}_XXXXXX")
echo "Launch-check records: $results"
srun --cpus-per-task=72 --cpu-bind=verbose --kill-on-bad-exit=0 \
    --output="jsc_jupiter/hdf5_launch_check_%j_worker_%t.out" \
    --error="jsc_jupiter/hdf5_launch_check_%j_worker_%t.err" \
    python -u jsc_jupiter/hdf5_4nodes_launch_check.py --result-dir "$results"
python -u jsc_jupiter/hdf5_4nodes_launch_check.py --result-dir "$results" --verify
