#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=hdf5_smoke
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_smoke_%j.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_smoke_%j.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
args=(--config_path jsc_jupiter/truckscenes_hdf5_smoke.yaml)
if [[ -n "${HDF5_STAGING_ROOT:-}" ]]; then
    args+=(--staging_root "$HDF5_STAGING_ROOT")
fi
python /e/project1/nxtaim-1/huber7/pretrained/offline_hdf5_smoke.py "${args[@]}"
