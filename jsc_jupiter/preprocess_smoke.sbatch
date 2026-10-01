#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=radargen_pre_smoke
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/preprocess_smoke_%j.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/preprocess_smoke_%j.err

set -e

cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh

echo "=== NODE ==="
hostname
nvidia-smi

echo
echo "=== RADAR MAP SMOKE TEST ==="
python jsc_jupiter/create_radar_maps_smoke.py \
    --config_path jsc_jupiter/truckscenes_radar_maps_smoke.yaml

echo
echo "=== BEV CONDITION MAP SMOKE TEST ==="
python jsc_jupiter/create_bev_condition_maps_smoke.py \
    --config_path jsc_jupiter/truckscenes_bev_condition_maps_smoke.yaml

echo
echo "=== GENERATED FILES ==="
find /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/preprocessing/smoke \
    -type f | head -50

echo
echo "=== DISK USAGE ==="
du -sh /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/preprocessing/smoke/*

echo
echo "RADARGEN PREPROCESSING SMOKE TEST FINISHED"
