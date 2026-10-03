#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-479%4
#SBATCH --job-name=hdf5_all
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_all_%A_%a.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/hdf5_all_%A_%a.err

set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
scene=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jsc_jupiter/hdf5_scene_tokens.txt)
[[ "$scene" =~ ^[0-9a-f]{32}$ ]] || { echo "Invalid scene token: $scene" >&2; exit 1; }
stage="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/staging/hdf5_full/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
echo "Scene: $scene | Task: $SLURM_ARRAY_TASK_ID | Staging: $stage"

python - "$scene" "$stage" <<'PY'
import importlib.util
import sys
from pathlib import Path

scene, staging = sys.argv[1:]
wrapper = Path("/e/project1/nxtaim-1/huber7/pretrained/offline_hdf5_smoke.py")
spec = importlib.util.spec_from_file_location("offline_models", wrapper)
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)

import jsc_jupiter.create_hdf5_dataset_scene as production

# Reuse the working offline setup, dispatching to the production runner.
def run_production(path, run_name):
    production.main()

offline.runpy.run_path = run_production
sys.argv = [
    str(wrapper),
    "--config_path", "jsc_jupiter/truckscenes_hdf5_all.yaml",
    "--scene_token", scene,
    "--staging_root", staging,
]
offline.main()
PY

rmdir "$stage"
echo "SUCCESS: $scene"
