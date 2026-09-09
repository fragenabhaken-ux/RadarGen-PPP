#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=radargen_import
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/full_import_%j.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/RadarGen/jsc_jupiter/logs/full_import_%j.err

set -euo pipefail

cd /e/project1/nxtaim-1/huber7/repos/RadarGen

source jsc_jupiter/activate_jupiter.sh

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V

python -u jsc_jupiter/test_full_imports.py
