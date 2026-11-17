#!/usr/bin/env bash
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 10 python -u scripts/ss50_ablation_count_sweeps.py --workers 24 \
 --specs-json output/paper_material/count_sweep_evaluation/rn50_sweep_specs.json \
 --usage-output output/paper_material/count_sweep_evaluation/rn50_count_sweep_resource_usage.json
