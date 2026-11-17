#!/bin/bash
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
source jsc_jupiter/activate_jupiter.sh
module load FFmpeg/.7.0.2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/resnet-first-standard10 MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec nice -n 10 python -u -m scripts.render_resnet_first_standard10
