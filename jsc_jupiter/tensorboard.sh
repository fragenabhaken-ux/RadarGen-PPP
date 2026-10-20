#!/usr/bin/env bash
# Usage: bash jsc_jupiter/tensorboard.sh [LOGDIR] [PORT]
# LOGDIR may be one run directory or a root containing multiple runs.
set -euo pipefail
module --force purge
module load Stages/2025 GCCcore/.13.3.0
module load tensorboard

logdir=${1:-/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen}
port=${2:-12356}
if [[ ! -d "$logdir" ]]; then
    echo "Log directory does not exist: $logdir" >&2
    exit 1
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Port must be an integer from 1 to 65535" >&2
    exit 1
fi
exec tensorboard --logdir "$logdir" --port "$port" --host 127.0.0.1
