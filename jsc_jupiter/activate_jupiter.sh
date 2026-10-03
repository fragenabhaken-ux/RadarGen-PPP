#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config_jupiter.sh"
source "$SCRIPT_DIR/modules_jupiter.sh"

source "$ENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1

VENV_SITE="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

export TORCH_HOME=/e/project1/nxtaim-1/huber7/pretrained/torch
export HF_HOME=/e/project1/nxtaim-1/huber7/pretrained/huggingface

export PYTHONPATH="${VENV_SITE}:${REPO_DIR}:${PYTHONPATH:-}"

CUDA_HOME_DETECTED="$(python - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME or "")
PY
)"

if [[ -n "$CUDA_HOME_DETECTED" ]]; then
    export CUDA_HOME="$CUDA_HOME_DETECTED"
fi

mkdir -p "$HF_HOME" "$TORCH_HOME"

echo
echo "Activated $ENV_NAME"
echo "ENV_DIR   = $ENV_DIR"
echo "REPO_DIR  = $REPO_DIR"
echo "CUDA_HOME = ${CUDA_HOME:-not detected}"
echo "HF_HOME   = $HF_HOME"
echo "TORCH_HOME = $TORCH_HOME"
echo
which python
python -V
