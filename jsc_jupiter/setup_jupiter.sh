#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config_jupiter.sh"
source "$SCRIPT_DIR/modules_jupiter.sh"

export PYTHONNOUSERSITE=1

echo
echo "============================================"
echo "RadarGen JUPITER setup"
echo "============================================"
echo "ENV_DIR = $ENV_DIR"
echo

rm -rf "$ENV_DIR"

python -m venv \
    --prompt "$ENV_NAME" \
    --system-site-packages \
    "$ENV_DIR"

source "$ENV_DIR/bin/activate"

VENV_SITE="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

export PYTHONPATH="${VENV_SITE}:${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p \
    "$HF_HOME" \
    "$TORCH_HOME" \
    "$SCRIPT_DIR/logs" \
    "$SCRIPT_DIR/packages" \
    "$SCRIPT_DIR/tmp"

echo
echo "=== Build tooling ==="

python -m pip install --upgrade pip
python -m pip install \
    "setuptools==69.5.1" \
    "wheel==0.43.0"

echo
echo "=== JSC base stack ==="

python - <<'PY'
import platform
import numpy
import scipy
import torch
import torchvision
import torchaudio

print("machine:     ", platform.machine())
print("numpy:       ", numpy.__version__, numpy.__file__)
print("scipy:       ", scipy.__version__)
print("torch:       ", torch.__version__)
print("torchvision: ", torchvision.__version__)
print("torchaudio:  ", torchaudio.__version__)
print("CUDA:        ", torch.version.cuda)
PY

echo
echo "=== Triton 3.5.1 for JUPITER/aarch64 ==="

python -m pip install \
    --no-deps \
    "triton==3.5.1"

echo
echo "=== RadarGen Python dependencies ==="

python -m pip install \
    -c "$SCRIPT_DIR/constraints_jupiter.txt" \
    -r "$SCRIPT_DIR/requirements_jupiter.txt"

echo
echo "=== OpenAI CLIP ==="

python -m pip install \
    --no-deps \
    "git+https://github.com/openai/CLIP.git"

echo
echo "=== utils3d pinned by RadarGen ==="

python -m pip install \
    --no-deps \
    "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

echo
echo "=== MMCV 1.7.2, Python-only/no ops ==="

MMCV_WITH_OPS=0 \
python -m pip install \
    --no-deps \
    --no-build-isolation \
    "mmcv==1.7.2"

echo
echo "=== Patch MMCV 1.7.2 for Python 3.12 ==="

python - <<'PY_MMCV'
from pathlib import Path
import os

path = Path(os.environ["ENV_DIR"]) / "lib/python3.12/site-packages/mmcv/device/npu/data_parallel.py"

text = path.read_text()

old = "for m in sys.modules:"
new = "for m in list(sys.modules):"

if old in text:
    path.write_text(text.replace(old, new))
    print("Patched MMCV for Python 3.12:", path)
elif new in text:
    print("MMCV patch already present:", path)
else:
    raise RuntimeError(f"Expected MMCV line not found in {path}")
PY_MMCV

echo
echo "=== UniDepth ==="

python -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "$REPO_DIR/UniDepth"

echo
echo "=== UniCeption ==="

python -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "$REPO_DIR/UFM/UniCeption"

echo
echo "=== UFM ==="

python -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "$REPO_DIR/UFM"

echo
echo "=== RadarGen ==="

python -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "$REPO_DIR"

echo
echo "============================================"
echo "Installation finished"
echo "============================================"
