import faulthandler
faulthandler.enable()

def step(name):
    print(f"\n=== {name} ===", flush=True)

step("Base stack")
import numpy
import scipy
import torch
import torchvision
import torchaudio
import triton
import cv2
import mmcv
import diffusers
import transformers

print("numpy:", numpy.__version__, numpy.__file__, flush=True)
print("scipy:", scipy.__version__, flush=True)
print("torch:", torch.__version__, flush=True)
print("torchvision:", torchvision.__version__, flush=True)
print("torchaudio:", torchaudio.__version__, flush=True)
print("triton:", triton.__version__, flush=True)
print("cv2:", cv2.__version__, flush=True)
print("mmcv:", mmcv.__version__, flush=True)
print("diffusers:", diffusers.__version__, flush=True)
print("transformers:", transformers.__version__, flush=True)

step("RadarGen package")
import radargen
print("OK", flush=True)

step("Dataset registry")
from radargen.datasets import get_adapter, list_adapters
print("adapters:", list_adapters(), flush=True)

step("UniDepthV2")
from unidepth.models import UniDepthV2
print("OK", flush=True)

step("UFM")
from uniflowmatch.models.ufm import UniFlowMatchConfidence
print("OK", flush=True)

step("Mask2Former")
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
print("OK", flush=True)

step("RadarGen preprocessing")
from radargen.bev_condition_maps.foundation_models import load_models
print("OK", flush=True)

step("Diffusion builder")
import diffusion.model.builder
print("OK", flush=True)

step("Diffusion optimizer / MMCV")
import diffusion.utils.optimizer
print("OK", flush=True)

step("SANA blocks")
import diffusion.model.nets.sana_blocks
print("OK", flush=True)

step("Training entry point")
import scripts.train
print("OK", flush=True)

print("\n================================", flush=True)
print("FULL RADARGEN IMPORT TEST: OK", flush=True)
print("================================", flush=True)
