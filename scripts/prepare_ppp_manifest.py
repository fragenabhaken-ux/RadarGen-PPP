#!/usr/bin/env python3
"""CPU-only PPP metadata/header validation and atomic manifest preparation."""

import argparse
import os
from pathlib import Path
import sys

# Hide GPUs before importing torch or any dataset dependencies.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default='configs/RadarGen_PPP_4GPU_utilization.yaml')
    parser.add_argument('--rebuild', action='store_true', help='Fully revalidate and atomically replace an existing manifest')
    options = parser.parse_args()
    import pyrallis
    import torch
    from diffusion.utils.config import SanaConfig
    from radargen.training.ppp_manifest import prepare_ppp_manifest
    torch.set_num_threads(1)
    config = pyrallis.parse(SanaConfig, args=['--config_path', options.config_path])
    try:
        prepare_ppp_manifest(config, rebuild=options.rebuild)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'PPP manifest preparation failed: {error}', file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
