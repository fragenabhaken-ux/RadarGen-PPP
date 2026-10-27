"""One pretrained forward/backward diagnostic; full resolution requires a GPU.

The default CPU mode uses small synthetic images/output to check integration,
not memory capacity or spatial learning. --full uses native TruckScenes shapes.
"""
import argparse
import json
import time

import torch
import yaml

from radargen.cam2radar.model import Camera2Radar
from radargen.losses import ContinuousPPPLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/Camera2Radar_train.yaml')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = yaml.safe_load(open(args.config))
    kwargs = dict(settings['model']['extra'])
    kwargs['grad_checkpointing'] = True
    device = torch.device('cuda' if args.full else 'cpu')
    if args.full and not torch.cuda.is_available():
        raise RuntimeError('Full-resolution diagnostic requires an allocated GPU')
    height, width = (943, 1980) if args.full else (64, 64)
    kwargs['output_size'] = 512 if args.full else 16
    model = Camera2Radar(**kwargs).to(device).train()
    images = torch.randint(0, 256, (1, 4, 3, height, width), dtype=torch.uint8, device=device)
    intrinsics = torch.tensor([[width, 0., width/2], [0., width, height/2], [0., 0., 1.]], device=device).repeat(1, 4, 1, 1)
    transforms = torch.eye(4, device=device).repeat(1, 4, 1, 1)
    transforms[0, :, 0, 3] = torch.tensor([0., 1., -1., 2.], device=device)
    if args.full:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.full):
        fields = model(images, intrinsics, transforms)
    loss = ContinuousPPPLoss(out_img_size=fields.shape[-2:])(fields, torch.tensor([[0., 0., 1., -1.]], device=device),
                               torch.tensor([0], device=device), point_limit=50.)
    total = sum(loss.values())
    total.backward()
    if not torch.isfinite(total) or model.neck.layers[0].depth.weight.grad.norm() == 0:
        raise RuntimeError('Invalid loss or absent learned-geometry gradients')
    print(json.dumps(dict(mode='full' if args.full else 'small-synthetic',
        output_shape=list(fields.shape), loss=float(total.detach()),
        seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.full else None)))


if __name__ == '__main__':
    main()
