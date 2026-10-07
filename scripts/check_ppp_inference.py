#!/usr/bin/env python3
"""One real HDF5 sample: trained PPP inference, seeded sampling, physical marks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import pyrallis
    import torch
    from diffusion.utils.config import SanaConfig
    from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
    from radargen.training.radargen_ppp_dataset import validate_radargen_ppp_dataset_from_config
    from radargen.ppp.sampling import ppp_cell_masses
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/RadarGen_600M_512px_TS_PPP_inference.yaml')
    parser.add_argument('--checkpoint')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sana-checkpoint', default='/e/project1/nxtaim-1/huber7/pretrained/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth')
    parser.add_argument('--null-embed-path', default='output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth')
    parser.add_argument('--text-model-dir', default='/e/project1/nxtaim-1/huber7/pretrained/gemma-2-2b-it')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = pyrallis.load(SanaConfig, stream)
    dataset = validate_radargen_ppp_dataset_from_config(config)
    pipeline = RadarGenPPPPipeline.from_config(config, checkpoint_path=args.checkpoint,
        sana_checkpoint_path=args.sana_checkpoint, null_embed_path=args.null_embed_path,
        text_model_dir=args.text_model_dir, device=args.device, seed=args.seed)
    fields, identity = pipeline.predict_dataset_sample(dataset, args.sample_index)
    points = pipeline.sample_fields(fields)
    pipeline.generator.manual_seed(args.seed)
    repeated = pipeline.sample_fields(fields)
    np.testing.assert_array_equal(points, repeated)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError('Invalid point-cloud output')
    print('Sample identity:', identity, flush=True)
    print('Field shapes:', {k: tuple(v.shape) for k, v in fields.items()}, flush=True)
    print('Expected count:', ppp_cell_masses(fields['log_intensity']).sum().item(), flush=True)
    print('Sampled count:', len(points), flush=True)
    for column, name in ((2, 'rcs'), (3, 'doppler')):
        low, high = config.data.extra[name + '_min'], config.data.extra[name + '_max']
        fraction = float(np.mean((points[:, column] < low) | (points[:, column] > high))) if len(points) else None
        print(name, 'sampled out-of-range fraction:', fraction, '(None means empty sample)', flush=True)
    print('WP6 pretrained inference and seeded sampling passed (structural check only).', flush=True)


if __name__ == '__main__':
    main()
