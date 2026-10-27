"""Render only final PNGs from a trained Camera2Radar checkpoint."""
import argparse

import pyrallis
import torch

from diffusion.utils.config import SanaConfig
from radargen.cam2radar.dataset import CameraRadarDataset
from radargen.cam2radar.visualization import save_final_predictions
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/Camera2Radar_train.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', help='Override staged raw-data root')
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = pyrallis.load(SanaConfig, stream)
    dataset = CameraRadarDataset(config.data.ppp_manifest_path,
        args.dataset_dir or config.data.extra.get('raw_dataset_dir', config.data.dataset_dir))
    pipeline = RadarGenPPPPipeline.from_config(config, checkpoint_path=args.checkpoint,
                                              device=args.device, seed=config.train.seed)
    for path in save_final_predictions(pipeline.model, dataset, args.indices, args.output_dir,
                                      device=torch.device(args.device), seed=config.train.seed):
        print(path)


if __name__ == '__main__':
    main()
