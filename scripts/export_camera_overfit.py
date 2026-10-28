"""Export fixed-frame prediction PNGs from a trained raw-camera PPP checkpoint."""
import argparse
import logging
from pathlib import Path


def main():
    import pyrallis
    import torch
    from accelerate import Accelerator
    from torch.utils.data import Subset
    from diffusion.utils.config import SanaConfig
    from radargen.cam2radar.dataset import CameraRadarDataset
    from radargen.losses.ppp_loss import PPPLoss
    from radargen.training.ppp_checkpoint import load_ppp_checkpoint
    from radargen.training.ppp_initialization import initialize_dino_ppp_model
    from radargen.training.ppp_overfit import diagnose_overfit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.config) as stream:
        config = pyrallis.load(SanaConfig, stream)
    if config.model.model not in ('Camera2Radar', 'Image2RadarV2'):
        raise ValueError('Expected a raw-camera model')
    config.work_dir = str(Path(args.output_dir).resolve())
    config.train.grad_checkpointing = False
    accelerator = Accelerator(mixed_precision=config.model.mixed_precision)
    if accelerator.num_processes != 1:
        raise ValueError('Run export with one process/GPU')
    model = initialize_dino_ppp_model(config, device='cpu', load_pretrained=False).model
    _, _, _, rng = load_ppp_checkpoint(args.checkpoint, model,
                                      resume_optimizer=False, resume_lr_scheduler=False)
    step = rng['ppp_training']['progress']['global_step']
    model = model.to(accelerator.device).eval()
    dataset = CameraRadarDataset(config.data.ppp_manifest_path, config.data.extra['raw_dataset_dir'])
    for key, value in dataset.contract['normalization'].items():
        if config.data.extra.get(key) != value:
            raise ValueError(f'Normalization differs from manifest: {key}')
    selected = config.data.extra.get('ppp_subset_indices')
    if not selected:
        raise ValueError('Export requires an explicit fixed subset')
    dataset = Subset(dataset, selected)
    torch.manual_seed(config.train.seed)
    diagnose_overfit(model, None, dataset, None, PPPLoss(), config, accelerator,
                     step, indices=list(range(len(dataset))))
    logging.info('Exported %d frame PNGs to %s/prediction_images', len(dataset), config.work_dir)


if __name__ == '__main__':
    main()
