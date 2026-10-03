"""Direct PPP inference using the existing training wrapper and transforms."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from radargen.ppp.sampling import sample_ppp_grid

CONDITION_KEYS = ('bev_color_map', 'bev_seg_map', 'bev_velocity_map')


class RadarGenPPPPipeline:
    def __init__(self, model, condition_encoder, text_provider, *, resolution=512,
                 coordinate_range=50.0, device='cuda', mixed_precision='bf16', seed=42):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.condition_encoder = condition_encoder.to(self.device).eval()
        self.text_provider = text_provider
        self.resolution = resolution
        self.coordinate_range = coordinate_range
        if mixed_precision not in ('bf16', 'fp16', 'no', 'fp32'):
            raise ValueError('Unsupported inference precision')
        self.mixed_precision = mixed_precision
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    @classmethod
    def from_config(cls, config, *, checkpoint_path=None, sana_checkpoint_path,
                    null_embed_path, text_model_dir, device='cuda', seed=42):
        from radargen.training.ppp_initialization import initialize_ppp_model
        from radargen.training.ppp_checkpoint import load_ppp_checkpoint
        from scripts.train_ppp import load_local_text_provider

        # The inference config points at the trained wrapper; the fresh-model
        # initializer must receive the ORIGINAL SANA checkpoint separately.
        trained = checkpoint_path or config.model.load_from
        if not trained or not Path(trained).is_file():
            raise ValueError(f'Trained local PPP checkpoint unavailable: {trained}')
        fresh = deepcopy(config)
        fresh.resume_from = None
        fresh.model.resume_from = None
        fresh.load_from = None
        fresh.model.load_from = str(sana_checkpoint_path)
        fresh.train.grad_checkpointing = False
        components = initialize_ppp_model(fresh, null_embed_path=null_embed_path,
                                         checkpoint_path=sana_checkpoint_path, device=device)
        keys = components.model.state_dict()
        if not all(any(k.startswith(prefix) for k in keys) for prefix in ('dit.', 'ppp_decoder.')):
            raise ValueError('PPP wrapper must contain both DiT and decoder')
        # Existing loader checks all wrapper keys and tensor shapes strictly.
        # No optimizer/scheduler is constructed or restored for inference.
        load_ppp_checkpoint(trained, components.model,
                            resume_optimizer=False, resume_lr_scheduler=False)
        text = load_local_text_provider(config, torch.device(device), text_model_dir)
        return cls(components.model, components.condition_encoder, text,
                   resolution=config.model.image_size,
                   coordinate_range=config.data.extra['coordinate_range'],
                   device=device, mixed_precision=config.model.mixed_precision, seed=seed)

    @torch.no_grad()
    def predict_fields(self, bev_color_map, bev_seg_map, bev_velocity_map, *, data_info=None):
        """Accept already training-transformed tensors (3,H,W) or (1,3,H,W)."""
        maps = []
        for value in (bev_color_map, bev_seg_map, bev_velocity_map):
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if tuple(value.shape) != (1, 3, self.resolution, self.resolution):
                raise ValueError('Expected one transformed RGB condition per modality')
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite conditioning tensor')
            maps.append(value.to(self.device, dtype=torch.float32))
        self.model.eval()
        self.condition_encoder.eval()
        if data_info is None:
            data_info = dict(img_hw=torch.tensor([[self.resolution, self.resolution]], device=self.device),
                             aspect_ratio=torch.ones(1, device=self.device))
        else:
            data_info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in data_info.items()}
        context = (torch.autocast('cuda', dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[self.mixed_precision])
                   if self.device.type == 'cuda' and self.mixed_precision in ('bf16', 'fp16') else nullcontext())
        with context:
            latents = self.condition_encoder(maps)  # no condition dropout
            y, mask = self.text_provider(1, self.device)
            output = self.model(latents, y, mask=mask, data_info=data_info)
        if tuple(output.shape) != (1, 3, 1, self.resolution, self.resolution):
            raise ValueError('Unexpected decoded PPP output shape')
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite predicted fields')
        return {name: output[0, i, 0].float() for i, name in
                enumerate(('log_intensity', 'rcs_raw', 'doppler_raw'))}

    def predict_dataset_sample(self, dataset, index):
        """Use the validated HDF5 dataset's exact training transform and identity."""
        from torch.utils.data import default_collate
        sample = dataset[index]
        batch = default_collate([sample])
        fields = self.predict_fields(*(batch[k] for k in CONDITION_KEYS), data_info=batch['data_info'])
        return fields, sample['data_info']

    def sample_fields(self, fields):
        return sample_ppp_grid(fields['log_intensity'], fields['rcs_raw'],
                               fields['doppler_raw'], self.coordinate_range, self.generator)
