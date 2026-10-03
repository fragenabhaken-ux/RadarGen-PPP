"""Direct conditioning-only PPP prediction using RadarGen's joint modality blocks."""

import torch

from diffusion.model.builder import MODELS
from .radargen import RadarGen
from radargen.training.radargen_ppp_training_model import PPP_MODALITIES


class RadarGenPPP(RadarGen):
    def __init__(self, *, in_channels=32, num_maps=3, num_conditions=3,
                 pe_embed_dim=3, fixed_timestep=0.0, pred_sigma=False,
                 learn_sigma=False, patch_size=1, **kwargs):
        if (in_channels, num_maps, num_conditions, pe_embed_dim) != (32, 3, 3, 3):
            raise ValueError("PPP requires 32 latent channels, three conditions, and three 3-channel modality embeddings")
        if fixed_timestep != 0.0 or pred_sigma or learn_sigma:
            raise ValueError("PPP requires fixed timestep 0.0 and a 32-channel output without predicted variance")
        super().__init__(in_channels=in_channels, num_maps=num_maps, num_conditions=num_conditions,
                         pe_embed_dim=pe_embed_dim, pred_sigma=False, learn_sigma=False,
                         patch_size=patch_size, **kwargs)
        self.num_conditions = num_conditions
        self.fixed_timestep = 0.0

    def forward(self, timestep, y, mask=None, data_info=None, bev_condition_maps=None, **kwargs):
        """Three scaled condition latents (B,32,h,w) -> (3B,32,h,w).

        Text/mask/timestep must already have the modality-expanded 3B batch.
        No noisy radar latent or external modality ID is accepted.
        """
        if not isinstance(bev_condition_maps, (list, tuple)) or len(bev_condition_maps) != 3:
            raise ValueError("PPP requires appearance, semantics, velocity condition latents")
        first = bev_condition_maps[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[1] != 32 or any(n == 0 for n in first.shape):
            raise ValueError("Each condition must have nonempty shape (B,32,h,w)")
        if any(not isinstance(c, torch.Tensor) or c.shape != first.shape or c.device != first.device or c.dtype != first.dtype
               for c in bev_condition_maps):
            raise ValueError("Condition shapes/devices must agree")
        batch, _, height, width = first.shape
        expanded_batch = batch * len(PPP_MODALITIES)
        if (not isinstance(timestep, torch.Tensor) or timestep.shape != (expanded_batch,)
                or timestep.device != first.device or not torch.all(timestep == 0)):
            raise ValueError("PPP timestep must be a length-3B tensor of zeros on the condition device")
        if not isinstance(y, torch.Tensor) or y.ndim != 4 or y.shape[:2] != (expanded_batch, 1) or y.device != first.device:
            raise ValueError("PPP text embeddings must have shape (3B,1,L,C) on the condition device")
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim == 0 or mask.shape[0] != expanded_batch or mask.device != first.device):
            raise ValueError("PPP text mask must have batch dimension 3B on the condition device")
        if self.x_embedder.proj.in_channels != 99:
            raise ValueError("Load pretrained SANA weights and inflate the PPP projection before forwarding")
        conditions = torch.cat(bev_condition_maps, dim=1)
        streams = conditions[:, None].expand(-1, self.num_maps, -1, -1, -1)
        modality = self.modality_pe[None, :, :, None, None].expand(batch, -1, -1, height, width)
        spatial_input = torch.cat((streams, modality), dim=2).reshape(expanded_batch, 99, height, width)
        output = self.inner_forward(spatial_input, timestep, y, mask=mask, data_info=data_info, **kwargs)
        if output.shape != (expanded_batch, 32, height, width):
            raise ValueError(f"Unexpected PPP DiT output shape: {tuple(output.shape)}")
        return output


@MODELS.register_module()
def RadarGenPPP_600M_P1_D28(**kwargs):
    return RadarGenPPP(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)
