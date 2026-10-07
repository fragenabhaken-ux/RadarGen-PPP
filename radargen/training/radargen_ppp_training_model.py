"""Trainable direct PPP forward path; the frozen condition encoder stays outside."""

import torch
from torch import nn

PPP_MODALITIES = ("density", "rcs", "doppler")


class RadarGenPPPTrainingModel(nn.Module):
    def __init__(self, dit, ppp_decoder):
        super().__init__()
        self.dit = dit
        self.ppp_decoder = ppp_decoder

    def forward(self, bev_condition_maps, y, mask=None, data_info=None):
        """Scaled condition latents + 3B empty-text embeddings -> (B,3,1,H,W).
        y is the text-conditioning embedding produced by Gemma from the empty prompt.
        """
        batch = bev_condition_maps[0].shape[0]
        timestep = torch.zeros(len(PPP_MODALITIES) * batch,
                               device=bev_condition_maps[0].device, dtype=torch.float32)
        predicted = self.dit(timestep=timestep, y=y, mask=mask, data_info=data_info,
                             bev_condition_maps=bev_condition_maps)
        decoded = self.ppp_decoder(predicted)
        if decoded.ndim != 4 or decoded.shape[:2] != (len(PPP_MODALITIES) * batch, 1):
            raise ValueError("PPP decoder must return (3B,1,H,W)")
        return decoded.reshape(batch, len(PPP_MODALITIES), 1, *decoded.shape[-2:])
