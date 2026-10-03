"""Frozen condition encoding and differentiable scalar DC-AE decoding."""

import math

import torch
from torch import nn


def adapt_ppp_decoder_output(decoder, logger):
    """After loading RGB pretrained weights, initialize one channel by averaging."""
    try:
        last = decoder.project_out.op_list[-1]
        old = last.conv
    except (AttributeError, IndexError) as error:
        raise ValueError("Expected DC-AE decoder.project_out.op_list[-1].conv") from error
    if not isinstance(old, nn.Conv2d) or old.out_channels != 3 or old.groups != 1:
        raise ValueError("Expected the configured DC-AE's ungrouped three-channel output Conv2d")
    logger.info("DC-AE output before adaptation: in=%s kernel=%s stride=%s padding=%s dtype=%s device=%s",
                old.in_channels, old.kernel_size, old.stride, old.padding, old.weight.dtype, old.weight.device)
    new = nn.Conv2d(old.in_channels, 1, old.kernel_size, stride=old.stride,
                    padding=old.padding, dilation=old.dilation, groups=old.groups,
                    bias=old.bias is not None, padding_mode=old.padding_mode,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=0, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias.mean().reshape(1))
    last.conv = new
    return new


def _scale(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("DC-AE scaling factor must be positive and finite")
    return value


class PPPDecoder(nn.Module):
    def __init__(self, decoder: nn.Module, scaling_factor: float):
        super().__init__()
        self.decoder = decoder.float().requires_grad_(True)
        self.scaling_factor = _scale(scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # FP32 parameters; caller's BF16 autocast may still govern decoder compute.
        # The cast and division preserve autograd; no detach/no_grad/postprocessing.
        return self.decoder(latent.float() / self.scaling_factor)


class FrozenConditionEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, scaling_factor: float):
        super().__init__()
        self.encoder = encoder.float().requires_grad_(False)
        self.scaling_factor = _scale(scaling_factor)
        self.eval()

    def train(self, mode=True):
        # Keep the frozen encoder in eval even when callers switch modes.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        """Appearance/semantics/velocity images -> scaled latents in that order."""
        if not isinstance(images, (list, tuple)) or len(images) != 3:
            raise ValueError("Three conditioning images are required")
        if any(x.ndim != 4 or x.shape[1] != 3 or x.shape != images[0].shape for x in images):
            raise ValueError("Conditioning images must have matching (B,3,H,W) shapes")
        return [self.encoder(image.float()) * self.scaling_factor for image in images]
