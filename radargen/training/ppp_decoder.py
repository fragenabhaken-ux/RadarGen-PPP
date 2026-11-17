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
        # Scale in FP32, then match autocast so residual shortcuts do not promote
        # the decoder's feature maps back to FP32. Parameters remain FP32.
        latent = latent.float() / self.scaling_factor
        if torch.is_autocast_enabled(latent.device.type):
            latent = latent.to(torch.get_autocast_dtype(latent.device.type))
        return self.decoder(latent)


def adapt_ppp_decoder_input(decoder, in_channels, logger):
    """Replace only the native 32→1024 residual input with a full-width 3×3 conv."""
    from diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import build_decoder_project_in_block
    from diffusion.model.dc_ae.efficientvit.models.nn.ops import (
        ChannelDuplicatingPixelUnshuffleUpSampleLayer, ConvLayer, ResidualBlock,
    )
    block = decoder.project_in
    if (not isinstance(block, ResidualBlock) or not isinstance(block.main, ConvLayer)
            or not isinstance(block.shortcut, ChannelDuplicatingPixelUnshuffleUpSampleLayer)):
        raise ValueError('Expected the native DC-AE convolution plus fixed duplication input shortcut')
    old = block.main.conv
    if (old.in_channels != 32 or old.kernel_size != (3, 3) or old.stride != (1, 1)
            or old.padding != (1, 1) or old.groups != 1 or block.shortcut.factor != 1
            or type(in_channels) is not int or in_channels <= 32):
        raise ValueError('Expected a 32-channel DC-AE input and a wider transformer feature input')
    decoder.project_in = build_decoder_project_in_block(in_channels, old.out_channels, shortcut=None).to(
        device=old.weight.device, dtype=old.weight.dtype)
    decoder.cfg.latent_channels = in_channels
    decoder.cfg.in_shortcut = None
    logger.info('DC-AE full-width input: %s→%s 3x3; fixed duplication shortcut removed',
                in_channels, old.out_channels)
    return decoder.project_in


class FullWidthDCAEDecoder(nn.Module):
    """Original shared scalar decoder taking DiT features, not scaled VAE latents."""
    def __init__(self, decoder, in_channels, logger):
        super().__init__()
        self.decoder = decoder.float().requires_grad_(True)
        adapt_ppp_decoder_input(self.decoder, in_channels, logger)
        self.in_channels = in_channels
        self.output_scale = 1

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError('Full-width DC-AE requires (3B, feature_channels, H, W)')
        # Transformer features have no VAE latent scaling to invert.
        if features.device.type != 'meta' and torch.is_autocast_enabled(features.device.type):
            features = features.to(torch.get_autocast_dtype(features.device.type))
        return self.decoder(features)


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
