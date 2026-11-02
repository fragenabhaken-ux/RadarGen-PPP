"""Decode full-width DiT features directly to scalar BEV fields."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualRefinement(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels), nn.SiLU(),
            nn.Conv2d(channels, hidden, 1),
            nn.GroupNorm(1, hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class WidePPPDecoder(nn.Module):
    """Shared scalar decoder; five doubling stages replace DC-AE's factor 32.

    Fit useful low-resolution refinement weights to the scalar DC-AE budget.
    No VAE latent scaling or output activation applies to DiT features.
    """
    def __init__(self, in_channels=1152, widths=(768, 384, 192, 96, 48),
                 parameter_budget=159220865, refinement_blocks=6,
                 gradient_checkpointing=False):
        super().__init__()
        if (len(widths) != 5 or any(type(c) is not int or c <= 0 for c in widths)
                or type(in_channels) is not int or in_channels <= 0
                or type(refinement_blocks) is not int or refinement_blocks <= 0
                or type(parameter_budget) is not int or parameter_budget <= 0):
            raise ValueError('Expected positive channel widths, budget, blocks and five upsampling stages')
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.parameter_budget = parameter_budget
        stages = []
        previous = in_channels
        for width in widths:
            stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(previous, width, 3, padding=1),
                ResidualRefinement(width, width),
            ))
            previous = width
        self.stages = nn.ModuleList(stages)
        self.output = nn.Conv2d(previous, 1, 3, padding=1)
        fixed = sum(p.numel() for p in self.parameters())
        # Each refinement contains 10*C*M + 3*M + 3*C parameters.
        available = parameter_budget - fixed - refinement_blocks * 3 * in_channels
        hidden = (available // (refinement_blocks * (10 * in_channels + 3))) // 8 * 8
        if hidden < 8:
            raise ValueError('Decoder parameter budget is too small for the requested widths')
        self.refinement = nn.ModuleList([
            ResidualRefinement(in_channels, hidden) for _ in range(refinement_blocks)
        ])
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > parameter_budget:
            raise ValueError('Decoder exceeds parameter budget')

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError(f'Expected (N,{self.in_channels},h,w) DiT features')
        x = features
        for stage in (*self.refinement, *self.stages):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return self.output(x)


def _efficient_block(channels, hidden=None):
    """Reuse DC-AE's multi-scale LiteMLA + gated local convolution block."""
    from diffusion.model.dc_ae.efficientvit.models.nn.ops import EfficientViTBlock
    from diffusion.model.dc_ae.efficientvit.models.nn.norm import RMSNorm2d

    block = EfficientViTBlock(
        channels, dim=32, expand_ratio=4 if hidden is None else hidden / channels,
        scales=(5,), norm='rms2d', act_func='silu', local_module='GLUMBConv',
    )
    # The standalone EfficientViT RMSNorm constructor uses torch.empty;
    # DC-AE normally initializes these affine parameters at model level.
    for module in block.modules():
        if isinstance(module, RMSNorm2d):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return block


def _efficient_parameter_count(channels, hidden):
    with torch.device('meta'):
        return sum(p.numel() for p in _efficient_block(channels, hidden).parameters())


class NearestEfficientPPPDecoder(nn.Module):
    """Nearest-neighbor decoding with DC-AE EfficientViT at 16, 32 and 64 px."""
    def __init__(self, in_channels=1152, widths=(768, 384, 192, 96, 48),
                 parameter_budget=159220865, refinement_blocks=6,
                 gradient_checkpointing=False):
        super().__init__()
        if (len(widths) != 5 or any(type(c) is not int or c <= 0 for c in widths)
                or type(in_channels) is not int or min(in_channels, *widths[:2]) < 32
                or type(refinement_blocks) is not int or refinement_blocks < 1
                or type(parameter_budget) is not int or parameter_budget <= 0):
            raise ValueError('Expected five positive widths, attention widths >=32, positive budget and blocks')
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.parameter_budget = parameter_budget
        stages = []
        previous = in_channels
        for index, width in enumerate(widths):
            stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(previous, width, 3, padding=1),
                _efficient_block(width) if index < 2 else ResidualRefinement(width, width),
            ))
            previous = width
        self.stages = nn.ModuleList(stages)
        self.output = nn.Conv2d(previous, 1, 3, padding=1)
        # Full-width pretrained features have not passed the old normalized
        # latent head. Start the exponential PPP link near unit intensity.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        fixed = sum(p.numel() for p in self.parameters())
        # GLUMBConv's parameter count is affine in its hidden width. Fit only
        # the six low-resolution blocks; the two upsampling attention blocks
        # retain the original expansion ratio four.
        count8 = _efficient_parameter_count(in_channels, 8)
        step = _efficient_parameter_count(in_channels, 16) - count8
        available = (parameter_budget - fixed) // refinement_blocks
        hidden = 8 + ((available - count8) // step) * 8
        if hidden < 8:
            raise ValueError('Decoder parameter budget is too small for the requested widths')
        self.refinement_hidden = hidden
        self.refinement = nn.ModuleList([
            _efficient_block(in_channels, hidden) for _ in range(refinement_blocks)
        ])
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > parameter_budget:
            raise ValueError('Decoder exceeds parameter budget')

    forward = WidePPPDecoder.forward


class SpatialResNetBlock(nn.Module):
    """Pre-activation residual block with two spatial 3x3 convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class PixelShufflePPPDecoder(nn.Module):
    """Factor-32 decoder: PixelShuffle+bilinear, PixelShuffle+bilinear, PixelShuffle.

    A 1x1 projection expands full DiT features; no narrow latent raster is
    introduced. Full-size defaults use 33/3/3 spatial residual blocks to match
    the old decoder budget, with most weights at the first refinement stage.
    """
    def __init__(self, in_channels=1152, stage_channels=512, stage_depths=None,
                 parameter_budget=159220865, gradient_checkpointing=False):
        super().__init__()
        if (type(in_channels) is not int or in_channels <= 0
                or type(stage_channels) is not int or stage_channels < 16 or stage_channels % 16
                or type(parameter_budget) is not int or parameter_budget <= 0):
            raise ValueError('Expected positive input/budget and stage_channels divisible by 16')
        if stage_depths is not None and (len(stage_depths) != 3 or
                any(type(d) is not int or d < 1 for d in stage_depths)):
            raise ValueError('Expected three positive residual stage depths')
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.parameter_budget = parameter_budget
        widths = (stage_channels, stage_channels // 4, stage_channels // 16)
        projection = nn.Conv2d(in_channels, stage_channels * 4, 1)
        self.refinement = nn.ModuleList([projection])
        self.output = nn.Conv2d(widths[-1], 1, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        if stage_depths is None:
            # A two-convolution residual block has 18*C*C + 6*C parameters.
            block_count = lambda c: 18 * c * c + 6 * c
            fixed = sum(p.numel() for p in self.parameters())
            fixed += 3 * block_count(widths[1]) + 3 * block_count(widths[2])
            first_depth = (parameter_budget - fixed) // block_count(widths[0])
            if first_depth < 1:
                raise ValueError('Decoder parameter budget is too small for the requested widths')
            stage_depths = (first_depth, 3, 3)
        self.stage_depths = tuple(stage_depths)
        stages = []
        for index, (width, depth) in enumerate(zip(widths, stage_depths)):
            layers = [nn.PixelShuffle(2)]
            if index < 2:
                # Pixel-center alignment gives exactly 64/256 for a 16px input,
                # retaining the existing 512px target geometry after stage three.
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            # Checkpoint residual blocks individually, rather than retaining an
            # entire deep stage's activations during recomputation.
            stages.append(nn.Sequential(*layers))
            stages.extend(SpatialResNetBlock(width) for _ in range(depth))
        self.stages = nn.ModuleList(stages)
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > parameter_budget:
            raise ValueError('Decoder exceeds parameter budget')

    forward = WidePPPDecoder.forward


class JointDinoAttentionPPPDecoder(nn.Module):
    """Concatenate three full-width DiT streams and reuse DINO's shifted head.

    DiT uses sample-major (B, modality, C, h, w) flattened to (3B,C,h,w).
    The head receives (B,3C,h,w) and predicts (B,3,32h,32w) or (B,3,64h,64w).
    """
    joint_streams = True

    def __init__(self, in_channels=1152, parameter_budget=159220865,
                 gradient_checkpointing=False, channels=(128, 64, 32),
                 blocks=2, heads=4, window_size=8, output_scale=1):
        super().__init__()
        from radargen.models.dino_ppp import AttentionPixelShuffleHead
        if type(output_scale) is not int or output_scale not in (1, 2):
            raise ValueError("Joint DiT output_scale must be 1 or 2")
        self.output_scale = output_scale
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.parameter_budget = parameter_budget
        self.head = AttentionPixelShuffleHead(
            dim=3 * in_channels, channels=channels, blocks=blocks, heads=heads,
            window_size=window_size, output_scale=2,
            grad_checkpointing=gradient_checkpointing,
            projection_kernel_size=3, shifted_windows=True,
            final_upsample_factor=2 * output_scale,
        )
        # Keep the PPP exponential link well scaled at fresh initialization.
        nn.init.normal_(self.head.readout.weight, std=1e-3)
        nn.init.zeros_(self.head.readout.bias)
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > parameter_budget:
            raise ValueError('Joint DINO decoder exceeds parameter budget')

    def forward(self, features):
        if (features.ndim != 4 or features.shape[1] != self.in_channels
                or features.shape[0] == 0 or features.shape[0] % 3):
            raise ValueError(f'Expected sample-major (3B,{self.in_channels},h,w) DiT features')
        expanded, channels, height, width = features.shape
        joint = features.reshape(expanded // 3, 3 * channels, height, width)
        self.head.grad_checkpointing = self.gradient_checkpointing
        return self.head(joint)
