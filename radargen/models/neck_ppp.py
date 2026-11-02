"""Trainable BEV CNN stem with the existing DINO PPP neck and decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .dino_ppp import RoPETransformerNeck, ProgressivePixelShuffleHead


class NeckPPP(nn.Module):
    """Fuse aligned RGB conditions into nine channels; reconstruct at stride 16.

    Inputs use the existing dataset normalization. Output channels are
    log-intensity, raw RCS, raw Doppler, with links applied by the PPP loss.
    """

    def __init__(self, embed_dim=384, stem_channels=(32, 64, 128),
                 neck_layers=12, neck_heads=6, mlp_ratio=4.0,
                 attn_dropout=0.0, stage_channels=(128, 64, 32, 16),
                 local_refine_blocks=2, grad_checkpointing=False, use_skip_connections=True):
        super().__init__()
        if len(stem_channels) != 3 or any(c <= 0 for c in stem_channels):
            raise ValueError("stem_channels must contain three positive widths")
        self.use_skip_connections = use_skip_connections
        layers = []
        in_channels = 9
        for channels in (*stem_channels, embed_dim):
            layers.extend([nn.Conv2d(in_channels, channels, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(1, channels), nn.GELU()])
            in_channels = channels
        self.stem = nn.Sequential(*layers)
        self.neck = RoPETransformerNeck(embed_dim, neck_layers, neck_heads,
                                        mlp_ratio, attn_dropout, grad_checkpointing)
        self.head = ProgressivePixelShuffleHead(embed_dim, tuple(stage_channels), local_refine_blocks)
        if use_skip_connections:
            # Match decoder stages at H/8, H/4, H/2 and full input resolution.
            self.skip_projs = nn.ModuleList(
                nn.Conv2d(source, target, 1, bias=False)
                for source, target in zip((*reversed(stem_channels), 9), stage_channels)
            )
            self.skip_fusions = nn.ModuleList(
                nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
                              nn.GroupNorm(1, channels), nn.GELU())
                for channels in stage_channels
            )

    def forward(self, bev_color_map, bev_seg_map, bev_velocity_map):
        conditions = (bev_color_map, bev_seg_map, bev_velocity_map)
        if (bev_color_map.ndim != 4 or bev_color_map.shape[1] != 3
                or min(bev_color_map.shape) <= 0
                or any(x.shape != bev_color_map.shape for x in conditions)):
            raise ValueError("Expected three matching nonempty (B,3,H,W) conditions")
        height, width = bev_color_map.shape[-2:]
        x = torch.cat(conditions, dim=1)
        x = F.pad(x, (0, -width % 16, 0, -height % 16))
        if not self.use_skip_connections:
            return self.head(self.neck(self.stem(x)))[..., :height, :width]
        skips = [x]
        # Retain activations after each convolution/normalization/activation group.
        for index, layer in enumerate(self.stem):
            x = layer(x)
            if index % 3 == 2 and index < len(self.stem) - 1:
                skips.append(x)
        x = self.head.local_refine(self.neck(x))
        for stage, projection, fusion, skip in zip(
                self.head.upsample_stages, self.skip_projs, self.skip_fusions, reversed(skips)):
            x = stage(x)
            x = fusion(torch.cat((x, projection(skip)), dim=1))
        return self.head.readout(x)[..., :height, :width]
