"""Joint camera attention with per-layer learned ray depth and uncertainty."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from radargen.models.dino_ppp import _rms_norm
from .vendor.rayrope import RayRoPE_DotProductAttention


class RayBlock(nn.Module):
    def __init__(self, dim, heads, max_depth, origin_period):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        if dim % heads or self.head_dim % 24:
            raise ValueError('Three-ray RayRoPE requires head dimensions divisible by 24')
        self.max_depth = max_depth
        self.origin_period = origin_period
        self.norm = nn.LayerNorm(dim, bias=False)
        self.qkv_mlp = nn.Linear(dim, 7 * dim, bias=False)
        self.output = nn.Linear(5 * dim, dim, bias=False)
        self.depth = nn.Linear(dim, 2)
        nn.init.zeros_(self.depth.weight)
        # Broad initial ray segment, centered at 10 m rather than one coordinate unit.
        with torch.no_grad():
            self.depth.bias.copy_(torch.tensor([math.log(10.), 1.]))
        self.q_scale = nn.Parameter(torch.ones(self.head_dim))
        self.k_scale = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, tokens, intrinsics, transforms, image_hw):
        batch, length, dim = tokens.shape
        h = self.norm(tokens)
        qkv, mlp = self.qkv_mlp(h).split([3 * dim, 4 * dim], dim=-1)
        q, k, v = qkv.reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Build per-call geometry rather than caching batch tensors on modules:
        # checkpoint recomputation must not observe a later microbatch's calibration.
        height, width = image_hw
        attention = RayRoPE_DotProductAttention(
            self.head_dim, width // 16, height // 16, width, height,
            max_depth=self.max_depth, origin_period=self.origin_period)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            attention._precompute_and_cache_apply_fns(transforms.float(), intrinsics.float())
            raw_depth = self.depth(h.float())
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            attended = attention(_rms_norm(q, self.q_scale), _rms_norm(k, self.k_scale), v,
                                 predicted_d=raw_depth)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return tokens + self.output(torch.cat([attended.to(h.dtype), torch.nn.functional.selu(mlp)], -1))


class MultiCameraRayNeck(nn.Module):
    """Camera-major tokens attend to all views, then retain their patch anchors."""
    def __init__(self, dim=384, heads=4, layers=12, max_depth=75., origin_period=32.,
                 grad_checkpointing=True):
        super().__init__()
        self.layers = nn.ModuleList(RayBlock(dim, heads, max_depth, origin_period) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.grad_checkpointing = grad_checkpointing

    def forward(self, features, intrinsics, transforms, image_hw):
        b, views, dim, hp, wp = features.shape
        tokens = features.permute(0, 1, 3, 4, 2).reshape(b, views * hp * wp, dim)
        for layer in self.layers:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, intrinsics, transforms, image_hw, use_reentrant=False)
            else:
                tokens = layer(tokens, intrinsics, transforms, image_hw)
        return self.norm(tokens).reshape(b, views, hp, wp, dim).permute(0, 1, 4, 2, 3)
