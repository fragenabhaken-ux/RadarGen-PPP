"""DINOv3 features and learned patch-to-pixel reconstruction for BEV radar."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    """Share pretrained weights and learn one low-rank update per conditioning.

    Inputs are grouped as [appearance batch, semantics batch, velocity batch].
    Each contiguous group uses its own adapter without mixing batch elements.
    """

    def __init__(
        self, linear: nn.Linear, rank: int, alpha: float, dropout: float, num_adapters: int = 1
    ):
        super().__init__()
        if num_adapters < 1:
            raise ValueError("num_adapters must be positive.")
        self.linear = linear.requires_grad_(False)
        self.scale = alpha / rank
        self.num_adapters = num_adapters
        self.dropout = nn.Dropout(dropout)
        lora_a_layers = [
            nn.Linear(linear.in_features, rank, bias=False) for _ in range(num_adapters)
        ]
        lora_b_layers = [
            nn.Linear(rank, linear.out_features, bias=False) for _ in range(num_adapters)
        ]
        for lora_a, lora_b in zip(lora_a_layers, lora_b_layers):
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_b.weight)
        self.lora_A = nn.ModuleList(lora_a_layers)
        self.lora_B = nn.ModuleList(lora_b_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] % self.num_adapters:
            raise ValueError("Input batch must divide evenly across LoRA adapters.")
        updates = [
            lora_b(lora_a(self.dropout(chunk)))
            for lora_a, lora_b, chunk in zip(self.lora_A, self.lora_B, x.chunk(self.num_adapters, dim=0))
        ]
        return self.linear(x) + self.scale * torch.cat(updates, dim=0)


def _rope_grid(height: int, width: int, head_dim: int, device: torch.device):
    """Build rotations for the actual patch grid, half per spatial axis."""
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    axis_dim = head_dim // 2
    frequencies = 2000.0 ** (-torch.arange(0, axis_dim, 2, device=device).float() / axis_dim)
    angles = torch.cat(
        [rows.flatten()[:, None] * frequencies, cols.flatten()[:, None] * frequencies], dim=-1
    )
    # Broadcast across batch elements and attention heads: (1, 1, H*W, head_dim/2).
    return angles.cos()[None, None], angles.sin()[None, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    first, second = pairs.unbind(-1)
    rotated = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def _rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * scale).to(x.dtype)


class ParallelTransformerBlock(nn.Module):
    """Template neck block: attention and an MLP share one normalized input.

    Tokens have shape (B, H_patch*W_patch, D). A fused projection produces
    Q/K/V and the MLP features; their outputs are joined before the residual.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, attn_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden = int(dim * mlp_ratio)
        self.attn_dropout = attn_dropout
        self.pre_norm = nn.LayerNorm(dim, bias=False)
        self.linear1 = nn.Linear(dim, 3 * dim + self.mlp_hidden, bias=False)
        self.linear2 = nn.Linear(dim + self.mlp_hidden, dim, bias=False)
        self.q_scale = nn.Parameter(torch.ones(self.head_dim))
        self.k_scale = nn.Parameter(torch.ones(self.head_dim))
        self.mlp_act = nn.SELU()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv, hidden = self.linear1(self.pre_norm(x)).split([3 * dim, self.mlp_hidden], dim=-1)
        q, k, v = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = _apply_rope(_rms_norm(q, self.q_scale), cos, sin)
        k = _apply_rope(_rms_norm(k, self.k_scale), cos, sin)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return x + self.linear2(torch.cat([attended, self.mlp_act(hidden)], dim=-1))


class RoPETransformerNeck(nn.Module):
    """Refine (B, D, H_patch, W_patch) features without changing their grid."""

    def __init__(
        self, dim: int, num_layers: int, num_heads: int, mlp_ratio: float,
        attn_dropout: float, grad_checkpointing: bool,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 4:
            raise ValueError("Neck dimension must give a head dimension divisible by four for 2D RoPE.")
        self.head_dim = dim // num_heads
        self.grad_checkpointing = grad_checkpointing
        self.layers = nn.ModuleList(
            ParallelTransformerBlock(dim, num_heads, mlp_ratio, attn_dropout) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        cos, sin = _rope_grid(height, width, self.head_dim, x.device)
        tokens = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, cos, sin, use_reentrant=False)
            else:
                tokens = layer(tokens, cos, sin)
        return self.out_norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class LocalConvResidualBlock(nn.Module):
    """Mix neighboring patch features with a two-convolution residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class PatchRearrangeHead(nn.Module):
    """Recover 16x spatial resolution in two learned 4x pixel-shuffle stages.

    (B, D, H/16, W/16) becomes (B, refine_channels, H/4, W/4), then
    (B, refine_channels, H, W). Convolutions mix adjacent features at each
    stage. The last convolution returns three raw channels for the PPP loss.
    """

    def __init__(self, dim: int, refine_channels: int, num_refine_convs: int, local_refine_blocks: int):
        super().__init__()
        self.local_refine = nn.Sequential(*(LocalConvResidualBlock(dim) for _ in range(local_refine_blocks)))
        self.patch_proj = nn.Conv2d(dim, refine_channels * 16, 1)
        layers = []
        for _ in range(num_refine_convs):
            layers.extend([
                nn.Conv2d(refine_channels, refine_channels, 3, padding=1),
                nn.BatchNorm2d(refine_channels),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(refine_channels, refine_channels * 16, 1))
        self.pre_refine = nn.Sequential(*layers)
        self.refine = nn.Sequential(
            nn.Conv2d(refine_channels, refine_channels, 3, padding=1),
            nn.BatchNorm2d(refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_shuffle(self.patch_proj(self.local_refine(x)), 4)
        x = F.pixel_shuffle(self.pre_refine(x), 4)
        return self.refine(x)


def _icnr_init(conv: nn.Conv2d, upscale_factor: int) -> None:
    """Initialize pixel-shuffle subkernels identically to reduce phase artifacts."""
    scale_area = upscale_factor**2
    if conv.out_channels % scale_area:
        raise ValueError("Convolution output channels must be divisible by upscale_factor squared.")
    subkernel = torch.empty_like(conv.weight[: conv.out_channels // scale_area])
    nn.init.kaiming_normal_(subkernel, mode="fan_in", nonlinearity="relu")
    with torch.no_grad():
        conv.weight.copy_(subkernel.repeat_interleave(scale_area, dim=0))
        if conv.bias is not None:
            conv.bias.zero_()


class ProgressivePixelShuffleHead(nn.Module):
    """Reconstruct pixels through four 2x shuffles with refinement at each scale.

    Each 3x3 projection produces four subpixel phases per output channel. ICNR
    initializes those phases identically; after pixel shuffle, a residual block
    mixes neighboring features at the new resolution.
    """

    def __init__(
        self,
        dim: int,
        stage_channels: tuple[int, ...] = (128, 64, 32, 16),
        local_refine_blocks: int = 1,
    ):
        super().__init__()
        if len(stage_channels) != 4 or any(channels <= 0 for channels in stage_channels):
            raise ValueError("Four positive stage widths are required to reconstruct 16x16 patches.")
        self.local_refine = nn.Sequential(
            *(LocalConvResidualBlock(dim) for _ in range(local_refine_blocks))
        )
        self.upsample_stages = nn.ModuleList()
        in_channels = dim
        for out_channels in stage_channels:
            projection = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
            _icnr_init(projection, upscale_factor=2)
            self.upsample_stages.append(
                nn.Sequential(
                    projection,
                    nn.GELU(),
                    nn.PixelShuffle(2),
                    LocalConvResidualBlock(out_channels),
                )
            )
            in_channels = out_channels
        readout_channels = in_channels
        self.readout = nn.Sequential(
            nn.Conv2d(readout_channels, readout_channels, 3, padding=1),
            nn.BatchNorm2d(readout_channels),
            nn.GELU(),
            nn.Conv2d(readout_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_refine(x)
        for stage in self.upsample_stages:
            x = stage(x)
        return self.readout(x)


class DinoPPP(nn.Module):
    """Predict BEV radar directly from three spatially aligned BEV conditions.

    Each input is (B, 3, H, W), already normalized by the dataset. The inputs are
    stacked into the batch dimension and processed independently by the native
    RGB DINOv3 backbone with a separate LoRA for each conditioning. For every
    selected block, (3B,D,h,w) features are regrouped into (B,3D,h,w) before
    projection and fusion. The shared neck keeps this grid; four 2x upsampling
    stages reconstruct the three radar fields at the original pixel resolution.

    Returns (B, 3, H, W) in fixed order: log-intensity, raw RCS, raw Doppler.
    The existing PPP loss applies the intensity and physical-mark links.
    There are no target latents, diffusion steps, or modality indicators.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        checkpoint_path: str = "",
        pretrained: bool = False,
        frozen_base: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        target_modules: tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        out_indices: tuple[int, ...] = (2, 5, 8, 11),
        per_layer_channels: int = 128,
        embed_dim: int = 384,
        neck_layers: int = 12,
        neck_heads: int = 6,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        decoder: Literal["patchrearrange", "progressive"] = "progressive",
        stage_channels: tuple[int, ...] = (128, 64, 32, 16),
        refine_channels: int = 128,
        num_refine_convs: int = 5,
        local_refine_blocks: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        import timm

        self.frozen_base = frozen_base
        self.out_indices = tuple(out_indices)
        # Keep the native RGB patch projection so pretrained weights load unchanged.
        model_kwargs = dict(pretrained=pretrained, in_chans=3, num_classes=0)
        if checkpoint_path:
            model_kwargs["checkpoint_path"] = str(checkpoint_path)
        self.backbone = timm.create_model(model_name, **model_kwargs)
        if frozen_base:
            self.backbone.requires_grad_(False)

        original = self.backbone.patch_embed.proj
        if original.kernel_size != (16, 16) or original.stride != (16, 16):
            raise ValueError("DinoPPP requires a backbone with nonoverlapping 16x16 patches.")
        if use_lora:
            if lora_r <= 0:
                raise ValueError("lora_r must be positive when LoRA is enabled.")
            last_block = max(index % len(self.backbone.blocks) for index in self.out_indices)
            for name, module in list(self.backbone.named_modules()):
                if not isinstance(module, nn.Linear) or not any(name.endswith(target) for target in target_modules):
                    continue
                if name.startswith("blocks.") and int(name.split(".")[1]) > last_block:
                    continue
                parent_name, _, child_name = name.rpartition(".")
                parent = self.backbone.get_submodule(parent_name)
                setattr(
                    parent,
                    child_name,
                    LoRALinear(module, lora_r, lora_alpha, lora_dropout, num_adapters=3),
                )
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        self.layer_projs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3 * original.out_channels, per_layer_channels, 1, bias=False),
                nn.GroupNorm(1, per_layer_channels),
                nn.GELU(),
            )
            for _ in self.out_indices
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(len(self.out_indices) * per_layer_channels, embed_dim, 1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )
        self.neck = RoPETransformerNeck(
            embed_dim, neck_layers, neck_heads, mlp_ratio, attn_dropout, grad_checkpointing
        )
        if decoder == "patchrearrange":
            self.head = PatchRearrangeHead(
                embed_dim, refine_channels, num_refine_convs, local_refine_blocks
            )
        elif decoder == "progressive":
            self.head = ProgressivePixelShuffleHead(
                embed_dim, stage_channels=stage_channels, local_refine_blocks=local_refine_blocks
            )
        else:
            raise ValueError(f"Unsupported decoder: {decoder}")
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_base:
            # Keep pretrained stochastic layers disabled while LoRA dropout follows
            # the predictor's mode. Evaluation disables dropout in every component.
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
        return self

    def forward(
        self, bev_color_map: torch.Tensor, bev_seg_map: torch.Tensor, bev_velocity_map: torch.Tensor
    ) -> torch.Tensor:
        height, width = bev_color_map.shape[-2:]
        batch_size = bev_color_map.shape[0]
        # Modality-major order keeps each modality in a contiguous batch chunk for LoRA routing.
        x = torch.cat([bev_color_map, bev_seg_map, bev_velocity_map], dim=0)
        # Bottom/right padding preserves BEV cell spacing. Crop the corresponding
        # predictions before returning, so padded cells never enter the PPP loss.
        x = F.pad(x, (0, -width % 16, 0, -height % 16))
        features = self.backbone.forward_intermediates(
            x, indices=list(self.out_indices), norm=True, output_fmt="NCHW", intermediates_only=True
        )
        features = [torch.cat(feature.split(batch_size, dim=0), dim=1) for feature in features]
        x = torch.cat([projection(feature) for projection, feature in zip(self.layer_projs, features)], dim=1)
        x = self.head(self.neck(self.fuse(x)))
        return x[..., :height, :width]
