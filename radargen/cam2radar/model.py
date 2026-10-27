"""Shared DINOv3, joint RayRoPE, and implicitly learned Cartesian PPP readout."""
import torch
from torch import nn
from torch.nn import functional as F

from radargen.models.dino_ppp import LoRALinear, LocalConvResidualBlock, _icnr_init
from .input import CameraInput
from .fusion import CameraChannelFusion
from .neck import MultiCameraRayNeck


class CartesianHead(nn.Module):
    """Expand features progressively, then compress to the fixed BEV grid.

    Three isotropic 2x shuffles and one vertical 2x shuffle change the native
    59x124 grid to 944x992. Decreasing widths limit high-resolution activation
    memory. Pooling precedes the raw output projection, never resampling PPP
    logits. The tensor plane acquires BEV meaning only through supervision.
    """
    def __init__(self, dim=384, output_size=512, widths=(128, 64, 32, 16)):
        super().__init__()
        self.output_size = output_size
        self.local = LocalConvResidualBlock(dim)
        self.stages = nn.ModuleList()
        previous = dim
        for width in widths[:3]:
            projection = nn.Conv2d(previous, width * 4, 3, padding=1)
            _icnr_init(projection, 2)
            self.stages.append(nn.Sequential(projection, nn.PixelShuffle(2),
                                             nn.GELU(), LocalConvResidualBlock(width)))
            previous = width
        self.vertical_projection = nn.Conv2d(previous, widths[3] * 2, 3, padding=1)
        self.refine = LocalConvResidualBlock(widths[3])
        self.readout = nn.Sequential(LocalConvResidualBlock(widths[3]), nn.Conv2d(widths[3], 3, 1))

    def forward(self, x):
        x = self.local(x)
        for stage in self.stages:
            x = stage(x)
        x = self.vertical_projection(x)
        b, channels, h, w = x.shape
        x = x.reshape(b, channels // 2, 2, h, w).permute(0, 1, 3, 2, 4).reshape(b, channels // 2, 2*h, w)
        x = self.refine(x)
        if min(x.shape[-2:]) < self.output_size:
            raise ValueError('Expanded feature plane must cover the Cartesian output grid')
        return self.readout(F.adaptive_avg_pool2d(x, self.output_size))


class Camera2Radar(nn.Module):
    """Return (B,3,512,512): log-intensity, raw RCS, raw signed Doppler.

    Images are full uint8 RGB (B,4,3,H,W). Intrinsics use original pixels;
    reference_to_camera maps metric radar-flat-up coordinates into each camera.
    One shared LoRA adapter handles all cameras. DINO's own PE is unchanged.
    """
    def __init__(self, model_name='vit_small_patch16_dinov3', checkpoint_path='',
                 frozen_base=True, use_lora=True, lora_r=16, lora_alpha=32.,
                 out_indices=(2, 5, 8, 11), per_layer_channels=128, embed_dim=384,
                 neck_layers=12, neck_heads=4, max_depth=75., origin_period=32.,
                 grad_checkpointing=True, output_size=512):
        super().__init__()
        import timm
        self.input = CameraInput()
        self.frozen_base = frozen_base
        self.out_indices = tuple(out_indices)
        kwargs = dict(pretrained=False, in_chans=3, num_classes=0)
        if checkpoint_path:
            kwargs['checkpoint_path'] = str(checkpoint_path)
        self.backbone = timm.create_model(model_name, **kwargs)
        if frozen_base:
            self.backbone.requires_grad_(False)
        if use_lora:
            for name, module in list(self.backbone.named_modules()):
                if isinstance(module, nn.Linear) and any(name.endswith(s) for s in
                        ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')):
                    parent_name, _, child = name.rpartition('.')
                    setattr(self.backbone.get_submodule(parent_name), child,
                            LoRALinear(module, lora_r, lora_alpha, 0., num_adapters=1))
        self.backbone.set_grad_checkpointing(grad_checkpointing)
        width = self.backbone.patch_embed.proj.out_channels
        self.layer_projs = nn.ModuleList(nn.Sequential(nn.Conv2d(width, per_layer_channels, 1, bias=False),
            nn.GroupNorm(1, per_layer_channels), nn.GELU()) for _ in out_indices)
        self.fuse_layers = nn.Conv2d(len(out_indices) * per_layer_channels, embed_dim, 1)
        self.neck = MultiCameraRayNeck(embed_dim, neck_heads, neck_layers, max_depth,
                                       origin_period, grad_checkpointing)
        self.fuse_cameras = CameraChannelFusion(embed_dim)
        self.head = CartesianHead(embed_dim, output_size)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_base:
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
        return self

    def forward(self, camera_images, camera_intrinsics, reference_to_camera):
        images, native_hw = self.input(camera_images)
        b, views, channels, h, w = images.shape
        if views != 4:
            raise ValueError('Camera2Radar expects the four manifest-ordered cameras')
        features = self.backbone.forward_intermediates(images.flatten(0, 1),
            indices=list(self.out_indices), norm=True, output_fmt='NCHW', intermediates_only=True)
        fused = self.fuse_layers(torch.cat([p(f) for p, f in zip(self.layer_projs, features)], dim=1))
        fused = fused.reshape(b, views, *fused.shape[1:])
        fused = self.neck(fused, camera_intrinsics, reference_to_camera, (h, w))
        return self.head(self.fuse_cameras(fused))
