"""Full-frame RGB preparation and calibrated patch rays."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CameraInput(nn.Module):
    """Normalize uint8 (B,V,3,H,W), then pad bottom/right to patch boundaries.

    Zero padding in normalized space corresponds to the RGB training mean.
    Intrinsics stay in original pixel coordinates: no resize or origin shift.
    Return the native extent alongside the tensor for partial-patch geometry.
    """

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1))

    def forward(self, images):
        if images.dtype != torch.uint8 or images.ndim != 5 or images.shape[2] != 3:
            raise ValueError('Camera input must be uint8 (B,V,3,H,W) RGB')
        height, width = images.shape[-2:]
        x = (images.float() / 255. - self.mean) / self.std
        return F.pad(x, (0, -width % self.patch_size, 0, -height % self.patch_size)), (height, width)


def patch_center_rays(intrinsics, reference_to_camera, image_hw, patch_size=16):
    """Return origins (B,V,3) and unit directions (B,V,P,3) in the shared frame.

    Patch centres use padded-grid coordinates. This diagnostic geometry does
    not estimate depth and is not itself the RayRoPE positional encoding.
    """
    height, width = image_hw
    rows, cols = torch.meshgrid(
        torch.arange(height // patch_size, device=intrinsics.device, dtype=intrinsics.dtype),
        torch.arange(width // patch_size, device=intrinsics.device, dtype=intrinsics.dtype),
        indexing='ij')
    pixels = torch.stack(((cols + .5) * patch_size, (rows + .5) * patch_size,
                          torch.ones_like(rows)), dim=-1).reshape(-1, 3)
    directions_camera = torch.linalg.solve(intrinsics, pixels.T).transpose(-1, -2)
    rotations = reference_to_camera[..., :3, :3].transpose(-1, -2)
    origins = -(rotations @ reference_to_camera[..., :3, 3:4]).squeeze(-1)
    directions = torch.einsum('bvij,bvpj->bvpi', rotations, directions_camera)
    return origins, F.normalize(directions, dim=-1)
