"""One-shot camera-to-radar transformer with gathered angular cross attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from radargen.cam2radar.input import CameraInput
from radargen.models.dino_ppp import LoRALinear, LocalConvResidualBlock, _icnr_init


def bev_centers(size, limit, device):
    """Match continuous PPP targets: x increases right, y increases upward."""
    axis = (torch.arange(size, device=device, dtype=torch.float32) + .5) / size
    y, x = torch.meshgrid(limit - 2 * limit * axis, 2 * limit * axis - limit, indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


def camera_geometry(intrinsics, transforms, native_hw, grid_hw, patch_size):
    """Rays through valid patch footprints; partial edge patches exclude padding."""
    h, w = native_hw
    gh, gw = grid_hw
    device = intrinsics.device
    rows = torch.arange(gh, device=device).float() * patch_size
    cols = torch.arange(gw, device=device).float() * patch_size
    rows = (rows + (rows + patch_size).clamp(max=h)) / 2
    cols = (cols + (cols + patch_size).clamp(max=w)) / 2
    yy, xx = torch.meshgrid(rows, cols, indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    rotation = transforms.float()[..., :3, :3].transpose(-1, -2)
    origins = -(rotation @ transforms.float()[..., :3, 3:4]).squeeze(-1)
    rays = torch.linalg.solve(intrinsics.float(), pixels.T).transpose(-1, -2)
    rays = F.normalize(torch.einsum('bvij,bvpj->bvpi', rotation, rays), dim=-1)
    azimuth = torch.atan2(rays[..., 1], rays[..., 0])
    elevation = torch.atan2(rays[..., 2], rays[..., :2].norm(dim=-1))
    return origins, azimuth, elevation


class ResidualMLP(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ratio * dim),
                                 nn.GELU(), nn.Linear(ratio * dim, dim))

    def forward(self, x):
        return x + self.net(x)


class WindowAttention(nn.Module):
    """Alternating regular/shifted windows with no wraparound communication."""
    def __init__(self, dim, heads, window, shifted=False):
        super().__init__()
        self.heads, self.window = heads, window
        self.shift = window // 2 if shifted else 0
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, h, w, d = x.shape
        win = self.window
        ph, pw = -h % win, -w % win
        z = F.pad(self.norm(x), (0, 0, 0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        yy, xx = torch.meshgrid(torch.arange(hp, device=x.device),
                                 torch.arange(wp, device=x.device), indexing='ij')
        coords = torch.stack((yy, xx), -1)
        if self.shift:
            z = z.roll((-self.shift, -self.shift), (1, 2))
            coords = coords.roll((-self.shift, -self.shift), (0, 1))
        def partition(t):
            return t.reshape(-1, hp // win, win, wp // win, win, t.shape[-1]).permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win, t.shape[-1])
        z = partition(z)
        pos = partition(coords.unsqueeze(0))
        valid = (pos[..., 0] < h) & (pos[..., 1] < w)
        local = (pos[:, :, None] - pos[:, None, :]).abs().amax(-1) < win
        mask = (local & valid[:, None, :]).repeat(b, 1, 1)
        # Padded query rows get a harmless self key; their outputs are cropped.
        mask = mask | torch.eye(win * win, device=x.device, dtype=torch.bool)[None]
        q, k, v = self.qkv(z).reshape(-1, win * win, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        z = self.proj(z.transpose(1, 2).reshape(-1, win * win, d))
        z = z.reshape(b, hp // win, wp // win, win, win, d).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, d)
        if self.shift:
            z = z.roll((self.shift, self.shift), (1, 2))
        return x + z[:, :h, :w]


class AngularAttention(nn.Module):
    """Gather a bounded angular neighborhood without a dense query/key matrix.

    Sorted, periodic rays supply at most max_keys candidates per camera. The
    angular radius then masks candidates. Increase max_keys for wider strips;
    the cap is an explicit approximation, not a guarantee to include every ray.
    """
    def __init__(self, dim, camera_dim, heads, radius, max_keys, chunk_size, limit):
        super().__init__()
        self.heads, self.radius = heads, radius
        self.max_keys, self.chunk_size, self.limit = max_keys, chunk_size, limit
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(camera_dim, 2 * dim)
        self.range_embed = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.angle_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, heads))
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, camera, geometry):
        b, h, w, dim = x.shape
        origins, angles, _ = geometry
        points = bev_centers(h, self.limit, x.device)
        normalized = self.norm(x).reshape(b, h * w, dim)
        views, patches = camera.shape[1:3]
        k, v = self.kv(camera).chunk(2, -1)
        count = min(self.max_keys, patches)
        # Three copies handle the -pi/pi seam without special cases.
        sorted_angles, order = angles.sort(-1)
        sorted_angles = torch.cat((sorted_angles - 2 * math.pi, sorted_angles, sorted_angles + 2 * math.pi), -1)
        order = order.repeat(1, 1, 3)
        updates = []
        for batch in range(b):
            pieces = []
            for start in range(0, h * w, self.chunk_size):
                stop = min(start + self.chunk_size, h * w)
                delta = points[start:stop, None] - origins[batch, None, :, :2]
                distances = delta.norm(dim=-1)
                theta = torch.atan2(delta[..., 1], delta[..., 0]).T.contiguous()
                insertion = torch.searchsorted(sorted_angles[batch].contiguous(), theta)
                offsets = torch.arange(count, device=x.device) - count // 2
                slots = (insertion[..., None] + offsets).clamp(0, 3 * patches - 1)
                ids = order[batch].gather(1, slots.reshape(views, -1)).reshape(views, stop-start, count)
                view_ids = torch.arange(views, device=x.device)[:, None, None]
                selected_angles = angles[batch][view_ids, ids]
                difference = torch.atan2((selected_angles-theta[..., None]).sin(),
                                          (selected_angles-theta[..., None]).cos())
                valid = (difference.abs() <= self.radius) & (distances.T[..., None] > 1e-5)
                q = self.q(normalized[batch, start:stop])[:, None] + self.range_embed(
                    (distances / self.limit).unsqueeze(-1).to(x.dtype))
                q = q.reshape(stop-start, views, self.heads, dim // self.heads)
                keys = k[batch][view_ids, ids].permute(1, 0, 2, 3).reshape(stop-start, views, count, self.heads, dim // self.heads)
                values = v[batch][view_ids, ids].permute(1, 0, 2, 3).reshape_as(keys)
                logits = torch.einsum('nvhd,nvkhd->nhvk', q, keys) / math.sqrt(dim // self.heads)
                bias = self.angle_bias((difference / self.radius).unsqueeze(-1).to(x.dtype)).permute(1, 3, 0, 2)
                logits = (logits + bias).float().flatten(2)
                mask = valid.permute(1, 0, 2).flatten(1)[:, None]
                # Finite empty rows and exactly zero update for uncovered queries.
                weights = logits.masked_fill(~mask, -1e9).softmax(-1) * mask
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
                weights = weights.reshape(stop-start, self.heads, views, count).to(values.dtype)
                result = torch.einsum('nhvk,nvkhd->nhd', weights, values).reshape(stop-start, dim)
                pieces.append(self.proj(result))
            updates.append(torch.cat(pieces))
        return x + torch.stack(updates).reshape_as(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, camera_dim, heads, window, shifted, **angular):
        super().__init__()
        self.self_attention = WindowAttention(dim, heads, window, shifted)
        self.cross_attention = AngularAttention(dim, camera_dim, heads, **angular)
        self.mlp = ResidualMLP(dim)

    def forward(self, x, camera, geometry):
        return self.mlp(self.cross_attention(self.self_attention(x), camera, geometry))


def upsample(in_dim, out_dim):
    conv = nn.Conv2d(in_dim, out_dim * 4, 3, padding=1)
    _icnr_init(conv, 2)
    return nn.Sequential(conv, nn.PixelShuffle(2), nn.GELU())


class Image2RadarV2(nn.Module):
    """Shared DINO+LoRA -> 2+2 BEV decoder -> three continuous PPP fields.

    Input interface matches Camera2Radar. Intrinsics are undistorted original
    pixel coordinates; reference_to_camera is a rigid reference-frame transform.
    Default output channels: log intensity, raw RCS, raw signed Doppler.
    """
    def __init__(self, model_name='vit_small_patch16_dinov3', checkpoint_path='',
                 frozen_base=True, use_lora=True, lora_r=16, lora_alpha=32.,
                 widths=(256, 128, 64, 32), depths=(2, 2), heads=(8, 8, 4),
                 query_size=64, output_size=512, coordinate_range=50., window_size=8,
                 angular_radius_degrees=3., max_keys=256, query_chunk_size=128,
                 grad_checkpointing=True, backbone=None):
        super().__init__()
        if len(widths) != 4 or len(depths) != 2 or len(heads) != 3:
            raise ValueError('Expected four widths, two depths, and three head counts')
        if output_size != 8 * query_size or query_size < 1:
            raise ValueError('output_size must be eight times query_size')
        if any(d <= 0 for d in widths) or any(n < 1 for n in depths):
            raise ValueError('Widths and depths must be positive')
        if any(n < 1 or d % n for d, n in zip(widths, heads)):
            raise ValueError('Each attention width must be divisible by its heads')
        if min(window_size, max_keys, query_chunk_size, coordinate_range) <= 0 or not 0 < angular_radius_degrees < 180:
            raise ValueError('Invalid attention neighborhood or spatial extent')
        if backbone is None:
            import timm
            kwargs = dict(pretrained=False, in_chans=3, num_classes=0)
            if checkpoint_path:
                kwargs['checkpoint_path'] = str(checkpoint_path)
            backbone = timm.create_model(model_name, **kwargs)
        self.backbone = backbone
        self.frozen_base = frozen_base
        if frozen_base:
            backbone.requires_grad_(False)
        if use_lora:
            if lora_r <= 0:
                raise ValueError('LoRA rank must be positive')
            for name, module in list(backbone.named_modules()):
                if isinstance(module, nn.Linear) and name.endswith(('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')):
                    parent, _, child = name.rpartition('.')
                    setattr(backbone.get_submodule(parent), child, LoRALinear(module, lora_r, lora_alpha, 0., num_adapters=1))
        backbone.set_grad_checkpointing(grad_checkpointing)
        patch = backbone.patch_embed.patch_size
        if isinstance(patch, int):
            patch = (patch, patch)
        if patch[0] != patch[1]:
            raise ValueError('Square camera patches required')
        self.patch_size = patch[0]
        self.input = CameraInput(self.patch_size)
        self.adapter = nn.Sequential(nn.Linear(backbone.patch_embed.proj.out_channels, widths[0]), ResidualMLP(widths[0]))
        self.elevation = nn.Sequential(nn.Linear(1, widths[0]), nn.GELU(), nn.Linear(widths[0], widths[0]))
        self.query = nn.Parameter(torch.empty(1, query_size, query_size, widths[0]))
        nn.init.normal_(self.query, std=.02)
        self.position = nn.Linear(2, widths[0])
        self.query_size, self.limit = query_size, coordinate_range
        angular = dict(radius=math.radians(angular_radius_degrees), max_keys=max_keys,
                       chunk_size=query_chunk_size, limit=coordinate_range)
        self.stages = nn.ModuleList(nn.ModuleList(DecoderBlock(widths[i], widths[0], heads[i], window_size, j % 2 == 1, **angular)
            for j in range(depths[i])) for i in range(2))
        self.ups = nn.ModuleList(upsample(a, b) for a, b in zip(widths, widths[1:]))
        self.fine_cross = AngularAttention(widths[2], widths[0], heads[2], **angular)
        self.fine_refine = LocalConvResidualBlock(widths[2])
        self.readout = nn.Sequential(LocalConvResidualBlock(widths[3]), nn.Conv2d(widths[3], 3, 1))
        self.grad_checkpointing = grad_checkpointing
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_base:
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
        return self

    def _run(self, module, x, camera, geometry):
        if self.training and self.grad_checkpointing:
            return checkpoint(lambda a, b: module(a, b, geometry), x, camera, use_reentrant=False)
        return module(x, camera, geometry)

    def forward(self, camera_images, camera_intrinsics, reference_to_camera):
        images, native = self.input(camera_images)
        b, views = images.shape[:2]
        if views != 4 or camera_intrinsics.shape != (b, views, 3, 3) or reference_to_camera.shape != (b, views, 4, 4):
            raise ValueError('Expected four cameras and matching (B,4,3,3)/(B,4,4,4) calibration')
        features = self.backbone.forward_intermediates(images.flatten(0, 1), indices=[-1],
            norm=True, output_fmt='NCHW', intermediates_only=True)[0]
        gh, gw = features.shape[-2:]
        # Disable autocast for camera solves and trigonometry.
        with torch.autocast(device_type=images.device.type, enabled=False):
            geometry = camera_geometry(camera_intrinsics, reference_to_camera, native, (gh, gw), self.patch_size)
        camera = self.adapter(features.flatten(2).transpose(1, 2)).reshape(b, views, gh * gw, -1)
        camera = camera + self.elevation(geometry[2].unsqueeze(-1).to(camera.dtype))
        positions = bev_centers(self.query_size, self.limit, images.device) / self.limit
        x = self.query + self.position(positions.to(camera.dtype)).reshape(1, self.query_size, self.query_size, -1)
        x = x.expand(b, -1, -1, -1)
        for index, blocks in enumerate(self.stages):
            for block in blocks:
                x = self._run(block, x, camera, geometry)
            x = self.ups[index](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self._run(self.fine_cross, x, camera, geometry)
        x = self.fine_refine(x.permute(0, 3, 1, 2))
        return self.readout(self.ups[2](x))
