"""One-shot BEV transformer with learned factor-eight features and signed fields."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class JointBlock(nn.Module):
    def __init__(self, width, head_dim=32, mlp_ratio=2.5):
        super().__init__()
        self.heads = width // head_dim
        self.head_dim = head_dim
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        mid = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(width, mid, 1), nn.SiLU(),
                                 nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
                                 nn.SiLU(), nn.Conv2d(mid, width, 1))
        self.gates = nn.Parameter(torch.full((2, width), 0.1))

    def forward(self, x):
        b, v, h, w, c = x.shape
        tokens = self.norm1(x).reshape(b, v*h*w, c)
        q, k, value = self.qkv(tokens).reshape(b, -1, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # ReLU linear attention, normalized in FP32; jointly attends all streams.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = self.q_norm(q).float().relu()
            k = self.k_norm(k).float().relu()
            value = value.float()
            kv = k.transpose(-1, -2) @ value
            normalizer = (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
            attended = ((q @ kv) / normalizer).transpose(1, 2).reshape(b, v, h, w, c)
        x = x + self.gates[0] * self.proj(attended.to(tokens.dtype))
        local = self.norm2(x).reshape(b*v, h, w, c).permute(0, 3, 1, 2)
        local = self.mlp(local).permute(0, 2, 3, 1).reshape(b, v, h, w, c)
        return x + self.gates[1] * local


class DiTV2(nn.Module):
    """Three RGB BEV maps -> (B,3,H,W): log intensity, RCS dBsm, Doppler m/s.

    No pretrained assets, text, timestep or noise. Encoder and shared decoder
    train end-to-end. No final activation: physical marks use identity links.
    """
    def __init__(self, hidden_size=1152, depth=28, latent_channels=64,
                 head_dim=32, mlp_ratio=2.5, grad_checkpointing=True):
        super().__init__()
        if hidden_size % head_dim or hidden_size % 4 or depth < 1 or latent_channels < 1:
            raise ValueError('Invalid DiTV2 dimensions')
        def encoder():
            return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(64, latent_channels, 3, 2, 1), nn.SiLU())
        self.encoders = nn.ModuleList([encoder() for _ in range(3)])
        self.input_projection = nn.Conv2d(3*latent_channels, hidden_size, 1)
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden_size) * 0.02)
        self.blocks = nn.ModuleList([JointBlock(hidden_size, head_dim, mlp_ratio) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_size)
        self.latent_projection = nn.Conv2d(hidden_size, latent_channels, 1)
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, 128, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1))
        self.grad_checkpointing = grad_checkpointing

    def forward(self, appearance, semantics, velocity):
        maps = (appearance, semantics, velocity)
        if appearance.ndim != 4 or appearance.shape[1] != 3 or any(t.shape != appearance.shape for t in maps):
            raise ValueError('Expected matching (B,3,H,W) BEV conditions')
        if any(n % 8 for n in appearance.shape[-2:]):
            raise ValueError('Spatial dimensions must be divisible by eight')
        features = self.input_projection(torch.cat([enc(x) for enc, x in zip(self.encoders, maps)], 1))
        b, c, h, w = features.shape
        freq = torch.exp(-math.log(10000) * torch.arange(c//4, device=features.device).float() / max(c//4-1, 1))
        yy, xx = torch.meshgrid(torch.arange(h, device=features.device), torch.arange(w, device=features.device), indexing='ij')
        px, py = xx[..., None]*freq, yy[..., None]*freq
        pos = torch.cat((px.sin(), px.cos(), py.sin(), py.cos()), -1).to(features.dtype)
        x = features.permute(0, 2, 3, 1)[:, None] + pos[None, None] + self.modality_embedding[None, :, None, None]
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.grad_checkpointing and self.training else block(x)
        x = self.output_norm(x).reshape(b*3, h, w, c).permute(0, 3, 1, 2)
        return self.decoder(self.latent_projection(x)).reshape(b, 3, h*8, w*8)
