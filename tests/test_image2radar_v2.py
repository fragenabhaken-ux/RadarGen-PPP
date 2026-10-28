"""Geometry, locality, and end-to-end gradient checks without pretrained assets."""
import math
import torch
from torch import nn

from radargen.image2radar_v2.model import (
    Image2RadarV2, AngularAttention, WindowAttention, bev_centers, camera_geometry,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (4, 4)
        self.patch_embed.proj = nn.Conv2d(3, 12, 4, stride=4)

    def set_grad_checkpointing(self, enabled):
        pass

    def forward_intermediates(self, x, **kwargs):
        return [self.patch_embed.proj(x)]


def test_bev_and_partial_patch_geometry():
    torch.testing.assert_close(bev_centers(2, 2, 'cpu'), torch.tensor([[-1., 1.], [1., 1.], [-1., -1.], [1., -1.]]))
    k = torch.eye(3).reshape(1, 1, 3, 3)
    transform = torch.eye(4).reshape(1, 1, 4, 4)
    transform[..., 0, 3] = -3
    origins, azimuth, elevation = camera_geometry(k, transform, (5, 7), (2, 2), 4)
    torch.testing.assert_close(origins, torch.tensor([[[3., 0., 0.]]]))
    # Last footprint is [4,7] x [4,5], not the padded patch center (6,6).
    torch.testing.assert_close(azimuth[0, 0, -1], torch.atan2(torch.tensor(4.5), torch.tensor(5.5)))
    assert torch.isfinite(elevation).all()


def test_cross_attention_wraparound_and_uncovered_gradient():
    layer = AngularAttention(8, 8, 2, .1, 4, 2, 2.)
    x = torch.zeros(1, 1, 1, 8, requires_grad=True)
    camera = torch.randn(1, 1, 2, 8, requires_grad=True)
    # Query is at (0,0), camera at (1,-.01): query azimuth just below +pi.
    origins = torch.tensor([[[1., -.01, 0.]]])
    angles = torch.tensor([[[-math.pi + .01, 0.]]])
    geometry = (origins, angles, torch.zeros_like(angles))
    result = layer(x, camera, geometry)
    result.sum().backward()
    assert camera.grad[0, 0, 0].abs().sum() > 0
    assert camera.grad[0, 0, 1].abs().sum() == 0
    uncovered = (origins, torch.zeros_like(angles), torch.zeros_like(angles))
    torch.testing.assert_close(layer(x, camera, uncovered), x)


def test_shifted_windows_do_not_wrap_edges():
    layer = WindowAttention(8, 2, 4, shifted=True)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    layer(x)[0, 0, 0].sum().backward()
    assert x.grad[0, -1].abs().sum() == 0
    assert x.grad[0, :, -1].abs().sum() == 0


def test_small_model_checkpointed_backward_and_batch_independence():
    torch.manual_seed(3)
    model = Image2RadarV2(backbone=TinyBackbone(), use_lora=False,
        widths=(16, 8, 8, 4), heads=(4, 2, 2), depths=(2, 2), query_size=2,
        output_size=16, window_size=2, max_keys=4, query_chunk_size=8,
        angular_radius_degrees=179, grad_checkpointing=True)
    images = torch.randint(0, 256, (2, 4, 3, 7, 7), dtype=torch.uint8)
    k = torch.tensor([[5., 0., 3.], [0., 5., 3.], [0., 0., 1.]]).repeat(2, 4, 1, 1)
    transforms = torch.eye(4).repeat(2, 4, 1, 1)
    model.train()
    output = model(images, k, transforms)
    assert output.shape == (2, 3, 16, 16) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.query.grad.abs().sum() > 0
    assert model.adapter[0].weight.grad.abs().sum() > 0
    assert model.fine_cross.kv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    model.eval()
    with torch.no_grad():
        together = model(images, k, transforms)
        separate = torch.cat([model(images[i:i+1], k[i:i+1], transforms[i:i+1]) for i in range(2)])
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)


def test_initializer_uses_v2_and_preserves_target_extent():
    from types import SimpleNamespace
    from unittest.mock import patch
    from radargen.training.ppp_initialization import initialize_dino_ppp_model
    config = SimpleNamespace(
        model=SimpleNamespace(model='Image2RadarV2', extra={}),
        data=SimpleNamespace(extra={'coordinate_range': 50.}),
        train=SimpleNamespace(grad_checkpointing=True))
    with patch('radargen.image2radar_v2.Image2RadarV2', return_value=nn.Linear(1, 1)) as constructor:
        components = initialize_dino_ppp_model(config, load_pretrained=False)
    assert components.condition_encoder is None
    assert constructor.call_args.kwargs == dict(checkpoint_path='', grad_checkpointing=True, coordinate_range=50.)
    config.model.extra = {'coordinate_range': 75.}
    import pytest
    with pytest.raises(ValueError, match='coordinate ranges'):
        initialize_dino_ppp_model(config, load_pretrained=False)


def test_training_dispatch_passes_calibrated_cameras_without_bev_conditions():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from scripts.train_ppp import training_microbatch
    batch = {key: torch.tensor([index]) for index, key in enumerate(
        ('camera_images', 'camera_intrinsics', 'reference_to_camera'))}
    model = Mock(return_value=torch.ones(1, 3, 2, 2))
    config = SimpleNamespace(model=SimpleNamespace(model='Image2RadarV2'), train=SimpleNamespace(extra={}))
    accelerator = SimpleNamespace(autocast=nullcontext)
    with patch('scripts.train_ppp.compute_ppp_losses', return_value=('loss', {})) as losses:
        assert training_microbatch(model, None, batch, None, None, config, accelerator) == ('loss', {})
    model.assert_called_once_with(*batch.values())
    assert losses.call_args.args[0] is model.return_value
