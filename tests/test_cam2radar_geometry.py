"""Pixel-to-ray geometry with distinct translated and rotated cameras."""
import torch

from radargen.cam2radar.input import CameraInput, patch_center_rays


def test_padded_patch_rays_reproject_to_the_original_pixel_coordinates():
    images = torch.full((1, 2, 3, 31, 45), 127, dtype=torch.uint8)
    prepared, native = CameraInput()(images)
    assert native == (31, 45)
    assert prepared.shape[-2:] == (32, 48)
    assert torch.count_nonzero(prepared[..., 31:, :]) == 0
    assert torch.count_nonzero(prepared[..., :, 45:]) == 0
    k = torch.tensor([[40., 0., 22.], [0., 42., 15.], [0., 0., 1.]]).expand(1, 2, 3, 3)
    transforms = torch.eye(4).repeat(1, 2, 1, 1)
    transforms[0, 1, :3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    transforms[0, 1, :3, 3] = torch.tensor([2., -1., 3.])
    origins, rays = patch_center_rays(k, transforms, prepared.shape[-2:])
    points = origins[:, :, None] + 10 * rays
    camera_points = torch.einsum('bvij,bvpj->bvpi', transforms[..., :3, :3], points)
    camera_points += transforms[..., :3, 3].unsqueeze(-2)
    projected = torch.einsum('bvij,bvpj->bvpi', k, camera_points)
    pixels = projected[..., :2] / projected[..., 2:]
    expected = torch.tensor([[8., 8.], [24., 8.], [40., 8.],
                             [8., 24.], [24., 24.], [40., 24.]])
    torch.testing.assert_close(pixels, expected.expand(1, 2, 6, 2), atol=1e-5, rtol=1e-5)
