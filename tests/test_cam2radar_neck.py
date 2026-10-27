"""A small joint-view backward pass checks learned geometry, not just shapes."""
import torch
from radargen.cam2radar.neck import MultiCameraRayNeck
from radargen.cam2radar.fusion import CameraChannelFusion
from radargen.cam2radar.model import CartesianHead


def test_joint_neck_depth_gradients_and_channel_bev_readout():
    torch.manual_seed(4)
    x = torch.randn(1, 4, 48, 2, 2, requires_grad=True)
    k = torch.tensor([[25., 0., 16.], [0., 25., 16.], [0., 0., 1.]]).repeat(1, 4, 1, 1)
    t = torch.eye(4).repeat(1, 4, 1, 1)
    t[0, :, 0, 3] = torch.tensor([0., 1., -1., 2.])
    neck = MultiCameraRayNeck(48, 2, 1, grad_checkpointing=True)
    y = neck(x, k, t, (32, 32))
    fields = CartesianHead(48, output_size=16, widths=(8, 8, 4, 4))(CameraChannelFusion(48)(y))
    fields.square().mean().backward()
    assert fields.shape == (1, 3, 16, 16)
    assert torch.isfinite(fields).all()
    assert neck.layers[0].depth.weight.grad.norm() > 0
    assert x.grad[0, 3].norm() > 0
