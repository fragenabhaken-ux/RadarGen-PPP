import numpy as np
import torch
from radargen.models.dit_v2 import DiTV2
from radargen.losses.continuous import ContinuousPPPLoss
from radargen.ppp.sampling import sample_ppp_grid
from radargen.training.ppp_overfit import summarize_fields


def test_factor_eight_and_gradients():
    torch.manual_seed(3)
    model = DiTV2(hidden_size=32, depth=2, latent_channels=8, head_dim=8)
    maps = [torch.randn(1, 3, 32, 32) for _ in range(3)]
    assert model.encoders[0](maps[0]).shape == (1, 8, 4, 4)
    fields = model(*maps)
    assert fields.shape == (1, 3, 32, 32)
    loss = fields.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(any(p.grad.abs().sum() > 0 for p in enc.parameters()) for enc in model.encoders)


def test_identity_loss_sampling_and_diagnostics():
    fields = torch.zeros(1, 3, 8, 8, requires_grad=True)
    with torch.no_grad():
        fields[:, 1] = -2
        fields[:, 2] = 3
        fields[:, 0] = np.log(64)
    targets = torch.tensor([[0., 0., -2., 3.]])
    losses = ContinuousPPPLoss(out_img_size=(8, 8), mark_link='identity')(
        fields, targets, torch.tensor([0]), point_limit=50.)
    assert losses['rcs'] == losses['doppler'] == 0
    points = sample_ppp_grid(*fields[0], 50, torch.Generator().manual_seed(42), mark_link='identity')
    assert len(points) > 0
    assert np.all(points[:, 2] == -2) and np.all(points[:, 3] == 3)
    mask = np.ones((8, 8), dtype=bool)
    stats, _, _ = summarize_fields(fields.detach()[0].numpy(), mask,
        np.full((8,8), -2), np.full((8,8), 3), {'rcs':(-20,66),'doppler':(-120,120)}, mark_link='identity')
    assert stats['rcs_mae_at_targets'] == stats['doppler_mae_at_targets'] == 0


def test_checkpoint_inference_roundtrip(tmp_path):
    from types import SimpleNamespace
    from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
    from scripts.train_ppp import compute_ppp_losses
    from radargen.losses import PPPLoss
    kwargs = dict(hidden_size=32, depth=1, latent_channels=8, head_dim=8,
                  grad_checkpointing=False)
    model = DiTV2(**kwargs).eval()
    path = tmp_path / 'model.pth'
    torch.save(dict(state_dict=model.state_dict(), ppp_training=dict(version=1),
                    epoch=0, rank_rng_states=[{}]), path)
    settings = dict(mark_link='identity', velocity_loss_mode='signed_l1',
                    ppp_weight=1., rcs_weight=1., doppler_weight=1.)
    config = SimpleNamespace(model=SimpleNamespace(model='DiTV2', extra=kwargs,
        image_size=32, mixed_precision='no', load_from=str(path), resume_from=None),
        train=SimpleNamespace(grad_checkpointing=False, extra=settings),
        data=SimpleNamespace(extra=dict(coordinate_range=50.)), resume_from=None, load_from=None)
    pipeline = RadarGenPPPPipeline.from_config(config, device='cpu')
    maps = [torch.randn(1, 3, 32, 32) for _ in range(3)]
    with torch.no_grad():
        expected = model(*maps)[0]
    fields = pipeline.predict_fields(*maps)
    assert pipeline.mark_link == 'identity'
    for i, name in enumerate(('log_intensity', 'rcs_raw', 'doppler_raw')):
        torch.testing.assert_close(fields[name], expected[i])
    # Exercise the training adapter with exact signed target marks.
    decoded = torch.zeros(1, 3, 4, 4, requires_grad=True)
    with torch.no_grad():
        decoded[:, 1] = -2
        decoded[:, 2] = 3
    batch = dict(batch_size=1, detections=torch.tensor([[0., 0., -2., 3.]]),
                 frame_indices=torch.tensor([0]), point_limit=50.)
    total, logs = compute_ppp_losses(decoded, batch, PPPLoss((4, 4)), settings)
    assert logs['loss_rcs_raw'] == logs['loss_doppler_raw'] == 0
    total.backward()
    assert torch.isfinite(decoded.grad).all()
