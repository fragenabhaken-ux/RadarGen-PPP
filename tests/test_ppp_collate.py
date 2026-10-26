"""Minimal packed batching and training-loss integration checks."""
import unittest

import torch
from torch.utils.data import DataLoader

from radargen.training.ppp_collate import collate_ppp_samples
from radargen.losses import ContinuousPPPLoss, PPPLoss


def sample(points):
    return dict(detections=torch.tensor(points, dtype=torch.float32).reshape(-1, 4),
                point_limit=1., bev_color_map=torch.zeros(3, 2, 2),
                bev_seg_map=torch.zeros(3, 2, 2), bev_velocity_map=torch.zeros(3, 2, 2),
                data_info=dict(frame_idx=1))


class PackedBatchTests(unittest.TestCase):
    def test_mixed_counts_and_framewise_reduction(self):
        samples = [sample([[0., 0., 1., 2.], [0., 0., 4., -3.]]), sample([]),
                   sample([[-1., -1., 3., 5.]])]
        batch = next(iter(DataLoader(samples, batch_size=3, collate_fn=collate_ppp_samples)))
        self.assertEqual(batch['bev_color_map'].shape, (3, 3, 2, 2))
        self.assertEqual(batch['batch_size'], 3)
        self.assertEqual(batch['frame_indices'].tolist(), [0, 0, 2])
        torch.testing.assert_close(batch['detections'], torch.cat([s['detections'] for s in samples]))
        fields = torch.randn(3, 3, 2, 2, requires_grad=True)
        loss = ContinuousPPPLoss((2, 2))
        actual = loss(fields, batch['detections'], batch['frame_indices'], point_limit=1.)
        individual = [loss(fields[i:i+1], s['detections'],
                           torch.zeros(len(s['detections']), dtype=torch.long), point_limit=1.)
                      for i, s in enumerate(samples)]
        for key in actual:
            torch.testing.assert_close(actual[key], torch.stack([v[key] for v in individual]).mean())
        sum(actual.values()).backward()
        self.assertTrue(torch.isfinite(fields.grad).all())

    def test_entirely_empty_batch_and_single_frame(self):
        for n in (1, 3):
            batch = collate_ppp_samples([sample([]) for _ in range(n)])
            self.assertEqual(batch['batch_size'], n)
            self.assertEqual(batch['detections'].shape, (0, 4))
            self.assertEqual(batch['frame_indices'].shape, (0,))
            losses = ContinuousPPPLoss((2, 2))(torch.zeros(n, 3, 2, 2),
                batch['detections'], batch['frame_indices'], point_limit=batch['point_limit'])
            self.assertEqual(losses['ppp'].item(), 1.)
            self.assertEqual(losses['rcs'].item(), 0.)

    def test_training_loss_entrypoint(self):
        from scripts.train_ppp import compute_ppp_losses
        batch = collate_ppp_samples([sample([[0., 0., 2., 3.]]), sample([])])
        settings = dict(velocity_loss_mode='signed_l1', ppp_weight=2., rcs_weight=3., doppler_weight=4.)
        fields = torch.zeros(2, 3, 2, 2, requires_grad=True)
        for decoded in (fields, fields.unsqueeze(2)):
            total, logs = compute_ppp_losses(decoded, batch, PPPLoss((2, 2)), settings)
            raw = ContinuousPPPLoss((2, 2))(fields, batch['detections'], batch['frame_indices'], point_limit=1.)
            torch.testing.assert_close(total, sum(raw[k] * settings[k+'_weight'] for k in raw))
            self.assertIn('loss_total', logs)
        total.backward()
        self.assertTrue(torch.isfinite(fields.grad).all())
