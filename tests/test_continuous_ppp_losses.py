"""Analytical checks for detection-wise supervision, including collisions."""
import math
import unittest
import torch
from radargen.losses import ContinuousPPPLoss, PPPLoss, masked_rcs_l1, masked_doppler_loss


class ContinuousLossTests(unittest.TestCase):
    def test_collision_preserves_marks_and_gradient(self):
        fields = torch.zeros(2, 3, 2, 2, requires_grad=True)
        points = torch.tensor([[0., 0., 2., -3.], [0., 0., -4., 5.]])
        loss = ContinuousPPPLoss((2, 2))(fields, points, torch.tensor([0, 0]), point_limit=1.)
        self.assertAlmostEqual(loss['ppp'].item(), (0.5 + math.log(4) + 1) / 2, places=6)
        self.assertEqual(loss['rcs'].item(), 1.5)
        self.assertEqual(loss['doppler'].item(), 2.)
        loss['ppp'].backward()
        self.assertAlmostEqual(fields.grad[0, 0, 0, 1].item(), -.4375)

    def test_no_collision_matches_grid_values_and_gradients(self):
        fields = torch.randn(2, 3, 2, 2, requires_grad=True)
        points = torch.tensor([[-1., -1., 2., -3.], [1., 1., -4., 5.]])
        indices = torch.tensor([0, 0])
        mask = torch.zeros(2, 1, 2, 2)
        rcs, doppler = torch.zeros_like(mask), torch.zeros_like(mask)
        mask[0, 0, 1, 0] = mask[0, 0, 0, 1] = 1
        rcs[0, 0, 1, 0], rcs[0, 0, 0, 1] = 2, -4
        doppler[0, 0, 1, 0], doppler[0, 0, 0, 1] = -3, 5
        actual = ContinuousPPPLoss((2, 2))(fields, points, indices, point_limit=1.)
        expected = dict(ppp=PPPLoss((2, 2))(fields[:, :1], mask),
                        rcs=masked_rcs_l1(fields[:, 1:2], rcs, mask),
                        doppler=masked_doppler_loss(fields[:, 2:3], doppler, mask, 'signed_l1'))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name])
            torch.testing.assert_close(torch.autograd.grad(actual[name], fields, retain_graph=True)[0],
                                       torch.autograd.grad(expected[name], fields, retain_graph=True)[0])

    def test_empty_batch_and_outside_points(self):
        for points, indices in ((torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
                                (torch.tensor([[2., 0., 4., 5.]]), torch.tensor([0]))):
            fields = torch.zeros(3, 3, 2, 2, requires_grad=True)
            losses = ContinuousPPPLoss((2, 2))(fields, points, indices, point_limit=1.)
            self.assertEqual(losses['ppp'].item(), 1.)
            self.assertEqual(losses['rcs'].item(), 0.)
            self.assertEqual(losses['doppler'].item(), 0.)
            sum(losses.values()).backward()
            self.assertTrue(torch.isfinite(fields.grad).all())
