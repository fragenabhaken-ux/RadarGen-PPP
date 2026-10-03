"""CPU-only analytical checks for the standalone WP3 losses."""

import importlib.abc
import math
import sys
import unittest


class _NoModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mmseg', 'mmcv', 'transformers', 'diffusers', 'accelerate'}:
            raise AssertionError(f'Unexpected dependency: {fullname}')


sys.meta_path.insert(0, _NoModelDependencies())

import torch

from radargen.losses import PPPLoss, masked_doppler_loss, masked_rcs_l1

AREA = 512 * 512


def grid(rows, *, dtype=torch.float32, requires_grad=False):
    return torch.tensor(rows, dtype=dtype).reshape(len(rows), 1, 1, -1).requires_grad_(requires_grad)


class PPPLossTests(unittest.TestCase):
    def test_both_ppp_reductions_values_and_gradients(self):
        values = [[.2, -.4, .7], [-.3, .5, -.8]]
        occupied = [[1, 0, 0], [1, 1, 0]]
        for normalize in (True, False):
            with self.subTest(normalize=normalize):
                raw = grid(values, requires_grad=True)
                mask = grid(occupied)
                loss_fn = PPPLoss((512, 512), normalize_by_gt_count=normalize)
                self.assertEqual(loss_fn.state_dict()['pixel_scale'].item(), AREA)
                self.assertEqual(loss_fn.pixel_scale.dtype, torch.float32)
                expected, gradients = 0., []
                for row, active in zip(values, occupied):
                    denominator = max(sum(active), 1) if normalize else 1
                    integral = sum(math.exp(f) / AREA for f in row)
                    observation = sum(f - math.log(AREA) for f, m in zip(row, active) if m)
                    expected += (integral - observation) / denominator / 2
                    gradients.append([(math.exp(f)/AREA-m)/denominator/2 for f, m in zip(row, active)])
                loss = loss_fn(raw, mask)
                self.assertAlmostEqual(loss.item(), expected, places=5)
                loss.backward()
                torch.testing.assert_close(raw.grad, grid(gradients), rtol=1e-6, atol=1e-8)

    def test_ppp_empty_mask_integral_and_gradient(self):
        for normalize in (True, False):
            raw = grid([[0., 0.], [0., 0.]], requires_grad=True)
            loss = PPPLoss(normalize_by_gt_count=normalize)(raw, torch.zeros_like(raw))
            self.assertEqual(loss.item(), 2 / AREA)
            loss.backward()
            torch.testing.assert_close(raw.grad, torch.full_like(raw, 1 / (2*AREA)), rtol=0, atol=0)

    def test_mark_values_gradients_unequal_counts_and_empty_sample(self):
        values = [[1., -1., 2.], [.5, -2., 1.5], [2., -3., 1.]]
        targets = [[0., -4., 999.], [999., -6., 999.], [0., 0., 0.]]
        occupied = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
        expected, gradients = 0., []
        for row, gt, active in zip(values, targets, occupied):
            denominator = max(sum(active), 1)
            expected += sum(abs(q**3-t) for q, t, m in zip(row, gt, active) if m) / denominator / 3
            gradients.append([m * 3*q*q * (1 if q**3 > t else -1) / denominator / 3
                              for q, t, m in zip(row, gt, active)])
        for name, loss_fn in [('rcs', masked_rcs_l1), ('doppler', lambda q,t,m: masked_doppler_loss(q,t,m,'signed_l1'))]:
            with self.subTest(loss=name):
                raw = grid(values, requires_grad=True)
                loss = loss_fn(raw, grid(targets), grid(occupied))
                self.assertAlmostEqual(loss.item(), expected, places=6)
                loss.backward()
                torch.testing.assert_close(raw.grad, grid(gradients), rtol=1e-6, atol=1e-7)
                self.assertNotEqual(raw.grad[0,0,0,0].item(), 0)  # Active zero-valued GT is supervised.

    def test_mark_empty_masks_zero_loss_and_gradient(self):
        for name, loss_fn in [('rcs', masked_rcs_l1),
                              ('signed_l1', lambda q,t,m: masked_doppler_loss(q,t,m,'signed_l1')),
                              ('signed_circular', lambda q,t,m: masked_doppler_loss(q,t,m,'signed_circular', 10.))]:
            with self.subTest(loss=name):
                raw = grid([[1., -2.]], requires_grad=True)
                loss = loss_fn(raw, grid([[0., 1.]]), torch.zeros_like(raw))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(raw.grad, torch.zeros_like(raw), rtol=0, atol=0)

    def test_circular_sample_periods_values_and_gradients(self):
        raw = grid([[1., -1., 2.], [-1., 2., -2.]], requires_grad=True)
        target = grid([[-2., 0., 1.], [2., 0., 1.]])
        mask = grid([[1, 1, 0], [1, 1, 1]])
        # Physical errors [3,-1,7] and [-3,8,-9] wrap to [-1,-1,-1]
        # and [-3,-2,1] for periods 4 and 10. All active errors avoid cusps.
        loss = masked_doppler_loss(raw, target, mask, 'signed_circular', torch.tensor([4., 10.]))
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        torch.testing.assert_close(raw.grad, grid([[-.75,-.75,0.],[-.5,-2.,2.]]), rtol=0, atol=0)
        shaped_period = torch.tensor([4.,10.]).reshape(2,1,1,1)
        self.assertEqual(masked_doppler_loss(raw, target, mask, 'signed_circular', shaped_period).item(), 1.5)

    def test_circular_scalar_multiwrap(self):
        # Both positive and negative errors cross multiple synthetic periods.
        raw = grid([[2., -2.]], requires_grad=True)
        loss = masked_doppler_loss(raw, grid([[.5, -.5]]), torch.ones_like(raw), 'signed_circular', 3.)
        # Errors +/-7.5 are at the half-period cusp: check value only here.
        self.assertEqual(loss.item(), 1.5)
        raw = grid([[2., -2.]], requires_grad=True)
        loss = masked_doppler_loss(raw, grid([[1., -1.]]), torch.ones_like(raw), 'signed_circular', 3.)
        self.assertEqual(loss.item(), 1.)  # +/-7 wraps to +/-1, away from cusps.
        loss.backward()
        torch.testing.assert_close(raw.grad, grid([[6., -6.]]), rtol=0, atol=0)

    def test_invalid_shapes_masks_and_nonfinite_inputs(self):
        prediction = grid([[1., -1.]])
        mask = torch.ones_like(prediction)
        functions = [lambda q,m,t: PPPLoss()(q,m),
                     lambda q,m,t: masked_rcs_l1(q,t,m),
                     lambda q,m,t: masked_doppler_loss(q,t,m,'signed_l1')]
        bad_cases = [(prediction[0], mask, prediction),
                     (prediction.expand(1,2,1,2), mask, prediction),
                     (prediction[:0], mask[:0], prediction[:0]),
                     (prediction, mask[..., :1], prediction),
                     (prediction, grid([[.5,1.]]), prediction),
                     (prediction, grid([[1.+1e-12,1.]], dtype=torch.float64), prediction),
                     (grid([[float('nan'),1.]]), mask, prediction),
                     (prediction, grid([[float('inf'),1.]]), prediction),
                     (prediction.long(), mask, prediction)]
        for loss_fn in functions:
            for q,m,t in bad_cases:
                with self.subTest(shape=q.shape, mask=m.tolist()):
                    with self.assertRaises(ValueError):
                        loss_fn(q,m,t)
        for loss_fn in (masked_rcs_l1, lambda q,t,m: masked_doppler_loss(q,t,m,'signed_l1')):
            for target in (None, [0.,0.], prediction.to(torch.complex64),
                           grid([[float('nan'),0.]]), grid([[float('inf'),0.]]), prediction[..., :1]):
                with self.assertRaises(ValueError):
                    loss_fn(prediction, target, mask)

    def test_invalid_period_mode_and_reference_size(self):
        raw = grid([[1., -1.]])
        for period in (None, 0., -1., float('nan'), float('inf'), True, 1e40, 1e-50,
                       torch.tensor([3.,4.]), torch.ones(1,1,1,2), torch.tensor(1+2j)):
            with self.subTest(period=period), self.assertRaises(ValueError):
                masked_doppler_loss(raw, torch.zeros_like(raw), torch.ones_like(raw), 'signed_circular', period)
        with self.assertRaises(ValueError):
            masked_doppler_loss(raw, raw, raw, 'absolute')
        for size in ((0,512), (512,-1), (512,), (True,512), (512.,512)):
            with self.assertRaises(ValueError):
                PPPLoss(size)
        with self.assertRaises(ValueError):
            PPPLoss(normalize_by_gt_count=1)

    def test_transformation_and_reduction_overflow_rejected_even_when_masked(self):
        for value in (100., 1e300):
            raw = grid([[value]], dtype=torch.float64)
            with self.assertRaises(ValueError):
                PPPLoss()(raw, torch.zeros_like(raw))
        with self.assertRaises(ValueError):
            PPPLoss((1,1))(grid([[88.,88.,88.,88.]]), grid([[0.,0.,0.,0.]]))
        for loss_fn in (masked_rcs_l1, lambda q,t,m: masked_doppler_loss(q,t,m,'signed_l1')):
            raw = grid([[1e20]])
            with self.assertRaises(ValueError):
                loss_fn(raw, torch.zeros_like(raw), torch.zeros_like(raw))
            raw = grid([[6e12,6e12]])
            with self.assertRaises(ValueError):
                loss_fn(raw, torch.zeros_like(raw), torch.ones_like(raw))

    def test_fp32_math_and_autograd_with_reduced_precision_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.autocast('cpu', dtype=torch.bfloat16):
                raw = grid([[12.,12.]], dtype=dtype, requires_grad=True)
                mask = torch.ones_like(raw)
                loss = PPPLoss()(raw, mask)
                expected = math.exp(12.)/AREA - 12. + math.log(AREA)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertAlmostEqual(loss.item(), expected, places=5)
                loss.backward()
                expected_gradient = torch.full_like(raw, (math.exp(12.)/AREA-1)/2)
                torch.testing.assert_close(raw.grad, expected_gradient, rtol=0, atol=0)
                for loss_fn in (masked_rcs_l1, lambda q,t,m: masked_doppler_loss(q,t,m,'signed_l1'),
                                lambda q,t,m: masked_doppler_loss(q,t,m,'signed_circular',1e6)):
                    q = grid([[50.,-50.]], dtype=dtype, requires_grad=True)
                    result = loss_fn(q, torch.zeros_like(q), mask)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.item(), 125000.)
                    result.backward()
                    torch.testing.assert_close(q.grad, grid([[3750.,-3750.]], dtype=dtype), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
