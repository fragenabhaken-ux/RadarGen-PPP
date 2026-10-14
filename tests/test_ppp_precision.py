"""Mixed-precision normalization accuracy and decoder activation dtypes."""

import copy
import unittest

import torch

from diffusion.model.dc_ae.efficientvit.models.nn.norm import RMSNorm2d
import test_ppp_decoder_checkpointing as checkpoint_tests


class PrecisionChecks(unittest.TestCase):
    def test_rmsnorm_matches_fp32_reference(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for affine, bias in ((True, True), (True, False), (False, False)):
                with self.subTest(dtype=dtype, affine=affine, bias=bias):
                    torch.manual_seed(42)
                    norm = RMSNorm2d(8, elementwise_affine=affine, bias=bias)
                    if affine:
                        torch.nn.init.uniform_(norm.weight, 0.8, 1.2)
                        if bias:
                            torch.nn.init.uniform_(norm.bias, -0.1, 0.1)
                    reference = copy.deepcopy(norm)
                    x = torch.randn(2, 8, 4, 4).to(dtype).requires_grad_()
                    other = x.detach().clone().requires_grad_()
                    actual = norm(x)
                    expected = other.float() / torch.sqrt(
                        other.float().square().mean(1, keepdim=True) + norm.eps)
                    if affine:
                        expected = expected * reference.weight.view(1, -1, 1, 1)
                        if bias:
                            expected = expected + reference.bias.view(1, -1, 1, 1)
                    expected = expected.to(dtype)
                    self.assertEqual(actual.dtype, dtype)
                    tolerance = dict(rtol=2e-2, atol=2e-2) if dtype != torch.float32 else {}
                    torch.testing.assert_close(actual, expected, **tolerance)
                    upstream = torch.randn_like(actual)
                    actual.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(x.grad, other.grad, **tolerance)
                    for parameter, ref in zip(norm.parameters(), reference.parameters()):
                        self.assertEqual(parameter.dtype, torch.float32)
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        torch.testing.assert_close(parameter.grad, ref.grad, **tolerance)

    def test_rmsnorm_extreme_inputs_are_finite(self):
        norm = RMSNorm2d(4, elementwise_affine=False)
        for scale in (0.0, 1e-6, 1e4):
            x = (torch.ones(2, 4, 3, 3) * scale).to(torch.float16).requires_grad_()
            output = norm(x)
            output.sum().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_decoder_stage_dtypes_and_gradients(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = checkpoint_tests.DecoderCheckpointingChecks().make_decoder().requires_grad_(not frozen)
                model.decoder.gradient_checkpointing = True
                observed = []
                modules = [model.decoder.project_in, *model.decoder.stages,
                           model.decoder.project_out]
                handles = [module.register_forward_hook(
                    lambda module, args, output: observed.append(output.dtype)) for module in modules]
                latent = torch.randn(2, 2, 3, 3, requires_grad=True)
                try:
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        output = model(latent)
                    self.assertEqual(observed, [torch.bfloat16] * len(modules))
                    output.float().square().mean().backward()
                    self.assertTrue(torch.isfinite(latent.grad).all())
                    self.assertGreater(latent.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                    self.assertTrue(all(p.grad is None for p in model.parameters()) if frozen
                                    else all(p.grad is not None and torch.isfinite(p.grad).all()
                                             for p in model.parameters()))
                finally:
                    for handle in handles:
                        handle.remove()


if __name__ == '__main__':
    unittest.main()
