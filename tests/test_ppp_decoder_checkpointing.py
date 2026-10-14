"""Checkpoint real DC-AE stages without pretrained assets or a GPU."""

import copy
import unittest
from unittest.mock import patch

import torch

from diffusion.model.dc_ae.efficientvit.models.efficientvit import dc_ae
from diffusion.model.dc_ae.efficientvit.models.nn.norm import RMSNorm2d
from radargen.training.ppp_decoder import PPPDecoder


class DecoderCheckpointingChecks(unittest.TestCase):
    def make_decoder(self):
        model = PPPDecoder(dc_ae.Decoder(dc_ae.DecoderConfig(
            in_channels=1, latent_channels=2, width_list=(4, 8),
            depth_list=(1, 1), block_type='ResBlock',
        )), 0.41407)
        # Production loads pretrained norms; RMSNorm2d allocates uninitialized
        # affine tensors when constructed without a checkpoint.
        for module in model.modules():
            if isinstance(module, RMSNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        return model

    def test_outputs_and_gradients_match(self):
        for frozen, input_grad, autocast in (
            (False, True, False), (True, True, False),
            (False, False, False), (False, True, True), (True, True, True),
        ):
            with self.subTest(frozen=frozen, input_grad=input_grad, autocast=autocast):
                torch.manual_seed(23)
                plain = self.make_decoder().requires_grad_(not frozen)
                checked = copy.deepcopy(plain)
                checked.decoder.gradient_checkpointing = True
                latent = torch.randn(2, 2, 3, 3, requires_grad=input_grad)
                other = latent.detach().clone().requires_grad_(input_grad)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
                    expected = plain(latent)
                    actual = checked(other)
                torch.testing.assert_close(actual, expected)
                expected.square().mean().backward()
                actual.square().mean().backward()
                if input_grad:
                    torch.testing.assert_close(other.grad, latent.grad)
                    self.assertGreater(other.grad.abs().sum().item(), 0)
                self.assertEqual(plain.state_dict().keys(), checked.state_dict().keys())
                for reference, parameter in zip(plain.parameters(), checked.parameters()):
                    if frozen:
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertIsNotNone(parameter.grad)
                        torch.testing.assert_close(parameter.grad, reference.grad)

    def test_stages_are_checkpointed_and_no_grad_bypasses(self):
        model = self.make_decoder()
        model.decoder.gradient_checkpointing = True
        latent = torch.randn(1, 2, 3, 3, requires_grad=True)
        with patch.object(dc_ae, 'checkpoint', wraps=dc_ae.checkpoint) as wrapped:
            model(latent).sum().backward()
            modules = [call.args[0] for call in wrapped.call_args_list]
            self.assertEqual(modules, [model.decoder.project_in,
                                      *reversed(model.decoder.stages),
                                      model.decoder.project_out])
            wrapped.reset_mock()
            with torch.no_grad():
                model(latent)
            wrapped.assert_not_called()

    def test_checkpointing_reduces_saved_tensors(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = self.make_decoder().requires_grad_(not frozen)

                def saved_bytes(enabled):
                    model.decoder.gradient_checkpointing = enabled
                    saved = []

                    def pack(tensor):
                        saved.append(tensor.numel() * tensor.element_size())
                        return tensor

                    latent = torch.randn(2, 2, 8, 8, requires_grad=True)
                    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                        output = model(latent)
                    self.assertTrue(output.requires_grad)
                    return sum(saved)

                self.assertLess(saved_bytes(True), saved_bytes(False))


if __name__ == '__main__':
    unittest.main()
