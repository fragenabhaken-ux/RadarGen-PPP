"""Attention decoder, full finetuning and continuous high-resolution contracts."""
import tempfile
import unittest
from pathlib import Path

import torch

import test_dino_ppp_model as baseline
from radargen.models.dino_ppp import load_dino_features
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
from radargen.losses.continuous import ContinuousPPPLoss
from radargen.ppp.sampling import ppp_cell_masses


class AttentionTests(unittest.TestCase):
    def build(self, **kwargs):
        options = dict(decoder='attention', frozen_base=False, output_scale=2,
                       feature_fusion='last_concat', out_indices=(1,), embed_dim=36,
                       attention_channels=(8, 4, 4), attention_blocks=1,
                       attention_heads=2, attention_window_size=4)
        options.update(kwargs)
        return baseline.DinoPPPTests().build(**options)

    def test_rectangular_output_and_full_backbone_gradients(self):
        model = self.build(grad_checkpointing=True)
        maps = [torch.randn(1, 3, 17, 19) for _ in range(3)]
        output = model(*maps)
        self.assertEqual(output.shape, (1, 3, 34, 38))
        detections = torch.tensor([[0., 0., 2., 3.], [0., 0., 5., -2.]])
        losses = ContinuousPPPLoss((34, 38))(output, detections, torch.zeros(2, dtype=torch.long), point_limit=50.)
        sum(losses.values()).backward()
        for parameter in (model.backbone.patch_embed.proj.weight,
                          model.backbone.blocks[0].attn['qkv'].linear.weight,
                          model.head.stages[0][2].attn.in_proj_weight):
            self.assertTrue(parameter.requires_grad)
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)
        model.train()
        self.assertTrue(model.backbone.training)

    def test_transfer_excludes_head_and_rejects_missing_features(self):
        source = baseline.DinoPPPTests().build()
        target = self.build()
        fresh = {k: v.clone() for k, v in target.state_dict().items() if not k.startswith('backbone.')}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            saved = source.state_dict()
            torch.save(dict(ppp_training=dict(version=1), state_dict=saved), path)
            load_dino_features(path, target)
            for key, value in source.backbone.state_dict().items():
                torch.testing.assert_close(target.backbone.state_dict()[key], value)
            for key, value in fresh.items():
                torch.testing.assert_close(target.state_dict()[key], value)
            del saved['backbone.patch_embed.proj.weight']
            torch.save(dict(ppp_training=dict(version=1), state_dict=saved), path)
            with self.assertRaises(ValueError):
                load_dino_features(path, target)

    def test_final_concat_reaches_neck_without_projection(self):
        model = self.build().eval()
        self.assertEqual(len(model.layer_projs), 0)
        self.assertIsInstance(model.fuse, torch.nn.Identity)
        self.assertFalse(any(k.startswith(('layer_projs.', 'fuse.')) for k in model.state_dict()))
        captured = []
        handle = model.neck.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        calls = []

        def features(x, **kwargs):
            calls.append(kwargs['indices'])
            return [x[:, :1, :1, :1].expand(-1, 12, 1, 1)]

        from unittest.mock import patch
        maps = [torch.full((2, 3, 16, 16), float(i)) for i in range(3)]
        with patch.object(model.backbone, 'forward_intermediates', side_effect=features):
            model(*maps)
        handle.remove()
        self.assertEqual(calls, [[1]])
        self.assertEqual(captured[0].shape, (2, 36, 1, 1))
        for i in range(3):
            torch.testing.assert_close(captured[0][:, i*12:(i+1)*12], torch.full((2, 12, 1, 1), float(i)))
        self.assertEqual(model.neck.layers[0].linear1.in_features, 36)
        self.assertEqual(model.neck.layers[0].linear2.out_features, 36)
        self.assertEqual(model.head.stages[0][0].in_channels, 36)

    def test_final_concat_rejects_compression_or_intermediate_blocks(self):
        with self.assertRaisesRegex(ValueError, 'width'):
            self.build(embed_dim=24)
        with self.assertRaisesRegex(ValueError, 'final DINO block'):
            self.build(out_indices=(0, 1))

    def test_inference_keeps_high_resolution_fields(self):
        pipeline = RadarGenPPPPipeline(self.build(), resolution=16, device='cpu', mixed_precision='no')
        fields = pipeline.predict_fields(*[torch.randn(3, 16, 16) for _ in range(3)])
        self.assertEqual(fields['log_intensity'].shape, (32, 32))
        self.assertEqual(pipeline.sample_fields(fields).shape[1], 4)

    def test_window_chunking_preserves_outputs_and_gradients(self):
        import copy
        from radargen.models.dino_ppp import WindowResidualAttentionBlock
        reference = WindowResidualAttentionBlock(8, heads=2, window_size=4)
        chunked = copy.deepcopy(reference)
        chunked.window_batch_size = 2
        x = torch.randn(2, 8, 9, 11, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected, actual = reference(x), chunked(other)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(other.grad, x.grad)
        for left, right in zip(reference.parameters(), chunked.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

    def test_shifted_windows_cross_boundaries_without_wrapping(self):
        from radargen.models.dino_ppp import WindowResidualAttentionBlock

        def uniform_attention(shift):
            block = WindowResidualAttentionBlock(4, heads=1, window_size=4, shift_size=shift)
            block.local = torch.nn.Identity()
            block.norm = torch.nn.Identity()
            block.mlp = torch.nn.Linear(4, 4)
            with torch.no_grad():
                for parameter in block.parameters():
                    parameter.zero_()
                block.attn.in_proj_weight[8:].copy_(torch.eye(4))
                block.attn.out_proj.weight.copy_(torch.eye(4))
            return block

        regular, shifted = uniform_attention(0), uniform_attention(2)
        x = torch.zeros(1, 4, 4, 8)
        x[:, :, 1, 3] = 1
        self.assertEqual(regular(x)[0, 0, 1, 4].item(), 0)
        self.assertGreater(shifted(x)[0, 0, 1, 4].item(), 0)
        x.zero_()
        x[:, :, 1, 7] = 1
        self.assertEqual(shifted(x)[0, 0, 1, 0].item(), 0)
        # Padding keys must not dilute constant features at the BEV edges.
        torch.testing.assert_close(shifted(torch.ones_like(x)), torch.full_like(x, 2))

    def test_shifted_chunking_and_checkpointed_decoder(self):
        import copy
        from radargen.models.dino_ppp import WindowResidualAttentionBlock
        reference = WindowResidualAttentionBlock(8, heads=2, window_size=4, shift_size=2)
        chunked = copy.deepcopy(reference)
        chunked.window_batch_size = 2
        x = torch.randn(2, 8, 9, 11, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected, actual = reference(x), chunked(other)
        torch.testing.assert_close(expected, actual)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad)
        for left, right in zip(reference.parameters(), chunked.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

        model = self.build(attention_blocks=2, attention_projection_kernel_size=3,
                           attention_shifted_windows=True, grad_checkpointing=True)
        for stage in model.head.stages:
            self.assertEqual(stage[0].kernel_size, (3, 3))
            self.assertEqual([block.shift_size for block in stage[2:]], [0, 2])
            phases = stage[1].upscale_factor**2
            for phase in range(1, phases):
                torch.testing.assert_close(stage[0].weight[phase::phases], stage[0].weight[::phases])
        maps = [torch.randn(1, 3, 17, 19) for _ in range(3)]
        output = model(*maps)
        self.assertEqual(output.shape, (1, 3, 34, 38))
        output.square().mean().backward()
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_resolution_preserves_integral_and_duplicate_mark_supervision(self):
        points = torch.tensor([[0., 0., 1., 2.], [0., 0., 3., 4.]])
        losses = []
        for size in (16, 32):
            fields = torch.zeros(1, 3, size, size, requires_grad=True)
            loss = ContinuousPPPLoss((size, size))(fields, points, torch.zeros(2, dtype=torch.long), point_limit=50.)
            self.assertEqual(loss['rcs'].item(), 2.)
            self.assertEqual(loss['doppler'].item(), 3.)
            self.assertAlmostEqual(ppp_cell_masses(fields[0, 0]).sum().item(), 1.)
            losses.append(loss['ppp'])
        # Existing likelihood uses per-cell masses: NLL differs by log(cell-count ratio).
        torch.testing.assert_close(losses[1] - losses[0], torch.tensor(4.).log())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
