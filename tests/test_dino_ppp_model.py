"""Shared RGB features, modality-specific LoRA, and BEV reconstruction contracts."""

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from radargen.models import DinoPPP
from radargen.models.dino_ppp import LoRALinear


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.ModuleDict({"qkv": nn.Linear(dim, 3 * dim), "proj": nn.Linear(dim, dim)})
        self.mlp = nn.ModuleDict({"fc1": nn.Linear(dim, dim), "fc2": nn.Linear(dim, dim)})

    def forward(self, x):
        q, k, v = self.attn["qkv"](x).chunk(3, dim=-1)
        x = x + self.attn["proj"]((q + k + v) / 3)
        return x + self.mlp["fc2"](nn.functional.gelu(self.mlp["fc1"](x)))


class TinyDino(nn.Module):
    """Small differentiable substitute for timm's intermediate-feature API."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 12, 16, stride=16)
        self.blocks = nn.ModuleList([TinyBlock(12), TinyBlock(12)])
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enabled):
        self.grad_checkpointing = enabled

    def forward_intermediates(self, x, indices, norm, output_fmt, intermediates_only):
        x = self.patch_embed.proj(x)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        features = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in indices:
                features.append(x.transpose(1, 2).reshape(batch, channels, height, width))
        return features


class DinoPPPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def build(self, **kwargs):
        original = TinyDino()
        options = dict(
            out_indices=(0, 1), per_layer_channels=4, embed_dim=24, neck_layers=1,
            neck_heads=3, stage_channels=(8, 8, 4, 4), refine_channels=4,
            num_refine_convs=1, local_refine_blocks=1, lora_r=2,
        )
        options.update(kwargs)
        with patch.dict(sys.modules, {"timm": SimpleNamespace(create_model=lambda *args, **options: original)}):
            model = DinoPPP(**options)
        return model

    def test_rgb_checkpoint_projection_is_preserved_and_frozen(self):
        original = TinyDino()
        original_stem = original.patch_embed.proj
        weight = original.patch_embed.proj.weight.detach().clone()
        bias = original.patch_embed.proj.bias.detach().clone()
        captured = {}

        def create_model(name, **kwargs):
            captured.update(kwargs)
            self.assertEqual(original.patch_embed.proj.in_channels, 3)
            return original

        with patch.dict(sys.modules, {"timm": SimpleNamespace(create_model=create_model)}):
            model = DinoPPP(
                checkpoint_path="local.safetensors", out_indices=(0, 1), embed_dim=24,
                neck_layers=0, neck_heads=3, refine_channels=4, local_refine_blocks=0,
                stage_channels=(8, 8, 4, 4),
            )
        self.assertEqual(captured["checkpoint_path"], "local.safetensors")
        self.assertEqual(captured["in_chans"], 3)
        stem = model.backbone.patch_embed.proj
        self.assertIs(stem, original_stem)
        self.assertEqual(stem.in_channels, 3)
        torch.testing.assert_close(stem.weight, weight, atol=0, rtol=0)
        torch.testing.assert_close(stem.bias, bias)
        self.assertFalse(stem.weight.requires_grad)
        self.assertFalse(stem.bias.requires_grad)

    def test_padding_crop_and_gradient_path_through_all_conditions(self):
        model = self.build()
        conditions = [torch.randn(2, 3, 19, 35, requires_grad=True) for _ in range(3)]
        captured = []
        handle = model.backbone.patch_embed.proj.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        output = model(*conditions)
        handle.remove()
        self.assertEqual(tuple(output.shape), (2, 3, 19, 35))
        self.assertEqual(tuple(captured[0].shape), (6, 3, 32, 48))
        torch.testing.assert_close(captured[0][..., :19, :35], torch.cat(conditions, dim=0))
        self.assertEqual(captured[0][..., 19:, :].count_nonzero(), 0)
        self.assertEqual(captured[0][..., :, 35:].count_nonzero(), 0)
        output.square().mean().backward()
        for condition in conditions:
            self.assertGreater(condition.grad.abs().sum().item(), 0)
        adapters = model.backbone.blocks[0].attn["qkv"].lora_B
        self.assertEqual(len(adapters), 3)
        trainable = [adapter.weight for adapter in adapters]
        trainable.extend([model.layer_projs[0][0].weight, model.neck.layers[0].linear1.weight,
                          model.head.readout[-1].weight])
        for parameter in trainable:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        base = model.backbone.blocks[0].attn["qkv"].linear.weight
        self.assertFalse(base.requires_grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(model.backbone.patch_embed.proj.weight.grad)

    def test_feature_fusion_pairs_each_sample_with_its_three_conditions(self):
        model = self.build().eval()
        captured = []
        handles = [projection.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        ) for projection in model.layer_projs]

        def tagged_features(x, **kwargs):
            # Samples carry distinct tags; preserve them through a fake backbone.
            tags = x[:, :1, :1, :1]
            grid = tags.expand(-1, 12, 2, 2)
            return [grid, grid + 10]

        samples = torch.arange(2.).view(2, 1, 1, 1).expand(2, 3, 32, 32)
        conditions = [samples + 2 * modality for modality in range(3)]
        with patch.object(model.backbone, "forward_intermediates", side_effect=tagged_features):
            model(*conditions)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(captured), 2)
        for layer, feature in enumerate(captured):
            self.assertEqual(tuple(feature.shape), (2, 36, 2, 2))
            for sample in range(2):
                for modality in range(3):
                    channels = feature[sample, modality * 12:(modality + 1) * 12]
                    expected = torch.full_like(channels, sample + 2 * modality + 10 * layer)
                    torch.testing.assert_close(channels, expected, atol=0, rtol=0)

    def test_frozen_base_mode_and_repeatable_evaluation(self):
        model = self.build(lora_dropout=0.5)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.backbone.blocks[0].attn["qkv"].dropout.training)
        self.assertTrue(model.head.training)
        model.eval()
        self.assertFalse(any(module.training for module in model.modules()))
        conditions = [torch.randn(1, 3, 16, 32) for _ in range(3)]
        with torch.no_grad():
            torch.testing.assert_close(model(*conditions), model(*conditions), atol=0, rtol=0)

    def test_neck_checkpointing_preserves_outputs_and_gradients(self):
        model = self.build()
        checkpointed = copy.deepcopy(model)
        checkpointed.neck.grad_checkpointing = True
        conditions = [torch.randn(1, 3, 16, 32) for _ in range(3)]
        output = model(*conditions)
        checkpointed_output = checkpointed(*conditions)
        torch.testing.assert_close(output, checkpointed_output)
        output.square().mean().backward()
        checkpointed_output.square().mean().backward()
        torch.testing.assert_close(model.neck.layers[0].linear1.weight.grad,
                                   checkpointed.neck.layers[0].linear1.weight.grad)
        for original, recomputed in zip(model.backbone.blocks[0].attn["qkv"].lora_B,
                                        checkpointed.backbone.blocks[0].attn["qkv"].lora_B):
            torch.testing.assert_close(original.weight.grad, recomputed.weight.grad)
        self.assertIsNone(model.backbone.patch_embed.proj.weight.grad)
        self.assertIsNone(checkpointed.backbone.patch_embed.proj.weight.grad)

    def test_lora_initially_preserves_linear_response(self):
        linear = nn.Linear(12, 24)
        adapted = LoRALinear(linear, rank=2, alpha=4.0, dropout=0.0)
        x = torch.randn(1, 5, 12)
        torch.testing.assert_close(adapted(x), linear(x), atol=0, rtol=0)

    def test_three_lora_adapters_route_contiguous_modality_batches(self):
        linear = nn.Linear(2, 1, bias=False)
        adapted = LoRALinear(linear, rank=1, alpha=1., dropout=0., num_adapters=3)
        with torch.no_grad():
            linear.weight.zero_()
            for index, (left, right) in enumerate(zip(adapted.lora_A, adapted.lora_B)):
                left.weight.copy_(torch.tensor([[1., 0.]]))
                right.weight.fill_(index + 1.)
        x = torch.zeros(6, 1, 2)
        x[:, 0, 0] = torch.arange(1., 7.)
        expected = torch.tensor([1., 2., 6., 8., 15., 18.]).view(6, 1, 1)
        torch.testing.assert_close(adapted(x), expected, atol=0, rtol=0)

    def test_progressive_decoder_initializes_subpixel_phases_equally(self):
        model = self.build()
        self.assertEqual(len(model.head.upsample_stages), 4)
        for stage in model.head.upsample_stages:
            projection, _, shuffle, _ = stage
            self.assertEqual(shuffle.upscale_factor, 2)
            for phase in range(1, 4):
                torch.testing.assert_close(projection.weight[phase::4], projection.weight[::4], atol=0, rtol=0)
            torch.testing.assert_close(projection.bias, torch.zeros_like(projection.bias), atol=0, rtol=0)

    def test_checkpoint_keys_identify_three_adapters_and_progressive_decoder(self):
        model = self.build()
        state = model.state_dict()
        for index in range(3):
            self.assertIn(f"backbone.blocks.0.attn.qkv.lora_A.{index}.weight", state)
            self.assertIn(f"backbone.blocks.0.attn.qkv.lora_B.{index}.weight", state)
        self.assertEqual(tuple(state["backbone.patch_embed.proj.weight"].shape), (12, 3, 16, 16))
        self.assertEqual(tuple(state["layer_projs.0.0.weight"].shape), (4, 36, 1, 1))
        self.assertIn("head.upsample_stages.0.0.weight", state)
        self.assertNotIn("head.patch_proj.weight", state)
        self.assertNotIn("backbone.blocks.0.attn.qkv.lora_B.weight", state)

    def test_original_patch_rearrange_decoder_remains_selectable(self):
        model = self.build(decoder="patchrearrange")
        conditions = [torch.randn(1, 3, 16, 32) for _ in range(3)]
        self.assertEqual(tuple(model(*conditions).shape), (1, 3, 16, 32))
        self.assertIn("head.patch_proj.weight", model.state_dict())
        self.assertNotIn("head.upsample_stages.0.0.weight", model.state_dict())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
