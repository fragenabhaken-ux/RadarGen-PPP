"""Full-width path, useful parameter budget, gradients and checkpoint transfer."""
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from diffusion.model.nets.radargen_ppp import RadarGenPPP
from diffusion.model_modification import inflate_sana_input_channels_for_ppp
from radargen.models.wide_ppp_decoder import WidePPPDecoder
from radargen.training.ppp_checkpoint import load_ppp_backbone
from radargen.training.radargen_ppp_training_model import RadarGenPPPTrainingModel


def tiny_dit():
    dit = RadarGenPPP(input_size=2, hidden_size=32, depth=1, num_heads=4,
                     caption_channels=8, model_max_length=4, attn_type='linear',
                     ffn_type='mlp', use_pe=False, class_dropout_prob=0.0)
    import logging
    inflate_sana_input_channels_for_ppp(dit, logging.getLogger(__name__))
    return dit


def test_parameter_budget_and_full_resolution():
    with torch.device('meta'):
        decoder = WidePPPDecoder()
        assert 0.99 * decoder.parameter_budget <= decoder.parameter_count <= decoder.parameter_budget
        assert decoder(torch.empty(3, 1152, 16, 16)).shape == (3, 1, 512, 512)
    with pytest.raises(ValueError, match='budget'):
        WidePPPDecoder(parameter_budget=100)


def test_full_width_training_and_checkpointing():
    torch.set_num_threads(2)
    dit = tiny_dit()
    dit.final_layer = nn.Identity()
    dit.return_features = True
    decoder = WidePPPDecoder(in_channels=32, widths=(16, 12, 8, 4, 4),
                             parameter_budget=60000, refinement_blocks=1)
    model = RadarGenPPPTrainingModel(dit, decoder)
    conditions = [torch.randn(1, 32, 2, 2) for _ in range(3)]
    text = torch.randn(3, 1, 4, 8)
    mask = torch.ones(3, 1, 1, 4)
    output = model(conditions, text, mask=mask)
    assert output.shape == (1, 3, 1, 64, 64)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert dit.x_embedder.proj.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    decoder.gradient_checkpointing = True
    checked = model(conditions, text, mask=mask)
    torch.testing.assert_close(checked, output)
    checked.square().mean().backward()
    assert all(p.grad is not None for p in decoder.parameters())
    # Strict variant checkpoint round trip, as used by resume/inference.
    clone = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(clone, strict=True)


def test_backbone_transfer_excludes_old_head_and_decoder():
    old = RadarGenPPPTrainingModel(tiny_dit(), nn.Conv2d(32, 1, 1))
    new = RadarGenPPPTrainingModel(tiny_dit(), nn.Conv2d(32, 1, 1))
    new.dit.final_layer = nn.Identity()
    new.dit.return_features = True
    before = new.ppp_decoder.weight.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'legacy.pth'
        state = {'ppp_training': {'version': 1}, 'state_dict': old.state_dict()}
        torch.save(state, path)
        load_ppp_backbone(path, new)
        torch.testing.assert_close(new.dit.blocks[0].attn.qkv.weight, old.dit.blocks[0].attn.qkv.weight)
        torch.testing.assert_close(new.ppp_decoder.weight, before)
        del state['state_dict']['dit.blocks.0.attn.qkv.weight']
        torch.save(state, path)
        with pytest.raises(ValueError, match='backbone mismatch'):
            load_ppp_backbone(path, new)


@pytest.mark.parametrize('variant,expected_count', [
    ('nearest_efficientvit', 159156673), ('pixelshuffle', 159119201),
])
def test_experiment_budget_and_full_resolution(variant, expected_count):
    from radargen.models.wide_ppp_decoder import NearestEfficientPPPDecoder, PixelShufflePPPDecoder
    cls = NearestEfficientPPPDecoder if variant == 'nearest_efficientvit' else PixelShufflePPPDecoder
    with torch.device('meta'):
        decoder = cls()
        assert decoder.parameter_count == expected_count
        assert 0.99 * decoder.parameter_budget <= decoder.parameter_count <= decoder.parameter_budget
        assert decoder(torch.empty(6, 1152, 16, 16)).shape == (6, 1, 512, 512)
        if variant == 'pixelshuffle':
            assert decoder.stage_depths == (33, 3, 3)
            assert sum(isinstance(m, nn.PixelShuffle) for m in decoder.modules()) == 3
            upsample = [m for m in decoder.modules() if isinstance(m, nn.Upsample)]
            assert len(upsample) == 2
            assert all(m.mode == 'bilinear' and not m.align_corners for m in upsample)
        else:
            from diffusion.model.dc_ae.efficientvit.models.nn.ops import EfficientViTBlock
            assert sum(isinstance(m, EfficientViTBlock) for m in decoder.modules()) == 8


def experiment_decoder(variant):
    from radargen.models.wide_ppp_decoder import NearestEfficientPPPDecoder, PixelShufflePPPDecoder
    if variant == 'nearest_efficientvit':
        return NearestEfficientPPPDecoder(in_channels=32, widths=(32, 32, 16, 8, 4),
                                         refinement_blocks=1, parameter_budget=200000)
    return PixelShufflePPPDecoder(in_channels=32, stage_channels=32,
                                  parameter_budget=200000, stage_depths=(2, 2, 2))


@pytest.mark.parametrize('variant', ['nearest_efficientvit', 'pixelshuffle'])
def test_experiment_dit_gradients_precision_checkpointing_and_reload(variant):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    dit = tiny_dit()
    dit.final_layer = nn.Identity()
    dit.return_features = True
    decoder = experiment_decoder(variant)
    model = RadarGenPPPTrainingModel(dit, decoder)
    conditions = [torch.randn(2, 32, 2, 3) for _ in range(3)]
    text, mask = torch.randn(6, 1, 4, 8), torch.ones(6, 1, 1, 4)
    output = model(conditions, text, mask=mask)
    assert output.shape == (2, 3, 1, 64, 96)
    output.square().mean().backward()
    assert dit.x_embedder.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
    gradients = {k: p.grad.clone() for k, p in decoder.named_parameters()}
    model.zero_grad(set_to_none=True)
    decoder.gradient_checkpointing = True
    checked = model(conditions, text, mask=mask)
    torch.testing.assert_close(checked, output)
    checked.square().mean().backward()
    for key, parameter in decoder.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[key])
    # BF16 normalization/attention must remain finite, including checkpointed backward.
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mixed = model(conditions, text, mask=mask)
        loss = mixed.float().square().mean()
    assert mixed.dtype == torch.bfloat16 and torch.isfinite(mixed).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'variant.pth'
        torch.save(model.state_dict(), path)
        clone_dit = tiny_dit()
        clone_dit.final_layer = nn.Identity()
        clone_dit.return_features = True
        clone = RadarGenPPPTrainingModel(clone_dit, experiment_decoder(variant))
        clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
        torch.testing.assert_close(clone(conditions, text, mask=mask), output)


def test_efficientvit_norm_initialization_and_invalid_budgets():
    from radargen.models.wide_ppp_decoder import NearestEfficientPPPDecoder, PixelShufflePPPDecoder
    from diffusion.model.dc_ae.efficientvit.models.nn.norm import RMSNorm2d
    decoder = experiment_decoder('nearest_efficientvit')
    norms = [m for m in decoder.modules() if isinstance(m, RMSNorm2d)]
    assert norms
    for norm in norms:
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
        torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias))
    with torch.device('meta'):
        for cls in (NearestEfficientPPPDecoder, PixelShufflePPPDecoder):
            with pytest.raises(ValueError, match='budget'):
                cls(parameter_budget=100)
        with pytest.raises(ValueError, match='divisible'):
            PixelShufflePPPDecoder(stage_channels=20)
