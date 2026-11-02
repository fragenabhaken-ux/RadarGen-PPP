"""Reuse the actual DINO head without mixing samples or losing DiT stream gradients."""
import torch
from torch import nn
import pytest
from radargen.models.dino_ppp import AttentionPixelShuffleHead, WindowResidualAttentionBlock
from radargen.models.wide_ppp_decoder import JointDinoAttentionPPPDecoder
from radargen.training.radargen_ppp_training_model import RadarGenPPPTrainingModel
from test_wide_ppp_decoder import tiny_dit


def small_head():
    return JointDinoAttentionPPPDecoder(in_channels=32, channels=(8, 4, 4),
                                       heads=2, window_size=4)


def test_joint_head_full_width_geometry_and_block_identity():
    with torch.device('meta'):
        decoder = JointDinoAttentionPPPDecoder()
        assert isinstance(decoder.head, AttentionPixelShuffleHead)
        assert decoder.head.stages[0][0].in_channels == 3456
        assert decoder.head.stages[0][0].out_channels == 2048
        assert decoder.head.stages[0][0].kernel_size == (3, 3)
        assert [s[1].upscale_factor for s in decoder.head.stages] == [4, 4, 2]
        assert [[b.shift_size for b in s if isinstance(b, WindowResidualAttentionBlock)]
                for s in decoder.head.stages] == [[0, 4]] * 3
        assert decoder(torch.empty(6, 1152, 16, 16)).shape == (2, 3, 512, 512)
        assert decoder.parameter_count < 159220865


def test_sample_major_stream_concatenation():
    torch.set_num_threads(2)
    decoder = small_head()
    features = torch.arange(6.).reshape(6, 1, 1, 1).expand(6, 32, 2, 2)
    captured = []
    hook = decoder.head.stages[0][0].register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    decoder(features)
    hook.remove()
    assert captured[0].shape == (2, 96, 2, 2)
    for b in range(2):
        for modality in range(3):
            torch.testing.assert_close(captured[0][b, modality*32:(modality+1)*32],
                                       features[3*b+modality])
    with pytest.raises(ValueError, match='3B'):
        decoder(torch.empty(4, 32, 2, 2))


def test_joint_head_dit_backward_checkpointing_and_reload():
    torch.set_num_threads(2)
    dit = tiny_dit()
    dit.final_layer = nn.Identity()
    dit.return_features = True
    model = RadarGenPPPTrainingModel(dit, small_head())
    conditions = [torch.randn(2, 32, 2, 2) for _ in range(3)]
    text, mask = torch.randn(6, 1, 4, 8), torch.ones(6, 1, 1, 4)
    output = model(conditions, text, mask=mask)
    assert output.shape == (2, 3, 1, 64, 64)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert dit.x_embedder.proj.weight.grad.abs().sum() > 0
    grads = {k:p.grad.clone() for k,p in model.ppp_decoder.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.ppp_decoder.gradient_checkpointing = True
    actual = model(conditions, text, mask=mask)
    torch.testing.assert_close(actual, output)
    actual.square().mean().backward()
    for name, parameter in model.ppp_decoder.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name])
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mixed = model(conditions, text, mask=mask)
        loss = mixed.float().square().mean()
    loss.backward()
    assert torch.isfinite(mixed).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.ppp_decoder.parameters())
    model.load_state_dict(model.state_dict(), strict=True)


def test_three_fourfold_stages_keep_final_residual_attention_blocks():
    with torch.device('meta'):
        decoder = JointDinoAttentionPPPDecoder(output_scale=2)
        assert [s[1].upscale_factor for s in decoder.head.stages] == [4, 4, 4]
        assert decoder.head.stages[-1][0].out_channels == 512
        last = decoder.head.stages[-1]
        assert len(last) == 4  # projection, shuffle, both full refinement blocks
        assert all(isinstance(b, WindowResidualAttentionBlock) for b in last[2:])
        assert decoder(torch.empty(6, 1152, 16, 16)).shape == (2, 3, 1024, 1024)
        assert decoder.parameter_count == 66304163


def test_1024_mode_reaches_inference_and_continuous_loss_at_correct_scale():
    from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
    from radargen.training.ppp_decoder import FrozenConditionEncoder
    from radargen.losses import ContinuousPPPLoss
    torch.set_num_threads(2)
    dit = tiny_dit()
    dit.final_layer = nn.Identity()
    dit.return_features = True
    decoder = JointDinoAttentionPPPDecoder(in_channels=32, channels=(8, 4, 4),
                heads=2, window_size=4, output_scale=2, gradient_checkpointing=True)
    model = RadarGenPPPTrainingModel(dit, decoder)
    assert model.output_scale == 2
    text = lambda b, device: (torch.ones(3*b,1,4,8,device=device), torch.ones(3*b,1,1,4,device=device))
    encoder = FrozenConditionEncoder(nn.Conv2d(3,32,32,stride=32),1.0)
    pipeline = RadarGenPPPPipeline(model, encoder, text, resolution=64, device='cpu',mixed_precision='no')
    fields = pipeline.predict_fields(*[torch.randn(3,64,64) for _ in range(3)])
    assert all(v.shape == (128,128) for v in fields.values())
    model.train()
    output = model([torch.randn(1,32,2,2) for _ in range(3)],*text(1,'cpu')[:1], mask=text(1,'cpu')[1])
    assert output.shape == (1,3,1,128,128)
    points = torch.tensor([[0.,0.,2.,3.],[0.,0.,4.,-1.]])
    loss = ContinuousPPPLoss((128,128))(output[:,:,0], points, torch.zeros(2,dtype=torch.long),point_limit=50.)
    sum(loss.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
