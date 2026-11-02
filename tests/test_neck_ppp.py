"""CNN-Neck PPP integration, gradient flow and checkpoint contracts."""
import copy
import unittest
from types import SimpleNamespace
import torch
from radargen.models.neck_ppp import NeckPPP
from radargen.losses.continuous import ContinuousPPPLoss
from radargen.training.ppp_initialization import initialize_dino_ppp_model
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline


class NeckPPPTests(unittest.TestCase):
    def test_loss_gradients_checkpoint_and_inference(self):
        options = dict(embed_dim=24, stem_channels=(8, 8, 16), neck_layers=1,
                       neck_heads=3, stage_channels=(8, 8, 4, 4), local_refine_blocks=1)
        config = SimpleNamespace(model=SimpleNamespace(model="NeckPPP", extra=options),
                                 train=SimpleNamespace(grad_checkpointing=True))
        model = initialize_dino_ppp_model(config).model
        inputs = [torch.randn(2, 3, 19, 35, requires_grad=True) for _ in range(3)]
        output = model(*inputs)
        self.assertEqual(output.shape, (2, 3, 19, 35))
        losses = ContinuousPPPLoss(out_img_size=(19, 35), mark_link="identity")(
            output, torch.tensor([[0., 0., 3., -2.]]), torch.tensor([0]), point_limit=50.)
        sum(losses.values()).backward()
        for x in inputs:
            self.assertGreater(x.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        restored = NeckPPP(**options)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        model.eval(); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs), restored(*inputs))
        pipeline = RadarGenPPPPipeline(restored, resolution=32, device="cpu", mixed_precision="fp32", mark_link="identity")
        maps = [torch.randn(3, 32, 32) for _ in range(3)]
        first = pipeline.predict_fields(*maps)
        second = pipeline.predict_fields(*maps)
        for key in first:
            self.assertEqual(first[key].shape, (32, 32))
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_skip_path_carries_spatial_detail_without_neck_features(self):
        model = NeckPPP(embed_dim=24, stem_channels=(8, 8, 16), neck_layers=1,
                        neck_heads=3, stage_channels=(8, 8, 4, 4)).eval()
        # Isolate the bypasses: suppress all information from the coarse neck.
        hook = model.neck.register_forward_hook(lambda module, args, output: output * 0)
        inputs = [torch.randn(1, 3, 32, 48, requires_grad=True) for _ in range(3)]
        output = model(*inputs)
        output.square().mean().backward()
        hook.remove()
        for x in inputs:
            self.assertGreater(x.grad.abs().sum().item(), 0)
        for projection, fusion in zip(model.skip_projs, model.skip_fusions):
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
            self.assertGreater(fusion[0].weight.grad.abs().sum().item(), 0)

    def test_rejects_mismatched_conditions(self):
        model = NeckPPP(neck_layers=0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 32), torch.zeros(1, 3, 16, 16))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
