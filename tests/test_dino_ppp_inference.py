"""CPU-only inference checks with tiny substitute predictors; no model downloads."""
import builtins
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling = load_file('dino_inference_sampling', 'radargen/ppp/sampling.py')
checkpoint = load_file('dino_inference_checkpoint', 'radargen/training/ppp_checkpoint.py')
with patch.dict(sys.modules, {'radargen.ppp.sampling': sampling}):
    pipeline_module = load_file('dino_inference_pipeline', 'radargen/inference/ppp_pipeline.py')
RadarGenPPPPipeline = pipeline_module.RadarGenPPPPipeline


class TinyDirectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = nn.Parameter(torch.tensor([1., 2., 3.]))

    def forward(self, color, segmentation, velocity):
        return torch.stack((color[:, 0], segmentation[:, 0], velocity[:, 0]), dim=1) * self.scales[None, :, None, None]


class DinoPPPInferenceTests(unittest.TestCase):
    def setUp(self):
        self.maps = [torch.full((3, 4, 4), value) for value in (1., 2., 3.)]

    def pipeline(self, model=None):
        return RadarGenPPPPipeline(model or TinyDirectModel(), resolution=4,
                                   coordinate_range=2., device='cpu', mixed_precision='no')

    def test_direct_predictions_keep_raw_fields_and_condition_order(self):
        model = TinyDirectModel().train()
        pipeline = self.pipeline(model)
        self.assertIsNone(pipeline.condition_encoder)
        self.assertIsNone(pipeline.text_provider)
        fields = pipeline.predict_fields(self.maps[0], self.maps[1][None], self.maps[2])
        self.assertFalse(model.training)
        self.assertEqual(tuple(fields), ('log_intensity', 'rcs_raw', 'doppler_raw'))
        repeated = pipeline.predict_fields(*self.maps)
        for name, expected in zip(fields, (1., 4., 9.)):
            self.assertEqual(fields[name].shape, (4, 4))
            self.assertEqual(fields[name].dtype, torch.float32)
            self.assertFalse(fields[name].requires_grad)
            torch.testing.assert_close(fields[name], torch.full((4, 4), expected))
            torch.testing.assert_close(fields[name], repeated[name], rtol=0, atol=0)

    def test_direct_input_and_output_contracts(self):
        pipeline = self.pipeline()
        for bad in (torch.zeros(2, 4, 4), torch.zeros(2, 3, 4, 4),
                    torch.full((3, 4, 4), float('nan'))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                pipeline.predict_fields(bad, *self.maps[1:])

        class InvalidModel(TinyDirectModel):
            def forward(self, *maps):
                return super().forward(*maps).unsqueeze(2)

        with self.assertRaisesRegex(ValueError, 'direct PPP output shape'):
            self.pipeline(InvalidModel()).predict_fields(*self.maps)
        model = TinyDirectModel()
        with torch.no_grad():
            model.scales[0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Nonfinite predicted fields'):
            self.pipeline(model).predict_fields(*self.maps)

    def test_legacy_encoder_and_text_interface_is_preserved(self):
        class Encoder(nn.Module):
            def forward(self, maps):
                return [value * 2 for value in maps]

        class LegacyModel(TinyDirectModel):
            def forward(self, latents, text, *, mask, data_info):
                self.received = text, mask, data_info
                return super().forward(*latents).unsqueeze(2)

        text = Mock(return_value=(torch.ones(1), torch.ones(1)))
        model, encoder = LegacyModel(), Encoder()
        pipeline = RadarGenPPPPipeline(model, encoder, text, resolution=4,
                                       device='cpu', mixed_precision='no')
        fields = pipeline.predict_fields(*self.maps)
        text.assert_called_once_with(1, torch.device('cpu'))
        self.assertFalse(encoder.training)
        torch.testing.assert_close(fields['rcs_raw'], torch.full((4, 4), 8.))
        self.assertEqual(tuple(model.received[2]['img_hw'].shape), (1, 2))

    def test_full_checkpoint_inference_needs_no_original_or_text_assets(self):
        source = TinyDirectModel()
        with torch.no_grad():
            source.scales.copy_(torch.tensor([2., 3., 4.]))
        state = dict(state_dict=source.state_dict(), ppp_training=dict(version=1),
                     epoch=2, rank_rng_states=[{}])
        initialization = ModuleType('radargen.training.ppp_initialization')
        initialization.initialize_dino_ppp_model = Mock(
            side_effect=lambda config, **kwargs: SimpleNamespace(
                model=TinyDirectModel(), condition_encoder=None))
        initialization.initialize_ppp_model = Mock(side_effect=AssertionError('Latent initializer called'))
        imported = builtins.__import__

        def reject_text_import(name, *args, **kwargs):
            if name == 'scripts.train_ppp':
                raise AssertionError('DINO inference imported legacy text setup')
            return imported(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trained.pth'
            torch.save(state, path)
            config = SimpleNamespace(
                model=SimpleNamespace(model='DinoPPP', load_from=str(path), resume_from='unused',
                                      image_size=4, mixed_precision='no'),
                data=SimpleNamespace(extra={'coordinate_range': 2.}),
                train=SimpleNamespace(grad_checkpointing=True),
                resume_from='unused', load_from='unused')
            with patch.dict(sys.modules, {
                'radargen.training.ppp_initialization': initialization,
                'radargen.training.ppp_checkpoint': checkpoint,
            }), patch('builtins.__import__', side_effect=reject_text_import):
                pipeline = RadarGenPPPPipeline.from_config(config, device='cpu')
                fresh = initialization.initialize_dino_ppp_model.call_args.args[0]
                self.assertIsNot(fresh, config)
                self.assertIsNone(fresh.model.load_from)
                self.assertIsNone(fresh.resume_from)
                self.assertFalse(fresh.train.grad_checkpointing)
                initialization.initialize_dino_ppp_model.assert_called_once_with(
                    fresh, device='cpu', load_pretrained=False)
                initialization.initialize_ppp_model.assert_not_called()
                self.assertEqual(config.model.load_from, str(path))
                torch.testing.assert_close(pipeline.model.scales, source.scales)

                # Full checkpoints must contain every predictor tensor with its
                # exact shape, even when no optimizer is being restored.
                for broken, error in (({}, ValueError), ({'scales': torch.ones(2)}, RuntimeError)):
                    with self.subTest(state=broken):
                        torch.save(dict(state, state_dict=broken), path)
                        with self.assertRaises(error):
                            RadarGenPPPPipeline.from_config(config, device='cpu')

    def test_dataset_and_point_sampling_keep_existing_interfaces(self):
        pipeline = self.pipeline()
        info = {'sample_id': 'example'}
        dataset = [dict(zip(pipeline_module.CONDITION_KEYS, self.maps), data_info=info)]
        fields, identity = pipeline.predict_dataset_sample(dataset, 0)
        self.assertEqual(identity, info)
        counts = torch.zeros(4, 4, dtype=torch.float64)
        counts[0, 0] = 1
        with patch.object(sampling.torch, 'poisson', return_value=counts):
            points = pipeline.sample_fields(fields)
        # Cubic mark decoding belongs to point sampling; predict_fields stays raw.
        np.testing.assert_allclose(points, [[-1.5, 1.5, 64., 729.]])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
