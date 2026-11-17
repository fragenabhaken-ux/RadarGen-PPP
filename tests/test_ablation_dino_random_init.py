"""Random ablation must never accidentally load pretrained or PPP weights."""
from pathlib import Path
from unittest.mock import patch
import unittest
import pyrallis
from torch import nn
from diffusion.utils.config import SanaConfig
from scripts.train_ppp import validate_ppp_config
from radargen.training.ppp_initialization import initialize_dino_ppp_model

ROOT = Path(__file__).resolve().parents[1]


class RandomDinoInitializationTests(unittest.TestCase):
    def config(self):
        with (ROOT/'configs/ablation_dino_random_init.yaml').open() as handle:
            return pyrallis.load(SanaConfig, handle)

    def test_random_mode_constructs_without_pretrained_file(self):
        config = self.config()
        validate_ppp_config(config)
        with patch('radargen.models.dino_ppp.DinoPPP', return_value=nn.Linear(2, 2)) as build:
            initialize_dino_ppp_model(config, device='cpu')
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs['checkpoint_path'], '')
        self.assertIs(kwargs['pretrained'], False)
        self.assertIs(kwargs['frozen_base'], False)
        self.assertNotIn('backbone_initialization', kwargs)

    def test_random_mode_rejects_weights_and_frozen_backbone(self):
        for changes in ({'checkpoint_path': '/tmp/not-a-checkpoint'}, {'pretrained': True}, {'frozen_base': True}):
            config = self.config()
            config.model.extra.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_ppp_config(config)
                with self.assertRaises(ValueError):
                    initialize_dino_ppp_model(config)

    def test_missing_pretrained_file_remains_an_error(self):
        config = self.config()
        config.model.extra['backbone_initialization'] = 'pretrained'
        config.model.extra['checkpoint_path'] = '/tmp/missing-dino-ablation-weights.safetensors'
        with self.assertRaises(FileNotFoundError):
            initialize_dino_ppp_model(config)

    def test_resume_can_reconstruct_random_architecture_without_rgb_weights(self):
        config = self.config()
        with patch('radargen.models.dino_ppp.DinoPPP', return_value=nn.Linear(2, 2)) as build:
            initialize_dino_ppp_model(config, device='cpu', load_pretrained=False)
        self.assertFalse(build.call_args.kwargs['pretrained'])


class BatchEstimateTests(unittest.TestCase):
    def test_ddp_estimate_keeps_memory_per_gpu_and_caps_global_batch(self):
        from scripts.probe_ablation_batch import recommend
        gib = 1024**3
        rows = [dict(batch_per_gpu=b, peak_allocated_bytes=(5+2*b)*gib,
                     peak_reserved_bytes=(6+2*b)*gib) for b in [1,4,8]]
        result = recommend(rows, gib, 95*gib, gib)
        self.assertEqual(result['effective_batch'],4*result['selected_batch_per_gpu'])
        self.assertLessEqual(result['predicted_per_gpu_bytes'],90_000_000_000)
        self.assertGreater(result['ddp_allowance_bytes'],gib)


class LRCalibrationTests(unittest.TestCase):
    def test_cadence_sets_thirty_minute_update_budget(self):
        import datetime as dt
        from scripts.run_ablation_lr_range import sweep_updates
        start = dt.datetime(2026, 11, 11)
        times = [start + dt.timedelta(seconds=4*i) for i in range(5)]
        updates, cadence = sweep_updates(times, 1800)
        self.assertEqual(updates, 450)
        self.assertEqual(cadence, 4)
        with self.assertRaises(ValueError):
            sweep_updates(times[:2], 1800)


if __name__ == '__main__':
    unittest.main()
