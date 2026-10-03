"""Run directly: python tests/test_ppp_inference.py (CPU only)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling = load_file('sampling_under_test', 'radargen/ppp/sampling.py')
checkpoint = load_file('checkpoint_under_test', 'radargen/training/ppp_checkpoint.py')


class SamplingTests(unittest.TestCase):
    def test_cell_centers_marks_and_multiplicity(self):
        # Controlled counts isolate inverse geometry from sampling randomness.
        counts = torch.tensor([[2., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        raw_rcs = torch.full((2, 3), -2.)
        raw_doppler = torch.full((2, 3), 3.)
        with patch.object(sampling.torch, 'poisson', return_value=counts):
            points = sampling.sample_ppp_grid(torch.zeros(2, 3), raw_rcs, raw_doppler, 3.,
                                              torch.Generator().manual_seed(1))
        np.testing.assert_allclose(points, [[-2., 1.5, -8., 27.], [-2., 1.5, -8., 27.], [2., -1.5, -8., 27.]])
        # Rasterize the centers again, including positive and negative y.
        cols = np.floor((points[:, 0] + 3.) / 6. * 3).astype(int)
        rows = 1 - np.floor((points[:, 1] + 3.) / 6. * 2).astype(int)
        np.testing.assert_array_equal(cols, [0, 0, 2])
        np.testing.assert_array_equal(rows, [0, 0, 1])

    def test_mass_and_reproducibility(self):
        log = torch.full((1, 8, 8), np.log(100.), dtype=torch.float64)
        self.assertAlmostEqual(sampling.ppp_cell_masses(log).sum().item(), 100.)
        def draw():
            return sampling.sample_ppp_grid(log, torch.ones_like(log), torch.zeros_like(log), 50.,
                                            torch.Generator().manual_seed(17))
        np.testing.assert_array_equal(draw(), draw())
        # Check actual Poisson cell statistics, separately from geometry checks.
        masses = torch.full((100000,), 2., dtype=torch.float64)
        counts = torch.poisson(masses, generator=torch.Generator().manual_seed(11))
        self.assertLess(abs(counts.mean().item() - 2.), .04)
        self.assertLess(abs(counts.var().item() - 2.), .08)

    def test_empty_and_invalid(self):
        zero = torch.zeros(2, 2)
        points = sampling.sample_ppp_grid(torch.full((2, 2), -1000.), zero, zero, 1.,
                                          torch.Generator().manual_seed(1))
        self.assertEqual(points.shape, (0, 4))
        for bad in (torch.full((2, 2), float('nan')), torch.full((2, 2), 10000.)):
            with self.assertRaises(ValueError):
                sampling.ppp_cell_masses(bad)
        with self.assertRaises(ValueError):
            sampling.sample_ppp_grid(zero, zero, zero, 1.)

    def test_strict_model_only_checkpoint(self):
        model = nn.Module()
        model.dit = nn.Linear(2, 2)
        model.ppp_decoder = nn.Linear(2, 1)
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state = dict(state_dict=original, ppp_training=dict(version=1), epoch=0,
                     rank_rng_states=[{}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save(state, path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            checkpoint.load_ppp_checkpoint(path, model, resume_optimizer=False, resume_lr_scheduler=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            state['state_dict'] = {k: v for k, v in original.items() if not k.startswith('ppp_decoder.')}
            torch.save(state, path)
            with self.assertRaises(ValueError):
                checkpoint.load_ppp_checkpoint(path, model)


if __name__ == '__main__':
    unittest.main()
