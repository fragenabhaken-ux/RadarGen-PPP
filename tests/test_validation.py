"""CPU integration checks for durable per-sample validation outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from evaluation.validation import evaluate_samples, json_values, sample_metrics
from radargen.core.normalization import NormalizationConfig


class ToyDataset:
    resolution = 8
    point_limit = 50.
    normalization = vars(NormalizationConfig()).copy()

    def __init__(self, directory):
        self.frame_index = []
        for i in range(2):
            path = Path(directory) / f'sample_{i:06d}.h5'
            with h5py.File(path, 'w') as h:
                points = np.array([[0., 0.], [2., 2.]], dtype=np.float32)
                h['ppp_targets/xy'] = points
                h['ppp_targets/rcs'] = np.array([1., 2.], dtype=np.float32)
                h['ppp_targets/doppler'] = np.array([-1., 2.], dtype=np.float32)
                for name in ('appearance', 'semantics', 'velocity'):
                    h[f'conditioning/{name}'] = np.zeros((8, 8, 3), dtype=np.uint8)
                for name in ('point_density', 'rcs', 'doppler'):
                    h[f'radargen_targets/{name}'] = np.ones((8, 8), dtype=np.float32)
            stat = path.stat()
            self.frame_index.append((path, dict(scene_token='scene', frame_index=i), stat.st_size, stat.st_mtime_ns))

    def __getitem__(self, index):
        mask = torch.zeros(1, 8, 8)
        mask[0, 4, 4] = 1
        return dict(point_mask=mask, rcs_target=mask, doppler_target=-mask)


class ToyPredictor:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def predict(self, sample, conditions):
        self.calls += 1
        points = np.array([[0., 0., 1., -1.], [2., 2., 2., 2.]])
        if self.calls == 2:
            points = np.empty((0, 4))
        if self.model == 'ppp':
            fields = dict(cell_mass_grid=np.full((8, 8), 2 / 64),
                          rcs=np.ones((8, 8)), doppler=-np.ones((8, 8)))
        else:
            fields = (np.ones((8, 8)), np.ones((8, 8)), -np.ones((8, 8)))
        return points, fields


class ValidationChecks(unittest.TestCase):
    def test_both_models_all_samples_including_empty_prediction(self):
        for model in ('ppp', 'radargen'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                dataset = ToyDataset(directory)
                output = Path(directory) / 'results'
                output.mkdir()
                predictor = ToyPredictor(model)
                config = dict(model=model, seed=42, checkpoint_path='toy.pth', plot_dpi=30)
                provenance = dict(config, sample_count=2)
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'):
                    rows = evaluate_samples(dataset, predictor, config, torch.device('cpu'), output, provenance)
                self.assertEqual(predictor.calls, 2)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]['status'], 'empty_cloud')
                self.assertIsNone(rows[1]['metrics']['chamfer_distance_xy'])
                self.assertEqual(rows[0]['metrics']['chamfer_distance_xy'], 0.)
                self.assertEqual(rows[1]['metrics']['count_abs_error'], 2)
                for row in rows:
                    self.assertTrue((output / row['plot']).is_file())
                self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 2)
                self.assertEqual(len((output / 'metrics.txt').read_text().splitlines()), 3)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertTrue(summary['complete'])
                self.assertEqual(summary['empty_generated_samples'], 1)
                self.assertEqual(summary['metrics']['chamfer_distance_xy']['valid_samples'], 1)
                if model == 'ppp':
                    self.assertAlmostEqual(rows[1]['metrics']['ppp_expected_total_count'], 2.)
                # Resume a completed run without regenerating either sample.
                restarted = ToyPredictor(model)
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'):
                    restored = evaluate_samples(dataset, restarted, config, torch.device('cpu'), output, provenance, resume=True)
                self.assertEqual(restarted.calls, 0)
                self.assertEqual(restored, rows)
                # Simulate a partial run with only its first committed record.
                (output / 'metrics.jsonl').write_text(json.dumps(rows[0]) + '\n')
                (output / rows[1]['plot']).unlink()
                restarted.calls = 1
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'):
                    restored = evaluate_samples(dataset, restarted, config, torch.device('cpu'), output, provenance, resume=True)
                self.assertEqual(restarted.calls, 2)
                self.assertEqual(len(restored), 2)
                self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 2)

    def test_failure_leaves_explicit_incomplete_reports(self):
        with TemporaryDirectory() as directory:
            dataset = ToyDataset(directory)
            output = Path(directory) / 'results'
            output.mkdir()
            predictor = ToyPredictor('radargen')
            config = dict(model='radargen', seed=42, checkpoint_path='toy.pth')
            with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata', side_effect=ValueError('bad metadata')):
                with self.assertRaisesRegex(ValueError, 'bad metadata'):
                    evaluate_samples(dataset, predictor, config, torch.device('cpu'), output, dict(config, sample_count=2))
            self.assertFalse(json.loads((output / 'summary.json').read_text())['complete'])
            self.assertEqual(predictor.calls, 0)

    def test_empty_ground_truth_and_json_policy(self):
        both = sample_metrics(np.empty((0, 4)), np.empty((0, 4)), None, None, 'cpu')
        self.assertEqual(both['distance_attr_f1'], 1.)
        self.assertIsNone(both['pcl_mmd_xy'])
        self.assertEqual(json_values(dict(x=np.float32('nan'), y=np.int64(3))), dict(x=None, y=3))


if __name__ == '__main__':
    unittest.main()
