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

from evaluation.validation import evaluate_samples, json_values, sample_metrics, plot_ppp_prediction, select_validation_indices, select_keyframe_indices, validation_boxes, foreground_metrics, write_reports, class_foreground_mmd
from radargen.core.normalization import NormalizationConfig


class ToyDataset:
    resolution = 8
    point_limit = 50.
    normalization = vars(NormalizationConfig()).copy()

    def __init__(self, directory):
        self.adapter = SimpleNamespace(get_bounding_boxes=lambda *a, **kw: [])
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
            self.frame_index.append((path, dict(scene_token='scene', frame_index=i, reference_is_key_frame=True, reference_sample_data_token=str(i)), stat.st_size, stat.st_mtime_ns))

    def __getitem__(self, index):
        return dict(detections=torch.tensor([[0., 0., 1., -1.], [2., 2., 2., 2.]]),
                    point_limit=self.point_limit)



class ToyPredictor:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.total_calls = 0

    def seed_sample(self, index):
        self.calls = index

    def predict(self, sample, conditions):
        self.total_calls += 1
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
    def test_scene_selection_is_reproducible_and_order_independent(self):
        inventory = [(Path(f'{scene}_{frame}.h5'),
                      dict(scene_token=f'scene_{scene}', frame_index=frame), 0, 0)
                     for scene in range(40) for frame in range(8)]
        chosen = select_validation_indices(inventory, 30)
        self.assertEqual(chosen, select_validation_indices(inventory, 30))
        self.assertEqual(len(set(chosen)), 30)
        self.assertEqual(len({inventory[i][1]['scene_token'] for i in chosen}), 30)
        reverse = inventory[::-1]
        self.assertEqual([inventory[i][1] for i in chosen],
                         [reverse[i][1] for i in select_validation_indices(reverse, 30)])
        self.assertEqual(select_validation_indices(inventory, len(inventory)), list(range(len(inventory))))
        small = inventory[:16]
        self.assertEqual(len(set(select_validation_indices(small, 12))), 12)

    def test_class_mmd_persisted_points_and_missing_classes(self):
        cloud = [[0., 0., 1., -1.], [1., 2., 3., 4.]]
        row = dict(foreground_points={side: {'vehicle.car': [cloud]} for side in ('gt', 'syn')})
        restored = json.loads(json.dumps(row))
        scores = class_foreground_mmd([restored])
        for feature in ('xy', 'rcs', 'doppler'):
            self.assertAlmostEqual(scores[f'invariant_mmd_{feature}_car'], 0.)
            self.assertIsNone(scores[f'invariant_mmd_{feature}_truck'])
            self.assertIsNone(scores[f'invariant_mmd_{feature}_trailer'])
        restored['foreground_points']['syn']['vehicle.car'][0][0][0] = 10.
        self.assertGreater(class_foreground_mmd([restored])['invariant_mmd_xy_car'], 0.)

    def test_keyframes_filtered_before_limit(self):
        inventory = [(Path(str(i)), dict(scene_token='scene', frame_index=i,
                      reference_is_key_frame=i % 3 == 0), 0, 0) for i in range(10)]
        self.assertEqual(select_keyframe_indices(inventory), [0, 3, 6, 9])
        selected = select_keyframe_indices(inventory, 2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(i % 3 == 0 for i in selected))
        with self.assertRaisesRegex(ValueError, 'No annotated'):
            select_keyframe_indices(inventory[1:3])

    def test_foreground_empty_boxes_and_object_weighting(self):
        box = SimpleNamespace(name='vehicle.car', center=np.zeros(3),
            rotation_matrix=np.eye(3), wlh=np.array([4., 4., 2.]),
            bottom_corners=lambda: np.array([[-2, -2, 2, 2], [-2, 2, 2, -2], [0, 0, 0, 0]]))
        adapter = SimpleNamespace(get_bounding_boxes=lambda *a, **kw:
            [box, SimpleNamespace(name='human.pedestrian')])
        boxes = validation_boxes(SimpleNamespace(adapter=adapter),
            dict(reference_is_key_frame=True, reference_sample_data_token='token'))
        self.assertEqual(boxes, [box])
        gt = np.array([[0., 0., 1., 1.]])
        hit = foreground_metrics(gt, gt, boxes, NormalizationConfig(), 'cpu')
        miss = foreground_metrics(np.empty((0, 4)), gt, boxes, NormalizationConfig(), 'cpu')
        empty = foreground_metrics(np.empty((0, 4)), np.empty((0, 4)), boxes, NormalizationConfig(), 'cpu')
        self.assertEqual(hit['per_box_chamfer_distance_xy'], [0.])
        self.assertEqual(miss['per_box_chamfer_distance_xy'], [])
        self.assertEqual(miss['per_box_num_points_similarity'], [0.])
        self.assertEqual(empty['per_box_num_points_similarity'], [1.])
        with TemporaryDirectory() as directory:
            rows = [dict(scene='s', frame=i, status='ok', plot='x',
                    metrics=dict(gt_count=1, generated_count=1),
                    foreground_box_count=1, foreground=result)
                    for i, result in enumerate((hit, miss, empty))]
            write_reports(Path(directory), rows, dict(model='toy', checkpoint_path='toy', sample_count=3))
            summary = json.loads((Path(directory) / 'summary.json').read_text())['foreground']
            self.assertEqual(summary['hit_rate'], 0.5)
            self.assertEqual(summary['per_box_num_points_similarity']['valid_boxes'], 3)
            self.assertAlmostEqual(summary['per_box_num_points_similarity']['mean'], 2 / 3)

    def test_ppp_plot_markers_density_range_and_gamma(self):
        mask = torch.zeros(1, 4, 6)
        mask[0, 1, 2] = 1
        sample = dict(point_mask=mask, rcs_target=mask, doppler_target=-mask)
        mass = np.linspace(0.0001, 0.002, 24).reshape(4, 6)
        fields = dict(cell_mass_grid=mass, rcs=np.ones((4, 6)), doppler=-np.ones((4, 6)))
        fig = plot_ppp_prediction(sample, fields, np.array([[0., 0., 1., -1.]]), 50., 'test', gamma=0.3)
        panels = [ax for ax in fig.axes if ax.get_title()]
        for ax in panels[:3]:
            self.assertEqual(ax.collections[0].get_sizes()[0], panels[7].collections[0].get_sizes()[0])
            np.testing.assert_allclose(ax.collections[0].get_offsets(), [[-50 / 6, 12.5]])
        self.assertEqual(panels[3].images[0].norm.vmin, mass.min())
        self.assertEqual(panels[3].images[0].norm.vmax, mass.max())
        np.testing.assert_allclose(panels[6].images[0].get_array(), mass ** 0.3)
        fig.canvas.draw()
        fig.clear()

    def test_both_models_all_samples_including_empty_prediction(self):
        for model in ('ppp', 'radargen'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                dataset = ToyDataset(directory)
                output = Path(directory) / 'results'
                output.mkdir()
                predictor = ToyPredictor(model)
                config = dict(model=model, seed=42, checkpoint_path='toy.pth', plot_dpi=30)
                provenance = dict(config, sample_count=2, selected_indices=[1, 0])
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'), patch(
                        'evaluation.validation.time.perf_counter', side_effect=[10., 10.25, 20., 20.75]):
                    rows = evaluate_samples(dataset, predictor, config, torch.device('cpu'), output, provenance)
                self.assertEqual(predictor.calls, 2)
                self.assertEqual(predictor.total_calls, 3)
                self.assertEqual(len(rows), 2)
                self.assertEqual([row["frame"] for row in rows], [1, 0])
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
                timings = [row['metrics']['inference_time_seconds'] for row in rows]
                self.assertEqual(timings, [0.25, 0.75])
                self.assertEqual(summary['metrics']['inference_time_seconds']['mean'], 0.5)
                self.assertEqual(summary['metrics']['inference_time_seconds']['std'], 0.25)
                self.assertEqual(summary['metrics']['inference_time_seconds']['median'], float(np.median(timings)))
                self.assertEqual(summary['empty_generated_samples'], 1)
                self.assertEqual(summary['metrics']['chamfer_distance_xy']['valid_samples'], 1)
                if model == 'ppp':
                    self.assertAlmostEqual(rows[1]['metrics']['ppp_expected_total_count'], 2.)
                for row in rows:
                    # Mark maps remain evaluable when point-cloud recovery is empty.
                    self.assertEqual(row['metrics']['rcs_mae_at_detections'], 0.5)
                    self.assertEqual(row['metrics']['rcs_mae_at_target_cells'], 0.5)
                    self.assertEqual(row['metrics']['doppler_mae_at_detections'], 1.5)
                    self.assertEqual(row['metrics']['doppler_mae_at_target_cells'], 1.5)
                self.assertEqual(summary['metrics']['rcs_mae_at_target_cells']['valid_samples'], 2)
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

    def test_tensorboard_replaces_png_and_allows_resume(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for model in ('ppp', 'radargen'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                dataset = ToyDataset(directory)
                output = Path(directory) / 'results'
                output.mkdir()
                config = dict(model=model, seed=42, checkpoint_path='toy.pth', tensorboard=True)
                provenance = dict(config, sample_count=2, selected_indices=[1, 0])
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'):
                    rows = evaluate_samples(dataset, ToyPredictor(model), config,
                                            torch.device('cpu'), output, provenance)
                self.assertFalse(list(output.rglob('*.png')))
                self.assertTrue(all(row['plot'] == '' for row in rows))
                events = EventAccumulator(str(output / 'tensorboard')).Reload()
                self.assertEqual([row.step for row in events.Images('validation/prediction')], [0, 1])
                restarted = ToyPredictor(model)
                with patch('radargen.training.radargen_ppp_dataset.validate_ppp_metadata'):
                    restored = evaluate_samples(dataset, restarted, config, torch.device('cpu'),
                                                output, provenance, resume=True)
                self.assertEqual(restored, rows)
                self.assertEqual(restarted.calls, 0)

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
