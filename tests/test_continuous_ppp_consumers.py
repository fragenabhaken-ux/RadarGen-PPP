"""Focused diagnostics and plotting checks with repeated cells and empty GT."""
import unittest
import numpy as np
import torch
from radargen.ppp_targets.continuous import locate_detections, display_grids
from radargen.training.ppp_overfit import summarize_fields
from evaluation.validation import plot_ppp_prediction


class ContinuousConsumerTests(unittest.TestCase):
    def test_collision_diagnostics_and_boundaries(self):
        points = np.array([[0., 0., 2., -3.], [.1, .1, -4., 5.], [-1., -1., 6., 7.], [1., 1., 8., 9.], [2., 0., 1., 1.]])
        valid, rows, cols = locate_detections(points, (2, 2), 1.)
        self.assertEqual(len(valid), 4)
        self.assertEqual(rows.tolist(), [0, 0, 1, 0])
        self.assertEqual(cols.tolist(), [1, 1, 0, 1])
        counts, rcs, doppler = display_grids(points, (2, 2), 1.)
        stats, _, _ = summarize_fields(np.zeros((3, 2, 2)), counts > 0, rcs, doppler,
                {'rcs': (-20, 66), 'doppler': (-10, 10)}, detections=valid, point_limit=1.)
        self.assertEqual(stats['detection_count'], 4)
        self.assertEqual(stats['collision_count'], 2)
        self.assertEqual(stats['rcs_mae_at_targets'], 5.)
        self.assertEqual(stats['doppler_mae_at_targets'], 6.)

    def test_continuous_plot_retains_positions_and_marks_including_empty(self):
        fields = dict(cell_mass_grid=np.ones((2, 2))/4, rcs=np.zeros((2, 2)), doppler=np.zeros((2, 2)))
        for points in (np.array([[0., 0., 2., -3.], [.1, .1, -4., 5.]]), np.empty((0, 4))):
            fig = plot_ppp_prediction(dict(detections=torch.tensor(points)), fields, np.empty((0, 4)), 1., 'test')
            axes = [ax for ax in fig.axes if ax.get_title()]
            for col in range(3):
                np.testing.assert_allclose(axes[col].collections[0].get_offsets(), points[:, :2])
                if col:
                    np.testing.assert_array_equal(axes[col].collections[0].get_array(), points[:, col+1])
            fig.canvas.draw()
            fig.clear()
        empty = np.empty((0, 4))
        grids = display_grids(empty, (2, 2), 1.)
        stats, _, _ = summarize_fields(np.zeros((3, 2, 2)), grids[0] > 0, *grids[1:],
                {'rcs': (-20, 66), 'doppler': (-10, 10)}, detections=empty, point_limit=1.)
        self.assertEqual(stats['detection_count'], 0)
        self.assertEqual(stats['count_abs_error_detections'], 1.)

    def test_prediction_diagnostics_write_complete_artifacts(self):
        from contextlib import nullcontext
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        import json
        import h5py
        from radargen.losses import PPPLoss
        from radargen.training.ppp_overfit import diagnose_overfit

        class Model(torch.nn.Module):
            def forward(self, *conditions):
                return torch.zeros(1, 3, 2, 2)

        with TemporaryDirectory() as directory:
            source = Path(directory) / 'sample.h5'
            with h5py.File(source, 'w') as handle:
                handle['ppp_targets/rcs'] = np.array([2., -4.])
                handle['metadata/conditioning_coordinate_range'] = 1.
                for name in ('appearance', 'semantics', 'velocity'):
                    handle[f'conditioning/{name}'] = np.zeros((2, 2, 3), dtype=np.uint8)
            class Dataset:
                frame_index = [(source,)]
                def __len__(self):
                    return 1
                def __getitem__(self, index):
                    return dict(detections=torch.tensor([[0., 0., 2., -3.], [.1, .1, -4., 5.]]),
                                point_limit=1., bev_color_map=torch.zeros(3, 2, 2),
                                bev_seg_map=torch.zeros(3, 2, 2), bev_velocity_map=torch.zeros(3, 2, 2),
                                data_info=dict(scene_token='test', frame_idx=0))
            config = SimpleNamespace(work_dir=directory, model=SimpleNamespace(model='DinoPPP'),
                data=SimpleNamespace(extra=dict(coordinate_range=1., rcs_min=-20., rcs_max=66.,
                                                 doppler_min=-10., doppler_max=10.)),
                train=SimpleNamespace(extra=dict(velocity_loss_mode='signed_l1',
                                                 ppp_weight=1., rcs_weight=1., doppler_weight=1.)))
            accelerator = SimpleNamespace(device=torch.device('cpu'), autocast=nullcontext)
            model = Model()
            diagnose_overfit(model, None, Dataset(), None, PPPLoss((2, 2)), config, accelerator, 0)
            output = Path(directory) / 'overfit_diagnostics'
            row = json.loads((output / 'metrics.jsonl').read_text())
            self.assertEqual(row['detection_count'], 2)
            self.assertEqual(row['collision_count'], 1)
            self.assertTrue((output / 'step_000000_sample_0.png').is_file())
            with np.load(output / 'step_000000_sample_0.npz') as saved:
                self.assertEqual(saved['detections'].shape, (2, 4))
                self.assertEqual(saved['detection_counts'].sum(), 2)
            self.assertTrue(model.training)
            # TensorBoard replaces loose PNG output, while metrics/raw data remain.
            from radargen.training.tensorboard_logging import tensorboard_writer
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            config.tensorboard = True
            with tensorboard_writer(directory, True) as writer:
                diagnose_overfit(model, None, Dataset(), None, PPPLoss((2, 2)),
                                 config, accelerator, 1, writer=writer)
            self.assertFalse((output / 'step_000001_sample_0.png').exists())
            self.assertTrue((output / 'step_000001_sample_0.npz').exists())
            events = EventAccumulator(str(Path(directory) / 'tensorboard')).Reload()
            self.assertEqual(events.Images('overfit/sample_0')[-1].step, 1)

