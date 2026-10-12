"""Plot layout, physical coordinates, and headless output checks."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from radargen.visualization import plot_radargen_prediction


class RadarGenVisualizationChecks(unittest.TestCase):
    def setUp(self):
        self.gt = (np.ones((4, 6)), np.full((4, 6), -10.), np.full((4, 6), -5.))
        self.pred = (np.full((4, 6), 2.), np.full((4, 6), 20.), np.full((4, 6), 7.))

    def test_layout_scales_coordinates_and_save(self):
        points = np.array([[25., -25., 0., 30., -12.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'plot.png'
            fig = plot_radargen_prediction(self.gt, self.pred, points, output_path=path)
            self.assertGreater(path.stat().st_size, 0)
        panels = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(panels), 9)
        self.assertEqual(panels[0].images[0].origin, 'upper')
        self.assertEqual(tuple(panels[0].images[0].get_extent()), (-50., 50., -50., 50.))
        np.testing.assert_array_equal(panels[6].collections[0].get_offsets(), points[:, :2])
        for col in (1, 2):
            self.assertIs(panels[col].images[0].norm, panels[col + 3].images[0].norm)
            self.assertIs(panels[col].images[0].norm, panels[col + 6].collections[0].norm)
        self.assertEqual(panels[1].images[0].norm.vmax, 30.)
        self.assertEqual(panels[2].images[0].norm.vmin, -12.)

    def test_empty_points_and_constant_maps(self):
        maps = np.zeros((3, 4, 4))
        fig = plot_radargen_prediction(maps, maps, np.empty((0, 4)))
        fig.canvas.draw()
        panels = [ax for ax in fig.axes if ax.get_title()]
        self.assertTrue(all(ax.texts[0].get_text() == 'No points generated' for ax in panels[6:]))
        self.assertTrue(all(ax.images[0].norm.vmin < ax.images[0].norm.vmax for ax in panels[:6]))

    def test_invalid_inputs(self):
        points = np.empty((0, 4))
        for kwargs in ({'coordinate_range': 0}, {'coordinate_range': float('nan')}):
            with self.assertRaises(ValueError):
                plot_radargen_prediction(self.gt, self.pred, points, **kwargs)
        for bad in (np.empty((0, 3)), np.array([[0., 0., float('nan'), 0.]])):
            with self.assertRaises(ValueError):
                plot_radargen_prediction(self.gt, self.pred, bad)
        for bad in ((-self.gt[0], *self.gt[1:]), (self.gt[0][:2], *self.gt[1:])):
            with self.assertRaises(ValueError):
                plot_radargen_prediction(bad, self.pred, points)


if __name__ == '__main__':
    unittest.main()
