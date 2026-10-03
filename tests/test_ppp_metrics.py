"""Small CPU tests; only NumPy and Matplotlib required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from evaluation.ppp_metrics import make_ppp_grid_xy, expected_count_in_box, compute_ppp_count_metrics, aggregate_ppp_metrics


class Box:
    def __init__(self, corners):
        self.corners = np.vstack((np.array(corners).T, np.zeros(4)))
    def bottom_corners(self):
        return self.corners


class MetricsTests(unittest.TestCase):
    def test_geometry_and_rotated_box(self):
        grid = make_ppp_grid_xy(3, 3, 1.5)
        np.testing.assert_array_equal(grid[[0, -1]], [[-1, 1], [1, -1]])
        # Diamond excludes bounding-rectangle corners but includes center/axes.
        box = Box([[0, 1.4], [1.4, 0], [0, -1.4], [-1.4, 0]])
        self.assertEqual(expected_count_in_box(np.ones((3, 3)), box, grid), 5.)

    def test_continuous_gt_counts_positive_subset_and_aggregation(self):
        mass = np.ones((3, 3))
        grid = make_ppp_grid_xy(3, 3, 1.5)
        center = Box([[-.4, -.4], [.4, -.4], [.4, .4], [-.4, .4]])
        outside = Box([[2, 2], [3, 2], [3, 3], [2, 3]])
        # Two distinct GT detections in the same training pixel remain TWO.
        gt = np.array([[.1, .1, 0, 0], [.2, .2, 0, 0]])
        m = compute_ppp_count_metrics(mass, gt, [center, outside], grid)
        self.assertEqual(m['ppp_total_count_abs_error'], 7.)
        self.assertEqual(m['ppp_expected_count_abs_error'], [1., 0.])
        np.testing.assert_allclose(m['ppp_hit_probability_gt_positive'], [-np.expm1(-1.)])
        agg = {'per_sample_scores': {'baseline': [12]}, 'per_box_scores': {}}
        aggregate_ppp_metrics(agg, m)
        self.assertEqual(agg['per_sample_scores']['baseline'], [12])
        self.assertEqual(agg['per_sample_scores']['ppp_expected_total_count'], [9.])

    def test_empty_boxes_zero_and_invalid_mass(self):
        grid = make_ppp_grid_xy(1, 1, 1.)
        m = compute_ppp_count_metrics(np.zeros((1, 1)), np.empty((0, 4)), [], grid)
        self.assertEqual(m['ppp_expected_total_count'], 0.)
        self.assertEqual(m['ppp_hit_probability_gt_positive'], [])
        for bad in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_ppp_count_metrics(np.array([[bad]]), np.empty((0, 4)), [], grid)


if __name__ == '__main__':
    unittest.main()
