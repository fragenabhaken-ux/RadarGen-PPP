"""Analytical discrete-PPP counts, separate from sampled RadarGen metrics."""
import numpy as np
from evaluation.box_utils import get_points_in_box


def make_ppp_grid_xy(H, W, coordinate_range):
    if H <= 0 or W <= 0 or not np.isfinite(coordinate_range) or coordinate_range <= 0:
        raise ValueError('Invalid grid geometry')
    r, c = np.indices((H, W))
    R = float(coordinate_range)
    return np.column_stack((((c + .5) / W * 2 * R - R).ravel(),
                            ((H - r - .5) / H * 2 * R - R).ravel()))


def _masses(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Expected finite nonnegative 2D cell masses')
    if not np.isfinite(value.sum()):
        raise ValueError('Nonfinite total intensity')
    return value


def expected_total_count(cell_mass_grid):
    return float(_masses(cell_mass_grid).sum())


def expected_count_in_box(cell_mass_grid, box, grid_xy):
    mass = _masses(cell_mass_grid)
    grid_xy = np.asarray(grid_xy)
    if grid_xy.shape != (mass.size, 2) or not np.isfinite(grid_xy).all():
        raise ValueError('Grid coordinates must match flattened masses')
    _, mask = get_points_in_box(grid_xy, box.bottom_corners())
    return float(mass.ravel()[mask].sum())


def compute_ppp_count_metrics(cell_mass_grid, gt_pcl, bounding_boxes, grid_xy):
    mass = _masses(cell_mass_grid)
    gt = np.asarray(gt_pcl)
    if gt.ndim != 2 or gt.shape[1] != 4 or not np.isfinite(gt).all():
        raise ValueError('Expected continuous evaluator GT with shape (N,4)')
    total = expected_total_count(mass)
    result = dict(ppp_expected_total_count=total,
                  ppp_total_count_abs_error=abs(total - len(gt)),
                  ppp_expected_count=[], ppp_expected_count_abs_error=[],
                  ppp_hit_probability_gt_positive=[])
    for box in bounding_boxes:
        expected = expected_count_in_box(mass, box, grid_xy)
        inside, _ = get_points_in_box(gt, box.bottom_corners())
        result['ppp_expected_count'].append(expected)
        result['ppp_expected_count_abs_error'].append(abs(expected - len(inside)))
        if len(inside):
            result['ppp_hit_probability_gt_positive'].append(float(-np.expm1(-expected)))
    return result


def aggregate_ppp_metrics(aggregator, metrics):
    for key in ('ppp_expected_total_count', 'ppp_total_count_abs_error'):
        aggregator['per_sample_scores'].setdefault(key, []).append(metrics[key])
    for key in ('ppp_expected_count', 'ppp_expected_count_abs_error', 'ppp_hit_probability_gt_positive'):
        aggregator['per_box_scores'].setdefault(key, []).extend(metrics[key])
