"""Read physical radar detections without rasterization or deduplication."""
import numpy as np


def read_continuous_targets(handle, coordinate_range):
    """Call only after validating the sample's identity and preprocessing metadata."""
    group = handle['ppp_targets']
    xy, rcs, doppler = (group[key][()] for key in ('xy', 'rcs', 'doppler'))
    n = len(rcs)
    if xy.shape != (n, 2) or rcs.shape != (n,) or doppler.shape != (n,):
        raise ValueError('Invalid continuous radar target shapes')
    points = np.column_stack((xy, rcs, doppler))
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite continuous radar target')
    if not np.isfinite(coordinate_range) or coordinate_range <= 0:
        raise ValueError('Invalid coordinate range')
    if not (np.abs(xy) < coordinate_range).all():
        raise ValueError('Continuous radar target outside strict square range')
    if not n:
        raise ValueError('Empty ground truth; evaluation policy unresolved')
    return points
