"""NumPy views for diagnostics; target lists always retain repeated cells."""
import numpy as np


def locate_detections(detections, shape, point_limit):
    points = np.asarray(detections)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError('detections must be finite (N,4)')
    if not np.isfinite(point_limit) or point_limit <= 0:
        raise ValueError('point_limit must be positive and finite')
    h, w = shape
    points = points[(np.abs(points[:, :2]) <= point_limit).all(axis=1)]
    uv = (points[:, :2] + point_limit) / (2 * point_limit)
    cols = np.floor(uv[:, 0] * w).astype(np.int64).clip(0, w-1)
    rows = h - 1 - np.floor(uv[:, 1] * h).astype(np.int64).clip(0, h-1)
    return points, rows, cols


def sample_detections(sample, shape, point_limit):
    if 'detections' in sample:
        value = sample['detections']
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return locate_detections(value, shape, point_limit)[0]
    mask = sample['point_mask'][0].numpy().astype(bool)
    rows, cols = np.nonzero(mask)
    h, w = shape
    return np.column_stack((point_limit*(2*(cols+.5)/w-1),
                            point_limit*(1-2*(rows+.5)/h),
                            sample['rcs_target'][0].numpy()[mask],
                            sample['doppler_target'][0].numpy()[mask]))


def display_grids(points, shape, point_limit):
    """Count and mean-mark grids for display only; metrics use original points."""
    points, rows, cols = locate_detections(points, shape, point_limit)
    counts = np.zeros(shape, dtype=np.float64)
    np.add.at(counts, (rows, cols), 1)
    marks = []
    for i in (2, 3):
        values = np.zeros(shape, dtype=np.float64)
        np.add.at(values, (rows, cols), points[:, i])
        marks.append(values / counts.clip(1))
    return counts, *marks
