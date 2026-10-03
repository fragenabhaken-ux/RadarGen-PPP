"""Rasterize exact filtered detections without changing physical marks."""

import numpy as np


def _select_indices(coords, rcs, indices):
    """Highest RCS per coordinate, then lowest original detection index."""
    _, groups = np.unique(coords, axis=0, return_inverse=True)
    order = np.lexsort((indices, -rcs[indices], groups))
    ordered_groups = groups[order]
    first = np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
    return indices[order[first]]


def create_ppp_targets_from_detections(xy, rcs, doppler, *, image_size, point_limit):
    """Return three float32 (H,W) grids and collision counts/index diagnostics.

    Inputs are metric flat-up xy, dBsm RCS, and m/s Doppler. Empty frames
    explicitly fail until an empty-frame training policy is approved.
    """
    xy, rcs, doppler = (np.asarray(v) for v in (xy, rcs, doppler))
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not np.isfinite(point_limit) or point_limit <= 0:
        raise ValueError("point_limit must be positive and finite")
    count = len(rcs) if rcs.ndim else -1
    if xy.shape != (count, 2) or rcs.shape != (count,) or doppler.shape != (count,):
        raise ValueError("PPP arrays must have aligned shapes (N,2), (N,), (N,)")
    if any(v.dtype.kind != 'f' or not np.isfinite(v).all() for v in (xy, rcs, doppler)):
        raise ValueError("PPP detections must be finite floating-point arrays")
    if not count:
        raise ValueError("Empty PPP frame: N_stored=0; empty-frame training policy is unresolved")
    if not (np.abs(xy) < point_limit).all():
        raise ValueError("PPP coordinates violate the strict square range; no clipping or refiltering")
    # Preserve the baseline helper's source-dtype arithmetic/np.round rule.
    continuous = (xy + point_limit) / (2 * point_limit) * image_size
    rounded = np.round(continuous).astype(np.int32)
    stage1 = _select_indices(rounded, rcs, np.arange(count))
    cells = np.floor(continuous[stage1]).astype(np.int64)
    if not ((cells >= 0) & (cells < image_size)).all():
        raise ValueError("PPP rasterization produced an out-of-range cell")
    selected = _select_indices(cells, rcs, stage1)
    cols = np.floor(continuous[selected, 0]).astype(np.int64)
    rows = image_size - 1 - np.floor(continuous[selected, 1]).astype(np.int64)
    mask, rcs_grid, doppler_grid = (np.zeros((image_size, image_size), np.float32) for _ in range(3))
    mask[rows, cols] = 1
    rcs_grid[rows, cols] = rcs[selected]
    doppler_grid[rows, cols] = doppler[selected]
    if not np.isfinite(rcs_grid).all() or not np.isfinite(doppler_grid).all():
        raise ValueError("PPP marks cannot be represented as finite float32")
    stats = dict(stored=count, after_rounded=len(stage1),
                 rounded_collisions=count-len(stage1),
                 final_cell_collisions=len(stage1)-len(selected), active_cells=len(selected),
                 rounded_indices=stage1, retained_indices=selected)
    assert int(mask.sum()) == stats['active_cells'] <= count
    return mask, rcs_grid, doppler_grid, stats
