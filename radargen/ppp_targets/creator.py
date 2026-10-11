"""Rasterize exact filtered detections without changing physical marks."""

import numpy as np


def _select_indices(coords, rcs, indices):
    """Highest RCS per coordinate, then lowest original detection index."""
    if not len(indices):
        return indices
    _, groups = np.unique(coords, axis=0, return_inverse=True)
    order = np.lexsort((indices, -rcs[indices], groups))
    ordered_groups = groups[order]
    first = np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
    return indices[order[first]]


def create_ppp_targets_from_detections(xy, rcs, doppler, *, image_size, point_limit):
    """Return three float32 (H,W) grids and collision counts/index diagnostics.

    Inputs are metric flat-up xy, dBsm RCS, and m/s Doppler. Boundary
    detections use the final cell; outside detections are excluded. Empty
    frames return zero grids.
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
    # Preserve the baseline helper's source-dtype arithmetic/np.round rule.
    continuous = (xy + point_limit) / (2 * point_limit) * image_size
    raw_cells = np.floor(continuous).astype(np.int64)
    inside = ((xy >= -point_limit) & (xy <= point_limit)).all(axis=1)
    boundary = (continuous == image_size) & inside[:, None]
    corrected_cells = np.where(boundary, image_size-1, raw_cells)
    valid = inside & ((corrected_cells >= 0) & (corrected_cells < image_size)).all(axis=1)
    excluded = np.flatnonzero(~valid)
    boundary_indices = np.flatnonzero(boundary.any(axis=1))
    eligible = np.flatnonzero(valid)
    rounded = np.round(continuous).astype(np.int32)
    stage1 = _select_indices(rounded[eligible], rcs, eligible)
    cells = corrected_cells[stage1]
    selected = _select_indices(cells, rcs, stage1)
    cols = corrected_cells[selected, 0]
    rows = image_size - 1 - corrected_cells[selected, 1]
    mask, rcs_grid, doppler_grid = (np.zeros((image_size, image_size), np.float32) for _ in range(3))
    mask[rows, cols] = 1
    rcs_grid[rows, cols] = rcs[selected]
    doppler_grid[rows, cols] = doppler[selected]
    if not np.isfinite(rcs_grid).all() or not np.isfinite(doppler_grid).all():
        raise ValueError("PPP marks cannot be represented as finite float32")
    stats = dict(stored=count, after_rounded=len(stage1),
                 rounded_collisions=len(eligible)-len(stage1),
                 final_cell_collisions=len(stage1)-len(selected), active_cells=len(selected),
                 rounded_indices=stage1, retained_indices=selected,
                 excluded_indices=excluded, boundary_indices=boundary_indices,
                 offending_indices=np.union1d(excluded, boundary_indices),
                 continuous_grid=continuous, raw_cell_xy=raw_cells,
                 corrected_cell_xy=corrected_cells)
    assert int(mask.sum()) == stats['active_cells'] <= count
    return mask, rcs_grid, doppler_grid, stats
