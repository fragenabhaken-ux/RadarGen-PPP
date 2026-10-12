"""Standalone RadarGen diagnostics from physical maps and recovered point clouds."""

from pathlib import Path

import numpy as np


def plot_radargen_prediction(
    gt_maps,
    predicted_maps,
    points,
    *,
    coordinate_range=50.0,
    title=None,
    output_path=None,
    dpi=110,
):
    """Render a 3 x 3 GT/prediction/generated-point comparison.

    ``gt_maps`` and ``predicted_maps`` are sequences of three finite (H, W)
    arrays in PD, RCS, Doppler order, all with the same shape. Pass the
    Gaussian-splatted PD and nearest-point (Voronoi) mark maps created by
    RadarGen's target pipeline as GT, and decoded physical maps as predictions.
    Do not pass normalized RGB training images or PPP log-intensity fields.

    ``points`` is a finite (N, 4) [x, y, rcs, doppler] or (N, 5)
    [x, y, z, rcs, doppler] array, in meters/dBsm/m/s. It should be the
    point cloud recovered from these predictions; this function never samples
    points or runs inference. Empty point clouds are supported.

    Maps use upper-origin raster coordinates over [-coordinate_range,
    coordinate_range] meters on both axes, matching radar_maps_to_pcl.
    Each modality shares a color scale across GT, prediction, and point marks.
    Returns a matplotlib Figure; optionally saves it, creating parent folders.
    Uses a standalone Agg canvas without changing matplotlib's global backend.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    if not np.isfinite(coordinate_range) or coordinate_range <= 0:
        raise ValueError("coordinate_range must be finite and positive")

    def validate_maps(maps, name):
        if len(maps) != 3:
            raise ValueError(f"{name} must contain PD, RCS, and Doppler maps")
        arrays = tuple(np.asarray(m, dtype=np.float64) for m in maps)
        if any(m.ndim != 2 or not m.size or not np.isfinite(m).all() for m in arrays):
            raise ValueError(f"{name} must contain nonempty finite 2D maps")
        return arrays

    gt = validate_maps(gt_maps, "gt_maps")
    pred = validate_maps(predicted_maps, "predicted_maps")
    if any(m.shape != gt[0].shape for m in (*gt, *pred)):
        raise ValueError("All GT and predicted maps must have the same shape")
    if np.any(gt[0] < 0) or np.any(pred[0] < 0):
        raise ValueError("PD maps must be nonnegative")
    pcl = np.asarray(points, dtype=np.float64)
    if pcl.ndim != 2 or pcl.shape[1] not in (4, 5) or not np.isfinite(pcl).all():
        raise ValueError("points must be a finite (N, 4) or (N, 5) array")
    marks = pcl[:, -2:]

    norms = []
    for column in range(3):
        lo = min(float(gt[column].min()), float(pred[column].min()))
        hi = max(float(gt[column].max()), float(pred[column].max()))
        if column and len(pcl):
            lo = min(lo, float(marks[:, column - 1].min()))
            hi = max(hi, float(marks[:, column - 1].max()))
        if column == 0:
            lo = 0.0
        if lo == hi:
            padding = max(abs(lo) * 0.01, 1e-6)
            lo, hi = (0.0, padding) if column == 0 else (lo - padding, hi + padding)
        norms.append(Normalize(vmin=lo, vmax=hi))

    fig = Figure(figsize=(14, 12), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 3)
    extent = (-coordinate_range, coordinate_range, -coordinate_range, coordinate_range)
    cmaps = ("hot", "viridis", "coolwarm")
    units = ("PD", "RCS (dBsm)", "Doppler (m/s)")
    panel_titles = (
        ("GT PD (Gaussian)", "GT RCS (Voronoi)", "GT Doppler (Voronoi)"),
        ("Predicted PD", "Predicted RCS", "Predicted Doppler"),
    )
    for row, maps in enumerate((gt, pred)):
        for col, image in enumerate(maps):
            ax = axes[row, col]
            im = ax.imshow(image, origin="upper", extent=extent,
                           cmap=cmaps[col], norm=norms[col], interpolation="nearest")
            ax.set_title(panel_titles[row][col])
            fig.colorbar(im, ax=ax, label=units[col])

    point_titles = ("Generated points", "Generated points with RCS", "Generated points with Doppler")
    for col, ax in enumerate(axes[2]):
        if len(pcl):
            if col == 0:
                ax.scatter(pcl[:, 0], pcl[:, 1], s=8, color="black")
            else:
                scatter = ax.scatter(pcl[:, 0], pcl[:, 1], s=8,
                                     c=marks[:, col - 1], cmap=cmaps[col], norm=norms[col])
                fig.colorbar(scatter, ax=ax, label=units[col])
        else:
            ax.text(0.5, 0.5, "No points generated", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_title(f"{point_titles[col]} (N={len(pcl)})")

    for ax in axes.flat:
        ax.set_xlim(-coordinate_range, coordinate_range)
        ax.set_ylim(-coordinate_range, coordinate_range)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    if title:
        fig.suptitle(title)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig
