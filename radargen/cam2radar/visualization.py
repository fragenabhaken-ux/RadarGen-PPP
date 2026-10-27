"""Final-frame camera/PPP evaluation panels in physical BEV coordinates."""
from pathlib import Path

import numpy as np


CAMERA_NAMES = ('Right front', 'Left front', 'Right back', 'Left back')


def plot_camera_radar_prediction(sample, fields, points, *, title='', gamma=0.2):
    """Four inputs above GT dots, predicted fields, and sampled dots.

    fields contains physical cell_mass_grid, rcs, doppler (H,W) arrays;
    points and sample detections contain metric [x,y,RCS,Doppler]. Mark colour
    limits are shared across GT/heatmap/samples. Gamma affects mass display only.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    cameras = sample['camera_images'].detach().cpu().permute(0, 2, 3, 1).numpy()
    gt = sample['detections'].detach().cpu().numpy()
    points = np.asarray(points)
    radius = float(sample['point_limit'])
    maps = [np.asarray(fields[k]) for k in ('cell_mass_grid', 'rcs', 'doppler')]
    norms = []
    for column in (1, 2):
        values = [maps[column].ravel(), gt[:, column+1], points[:, column+1]]
        nonempty = [v for v in values if v.size]
        low = min(float(v.min()) for v in nonempty)
        high = max(float(v.max()) for v in nonempty)
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        norms.append(Normalize(low, high))
    fig = Figure(figsize=(18, 18), constrained_layout=True)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(4, 1, height_ratios=(0.8, 2, 2, 2))
    camera_grid = grid[0].subgridspec(1, 4)
    for index, (camera, name) in enumerate(zip(cameras, CAMERA_NAMES)):
        ax = fig.add_subplot(camera_grid[0, index])
        ax.imshow(camera)
        ax.set_title(name)
        ax.axis('off')
    cmaps = ('gray', 'viridis', 'coolwarm')
    units = ('Locations', 'RCS (dBsm)', 'Velocity (m/s)')
    extent = (-radius, radius, -radius, radius)
    for row in range(3):
        row_grid = grid[row+1].subgridspec(1, 3)
        for col in range(3):
            ax = fig.add_subplot(row_grid[0, col])
            if row == 1:
                display = maps[col] ** gamma if col == 0 else maps[col]
                image = ax.imshow(display, origin='upper', extent=extent,
                    cmap='hot' if col == 0 else cmaps[col],
                    norm=None if col == 0 else norms[col-1], interpolation='nearest')
                fig.colorbar(image, ax=ax, label=f'Cell mass^{gamma:g}' if col == 0 else units[col])
                ax.set_title('Predicted intensity' if col == 0 else f'Predicted {units[col]}')
            else:
                cloud = gt if row == 0 else points
                prefix = 'GT' if row == 0 else 'Sampled'
                if col == 0:
                    ax.set_facecolor('black')
                    ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c='white', linewidths=0)
                else:
                    image = ax.scatter(cloud[:, 0], cloud[:, 1], s=5,
                        c=cloud[:, col+1], cmap=cmaps[col], norm=norms[col-1], linewidths=0)
                    fig.colorbar(image, ax=ax, label=units[col])
                ax.set_title(f'{prefix} {units[col]} (N={len(cloud)})')
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius), xlabel='x (m)', ylabel='y (m)')
            ax.set_aspect('equal')
    fig.suptitle(f'{title}\nGT: {len(gt)} | Expected: {maps[0].sum():.1f} | Sampled: {len(points)}')
    return fig


def save_final_predictions(model, dataset, indices, output_dir, *, device, seed=42, mixed_precision='bf16'):
    """Write one final PNG per selected sample; no intermediate prediction files."""
    import torch
    from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
    from radargen.ppp.sampling import ppp_cell_masses

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    pipeline = RadarGenPPPPipeline(model, device=device, mixed_precision=mixed_precision)
    paths = []
    try:
        for index in indices:
            sample = dataset[index]
            identity = sample['data_info']
            # Per-sample seed makes the sampling independent of iteration order.
            pipeline.generator.manual_seed(seed + int(index))
            raw = pipeline.predict_camera_fields(sample)
            points = pipeline.sample_fields(raw)
            fields = dict(cell_mass_grid=ppp_cell_masses(raw['log_intensity']).numpy(),
                          rcs=raw['rcs_raw'].double().cpu().numpy() ** 3,
                          doppler=raw['doppler_raw'].double().cpu().numpy() ** 3)
            scene, frame = identity['scene_token'], int(identity['frame_idx'])
            figure = plot_camera_radar_prediction(sample, fields, points,
                                                   title=f'Camera2Radar | {scene} | frame {frame}')
            path = output_dir / f'{scene}_frame_{frame:06d}.png'
            figure.savefig(path, dpi=120)
            figure.clear()
            paths.append(str(path))
    finally:
        model.train(was_training)
    return paths
