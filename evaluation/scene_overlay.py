"""Dense PPP intensity/mark maps with aligned ground-truth point overlays."""
from contextlib import ExitStack
import io
import json
import logging
from pathlib import Path
import subprocess
import tarfile

import h5py
import numpy as np

LOG = logging.getLogger(__name__)


def render_overlays(scene_dir, models, output, *, formats, fps, dpi, gamma,
                    doppler_limit, rcs_limits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.scene_comparison import identifier

    scene_dir, output = Path(scene_dir), Path(output)
    for model in models:
        path = scene_dir / 'predictions' / f'{identifier(model)}.h5'
        with h5py.File(path, 'r') as saved:
            if json.loads(saved.attrs['provenance'])['model'] != 'ppp':
                continue
        dense = scene_dir / 'predictions' / f'{model}_dense.h5'
        if not dense.is_file():
            raise ValueError(f'Missing dense fields: {dense}; rerun scene prediction first')
        destination = scene_dir / 'overlays' / identifier(model)
        destination.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            gt = stack.enter_context(h5py.File(scene_dir / 'ground_truth.h5', 'r'))
            prediction = stack.enter_context(h5py.File(dense, 'r'))
            truth_meta, meta = [json.loads(f.attrs['provenance']) for f in (gt, prediction)]
            keys = sorted(gt['frames'])
            if not gt.attrs.get('complete') or not prediction.attrs.get('complete') or sorted(prediction['frames']) != keys or any(
                    truth_meta[k] != meta[k] for k in ('scene', 'frames', 'dataset_contract', 'manifest_identity')):
                raise ValueError('Overlay exports have different frame/dataset identities or are incomplete')
            radius = meta['coordinate_range']
            maximum = max(float(np.max(prediction['frames'][k]['heatmap'])) for k in keys)
            archive = stack.enter_context(tarfile.open(destination / 'frames.tar', 'w')) if 'tar' in formats else None
            encoder = None
            try:
                for number, key in enumerate(keys):
                    group = prediction['frames'][key]
                    points = gt['frames'][key]['points'][()]
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                    elapsed = (group.attrs['timestamp_us'] - prediction['frames'][keys[0]].attrs['timestamp_us']) / 1e6
                    fig.suptitle(f"{meta['label']} | scene {meta['scene']} | frame {int(key)} | {elapsed:.1f} s")
                    try:
                        for ax, mark in zip(axes, ('heatmap', 'doppler', 'rcs')):
                            values = group[mark][()]
                            intensity = mark == 'heatmap'
                            cmap = 'hot' if intensity else ('coolwarm' if mark == 'doppler' else 'viridis')
                            low, high = (0, max(maximum, 1e-12) ** gamma) if intensity else (
                                (-doppler_limit, doppler_limit) if mark == 'doppler' else rcs_limits)
                            title = 'Intensity' if intensity else ('Doppler' if mark == 'doppler' else 'RCS')
                            artist = ax.imshow(np.maximum(values, 0) ** gamma if intensity else values,
                                origin='upper', extent=(-radius, radius, -radius, radius),
                                cmap=cmap, vmin=low, vmax=high, interpolation='nearest')
                            scatter = dict(c='white') if intensity else dict(
                                c=points[:, 3 if mark == 'doppler' else 2], cmap=cmap, vmin=low, vmax=high)
                            ax.scatter(points[:, 0], points[:, 1], s=2 if intensity else 9,
                                       linewidths=0 if intensity else .35,
                                       edgecolors='none' if intensity else 'black', zorder=3, **scatter)
                            ax.set(title=f'{title} + ground truth', xlabel='x (m)', ylabel='y (m)',
                                   xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')
                            label = f'Expected points/cell (power {gamma:g})' if intensity else (
                                'Doppler (m/s)' if mark == 'doppler' else 'RCS (dBsm)')
                            fig.colorbar(artist, ax=ax, label=label, shrink=.8)
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format='png', dpi=dpi)
                        payload = buffer.getvalue()
                        filename = f'frame_{number:06d}.png'
                        if archive:
                            info = tarfile.TarInfo(filename)
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
                        if 'png' in formats:
                            (destination / filename).write_bytes(payload)
                        if 'mp4' in formats:
                            if encoder is None:
                                encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe',
                                    '-vcodec', 'png', '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                                    '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                    str(destination / 'overlay.mp4')], stdin=subprocess.PIPE)
                            encoder.stdin.write(payload)
                    finally:
                        plt.close(fig)
                    LOG.info('Overlay %s: %d/%d', model, number + 1, len(keys))
            finally:
                if encoder is not None:
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError('Overlay ffmpeg failed')
            (destination / 'render_config.json').write_text(json.dumps(dict(
                model=model, scene=meta['scene'], panels=['intensity', 'doppler', 'rcs'],
                ground_truth_overlay=True, fps=fps, dpi=dpi, gamma=gamma,
                doppler_limit=doppler_limit, rcs_limits=rcs_limits,
                export_signatures=[gt.attrs['signature'], prediction.attrs['signature']]), indent=2))
