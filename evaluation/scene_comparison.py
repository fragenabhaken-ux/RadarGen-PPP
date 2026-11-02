"""Reusable scene inference and CPU-only comparison rendering (format version 1)."""
import gc
import hashlib
import io
import json
import logging
import re
import subprocess
import tarfile
from pathlib import Path

import h5py
import numpy as np

LOG = logging.getLogger(__name__)
MARKS = ('doppler', 'rcs', 'heatmap')


def read_config(path):
    import yaml
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Invalid identifier: {value}')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _array(group, name, value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}')
    group.create_dataset(name, data=value, compression='gzip', shuffle=True)


def _export(path, provenance, records, produce):
    """Resume committed frame groups; publish a complete file atomically."""
    signature = digest(provenance)
    if path.exists():
        with h5py.File(path, 'r') as f:
            if f.attrs.get('signature') != signature or not f.attrs.get('complete'):
                raise ValueError(f'Incompatible export: {path}; use a different model ID/output root')
        LOG.info('Reusing complete export %s', path)
        return
    partial = path.with_suffix('.partial.h5')
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(partial, 'a') as f:
        if 'signature' in f.attrs and f.attrs['signature'] != signature:
            raise ValueError(f'Incompatible partial export: {partial}')
        f.attrs.update(signature=signature, provenance=json.dumps(provenance), format_version=1,
                       point_columns='x,y,rcs,doppler', complete=False)
        frames = f.require_group('frames')
        for index, record in enumerate(records):
            expected = record[1]
            key = f"{expected['frame_index']:06d}"
            if key in frames:
                if frames[key].attrs.get('complete', False):
                    continue
                del frames[key]
            points, heatmap = produce(index, record)
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError('Expected Nx4 point cloud: x,y,rcs,doppler')
            group = frames.create_group(key)
            group.attrs.update(frame_index=expected['frame_index'],
                               timestamp_us=expected['reference_timestamp_us'],
                               reference_token=expected['reference_sample_data_token'])
            _array(group, 'points', points)
            group['points'].attrs.update(columns='x,y,rcs,doppler', units='m,m,dBsm,m/s')
            if heatmap is not None:
                if np.asarray(heatmap).ndim != 2:
                    raise ValueError('Expected 2D heatmap')
                _array(group, 'heatmap', heatmap)
                group['heatmap'].attrs.update(kind=provenance['heatmap_kind'], origin='upper',
                    extent=[-provenance['coordinate_range'], provenance['coordinate_range'],
                            -provenance['coordinate_range'], provenance['coordinate_range']])
            f.flush()
            group.attrs['complete'] = True
            f.flush()
            LOG.info('%s: frame %d/%d', path.stem, index + 1, len(records))
        f.attrs['complete'] = True
        f.flush()
    partial.replace(path)


def predict_scene(config_path, models=None):
    import pyrallis
    import torch
    from diffusion.utils.config import SanaConfig
    from radargen.training.ppp_manifest import dataset_contract
    from radargen.training.radargen_ppp_dataset import validate_radargen_ppp_dataset_from_config
    from evaluation.validation import PPPPredictor, RadarGenPredictor

    config = read_config(config_path)
    scene = identifier(config['scene'])
    chosen = models or list(config['models'])
    if len(set(chosen)) != len(chosen):
        raise ValueError('Select each model only once')
    for name in chosen:
        identifier(name)
        if name not in config['models']:
            raise ValueError(f'Unknown model: {name}')
    with open(config['dataset_config']) as stream:
        data = pyrallis.load(SanaConfig, stream)
    data.data.extra.update(eval_split='val', ppp_scene_tokens=None)
    contract = dataset_contract(data)
    key = digest(contract)
    default_index = Path(__file__).resolve().parents[1] / 'output/ppp_manifests' / f'validation_{key}.json'
    dataset = validate_radargen_ppp_dataset_from_config(
        data, index_path=config.get('dataset_index_path') or default_index)
    indices = [i for i, r in enumerate(dataset.frame_index) if r[1]['scene_token'] == scene]
    if not indices:
        raise ValueError(f'Unknown validation scene: {scene}')
    records = [dataset.frame_index[i] for i in indices]
    directory = Path(config['output_root']).expanduser() / scene
    common = dict(scene=scene, coordinate_range=dataset.point_limit, fps=config.get('fps', 10),
                  frames=[r[1] for r in records], dataset_contract=contract,
                  manifest_identity=dataset.manifest_identity)

    def ground_truth(index, record):
        # The dataset access checks indexed file identity and all consumed metadata.
        sample = dataset[indices[index]]
        return sample['detections'].numpy(), None

    _export(directory / 'ground_truth.h5', dict(common, model='ground_truth', label='Ground truth'),
            records, ground_truth)
    device = torch.device(config.get('device', 'cuda'))
    for name in chosen:
        entry = config['models'][name]
        cfg = read_config(entry['validation_config'])
        cfg.update(entry.get('overrides', {}))
        cfg['seed'] = config.get('seed', 42)
        checkpoint = Path(cfg['checkpoint_path']).expanduser().resolve()
        stat = checkpoint.stat()
        model_path = Path(cfg['model_config'])
        provenance = dict(common, model=cfg['model'], label=entry.get('label', name), seed=cfg['seed'],
                          validation_config=cfg, model_config_text=model_path.read_text(),
                          checkpoint=dict(path=str(checkpoint), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                          heatmap_kind='expected_points_per_cell' if cfg['model'] == 'ppp' else 'gaussianized_blobs')
        destination = directory / 'predictions' / f'{name}.h5'
        # Instantiate lazily so complete exports never reload model weights.
        predictor = None
        camera_dataset = None
        with model_path.open() as stream:
            model_config = pyrallis.load(SanaConfig, stream)
        if model_config.data.image_size != dataset.resolution or any(
                model_config.data.extra.get(k) != v for k, v in dataset.normalization.items()):
            raise ValueError(f'Model/data geometry or normalization mismatch: {name}')

        def produce(index, record):
            nonlocal predictor, camera_dataset
            if predictor is None:
                cls = {'ppp': PPPPredictor, 'radargen': RadarGenPredictor}[cfg['model']]
                predictor = cls(model_config, cfg, device)
                if getattr(predictor, 'raw_cameras', False):
                    from radargen.cam2radar.dataset import CameraRadarDataset
                    camera_dataset = CameraRadarDataset.from_records(
                        [dict(expected=r[1]) for r in records], dataset.adapter.trucksc.dataroot,
                        dataset.normalization)
            sample = dataset[indices[index]]
            with h5py.File(record[0], 'r') as source:
                conditions = [source[f'conditioning/{k}'][()] for k in ('appearance', 'semantics', 'velocity')]
            predictor.seed_sample(indices[index])
            with torch.inference_mode():
                points, fields = predictor.predict(camera_dataset[index] if camera_dataset is not None else sample,
                                                   conditions)
            heatmap = fields['cell_mass_grid'] if cfg['model'] == 'ppp' else fields[0]
            return points, np.asarray(heatmap).squeeze()

        _export(destination, provenance, records, produce)
        del predictor, camera_dataset
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def render_scene(scene_dir, models, output, *, marks=('doppler',), formats=('tar',), fps=None,
                 dpi=110, gamma=0.2, doppler_limit=30., rcs_limits=(-20., 66.)):
    """Render saved exports only; no model imports or inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contextlib import ExitStack

    if set(marks) - set(MARKS) or len(set(models)) != len(models):
        raise ValueError('Unknown mark or duplicate model selection')
    marks = [m for m in MARKS if m in marks]
    if not marks or gamma <= 0 or doppler_limit <= 0 or dpi <= 0 or rcs_limits[0] >= rcs_limits[1]:
        raise ValueError('Select valid rows and positive gamma/Doppler limits')
    if not formats or set(formats) - {'tar', 'png', 'mp4'}:
        raise ValueError('Formats must be tar, png and/or mp4')
    scene_dir, output = Path(scene_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(scene_dir / 'ground_truth.h5', 'r'))]
        files += [stack.enter_context(h5py.File(scene_dir / 'predictions' / f'{identifier(m)}.h5', 'r')) for m in models]
        provenance = [json.loads(f.attrs['provenance']) for f in files]
        keys = sorted(files[0]['frames'])
        for f, p in zip(files, provenance):
            if not f.attrs.get('complete') or sorted(f['frames']) != keys or any(
                    p[k] != provenance[0][k] for k in ('scene', 'frames', 'dataset_contract', 'manifest_identity')):
                raise ValueError('Exports are incomplete or have different frame/dataset identities')
        radius = provenance[0]['coordinate_range']
        fps = fps or provenance[0]['fps']
        if fps <= 0:
            raise ValueError('FPS must be positive')
        # Fixed per-export heatmap maxima across all frames; PPP and RadarGen units differ.
        maxima = [max(float(np.max(f['frames'][k]['heatmap'])) for k in keys)
                  if 'heatmap' in marks and i else 1. for i, f in enumerate(files)]
        archive = stack.enter_context(tarfile.open(output / 'frames.tar', 'w')) if 'tar' in formats else None
        encoder = None
        try:
            for number, key in enumerate(keys):
                fig, axes = plt.subplots(len(marks), len(files), squeeze=False,
                                         figsize=(4 * len(files), 3.8 * len(marks)), constrained_layout=True)
                timestamp = files[0]['frames'][key].attrs['timestamp_us']
                start = files[0]['frames'][keys[0]].attrs['timestamp_us']
                fig.suptitle(f"Scene {provenance[0]['scene']} | frame {int(key)} | {(timestamp-start)/1e6:.1f} s")
                for row, mark in enumerate(marks):
                    for col, (f, p) in enumerate(zip(files, provenance)):
                        ax = axes[row, col]
                        if mark == 'heatmap' and col == 0:
                            ax.axis('off')
                            continue
                        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal',
                               xlabel='x (m)', ylabel='y (m)', title=f"{p['label']} — {mark.upper() if mark == 'rcs' else mark.capitalize()}")
                        group = f['frames'][key]
                        if mark == 'heatmap':
                            mass = group['heatmap'][()]
                            artist = ax.imshow(np.maximum(mass, 0) ** gamma, origin='upper',
                                               extent=(-radius, radius, -radius, radius), cmap='magma',
                                               vmin=0, vmax=max(maxima[col], 1e-12) ** gamma)
                            unit = 'Expected points/cell' if p['model'] == 'ppp' else 'Gaussianized blobs'
                            label = f'{unit} (power {gamma:g})'
                        else:
                            points = group['points'][()]
                            is_doppler = mark == 'doppler'
                            lo, hi = (-doppler_limit, doppler_limit) if is_doppler else rcs_limits
                            artist = ax.scatter(points[:, 0], points[:, 1], c=points[:, 3 if is_doppler else 2],
                                                s=3, cmap='coolwarm' if is_doppler else 'viridis', vmin=lo, vmax=hi,
                                                linewidths=0)
                            label = 'Doppler (m/s)' if is_doppler else 'RCS (dBsm)'
                        fig.colorbar(artist, ax=ax, label=label, shrink=.8)
                try:
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=dpi)
                    payload = buffer.getvalue()
                    filename = f'frame_{number:06d}.png'
                    if archive:
                        info = tarfile.TarInfo(filename); info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                    if 'png' in formats:
                        (output / filename).write_bytes(payload)
                    if 'mp4' in formats:
                        if encoder is None:
                            encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe',
                                '-vcodec', 'png', '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                                '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                str(output / 'comparison.mp4')], stdin=subprocess.PIPE)
                        encoder.stdin.write(payload)
                finally:
                    plt.close(fig)
                LOG.info('Rendered %d/%d', number + 1, len(keys))
        finally:
            if encoder is not None:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('ffmpeg failed')
        (output / 'render_config.json').write_text(json.dumps(dict(
            scene_dir=str(scene_dir.resolve()), models=models, marks=marks, formats=list(formats), fps=fps,
            dpi=dpi, gamma=gamma, doppler_limit=doppler_limit, rcs_limits=rcs_limits,
            export_signatures=[f.attrs['signature'] for f in files]), indent=2))
