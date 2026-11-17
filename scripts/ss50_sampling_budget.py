"""CPU-only intensity-scale sweep over archived DINO fields on exact SS50 identities."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from evaluation.validation import (sample_metrics, foreground_metrics, validation_boxes,
                                   write_reports, json_values)
from evaluation.metrics import MMDLoss, RBF
from evaluation.ppp_metrics import compute_ppp_count_metrics, make_ppp_grid_xy
from radargen.core.normalization import NormalizationConfig

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen')
DINO = ROOT / 'validation/2788375_PPP_validation_ss50_2773475'
RADARGEN = ROOT / 'validation/2404702_RadarGen_validation'
ARCHIVE = ROOT / 'validation/FULL_validation_20261108/dino_2773475'


def read_rows(directory):
    return [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]


def sample_scaled(mass, rcs, doppler, radius, scale, rng):
    """Total-Poisson plus categorical draws is exactly the independent-cell PPP law."""
    total = float(mass.sum())
    count = rng.poisson(total * scale)
    indices = rng.choice(mass.size, size=count, replace=True, p=mass.ravel() / total)
    rows, cols = np.unravel_index(indices, mass.shape)
    height, width = mass.shape
    return np.column_stack(((cols + .5) / width * 2 * radius - radius,
                            (height - rows - .5) / height * 2 * radius - radius,
                            rcs.ravel()[indices], doppler.ravel()[indices]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=list(range(100, 2001, 100)))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--limit', type=int, default=50, help='Pilot frame limit; scaling always uses all SS50')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50 or min(args.budgets) <= 0 or args.workers < 1 or args.threads < 1:
        parser.error('Require 1..50 frames and positive budgets')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = read_rows(DINO)
    radar = read_rows(RADARGEN)
    ids = lambda rows: [(r['scene'], r['frame']) for r in rows]
    if len(baseline) != 50 or ids(baseline) != ids(radar):
        raise ValueError('Baseline SS50 identities/order differ')
    provenance = json.loads((DINO / 'validation_config.json').read_text())
    with h5py.File(ARCHIVE / 'predictions.h5', 'r') as archive:
        if not archive.attrs['complete']:
            raise ValueError('Incomplete field archive')
        archived_provenance = json.loads(archive['provenance_json'].asstr()[()])
        if archived_provenance['checkpoint_path'] != provenance['checkpoint_path']:
            raise ValueError('Checkpoint mismatch')
        norm = NormalizationConfig(**json.loads(archive['normalization_json'].asstr()[()]))
        radius = float(archive.attrs['coordinate_range'])
        frames = []
        for row in baseline:
            group = archive[f"scenes/{row['scene']}/frames/{row['frame']:06d}"]
            if not group.attrs['complete']:
                raise ValueError('Incomplete frame')
            metadata = json.loads(group.attrs['metadata_json'])
            mass, rcs, doppler = [group[f'fields/{k}'][()] for k in ('cell_mass_grid', 'rcs', 'doppler')]
            if not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
                raise ValueError('Invalid cell mass')
            source = str(group.attrs['source'])
            if source != row['source']:
                raise ValueError('Source mismatch')
            with h5py.File(source, 'r') as target:
                gt = np.column_stack([target[f'ppp_targets/{k}'][()] for k in ('xy', 'rcs', 'doppler')])
            frames.append(dict(mass=mass, rcs=rcs, doppler=doppler, gt=gt, metadata=metadata))
    mean_mass = float(np.mean([f['mass'].sum() for f in frames]))
    # Load annotations only; no network, model, conditioning maps, or GPU.
    from truckscenes import TruckScenes
    from radargen.datasets.truckscenes import TruckScenesAdapter
    from radargen.datasets.truckscenes.config import TRUCKSCENES_REFERENCE_SENSOR
    from radargen.core.protocols import DatasetConfig
    config = yaml.safe_load(Path(provenance['dataset_config']).read_text())['data']
    database = TruckScenes(config['extra']['dataset_version'], config['dataset_dir'], verbose=False)
    adapter = TruckScenesAdapter(database, config=DatasetConfig(
        name='truckscenes', camera_views=config['extra']['camera_views'],
        reference_sensor=TRUCKSCENES_REFERENCE_SENSOR, normalization=norm,
        data_root=database.dataroot, version=database.version))
    for frame in frames[:args.limit]:
        frame['boxes'] = validation_boxes(SimpleNamespace(adapter=adapter), frame['metadata'])
    device = torch.device('cpu')
    mmd = MMDLoss(kernel=RBF(device=device), device=device)
    manifest = dict(checkpoint=provenance['checkpoint_path'], archive=str(ARCHIVE),
        mean_original_expected_count=mean_mass, budgets=args.budgets, seeds=args.seeds,
        frames=args.limit, scaling_frames=50, threads=args.threads, workers=args.workers, device='cpu',
        ss50_identities=ids(baseline), baseline_dino=str(DINO), baseline_radargen=str(RADARGEN))
    (args.output / 'experiment.json').write_text(json.dumps(manifest, indent=2))
    for name, source in [('dino', DINO), ('radargen', RADARGEN)]:
        (args.output / f'baseline_{name}.json').write_text((source / 'summary.json').read_text())
    setup_seconds = time.perf_counter() - started
    def run_one(task):
        budget, seed = task
        directory = args.output / f'budget_{budget:04d}_seed_{seed}'
        directory.mkdir(exist_ok=True)
        if (directory / 'summary.json').exists():
            return None
        begin = time.perf_counter()
        rows = []
        scale = budget / mean_mass
        with (directory / 'metrics.jsonl').open('w') as stream:
            for index, frame in enumerate(frames[:args.limit]):
                tick = time.perf_counter()
                points = sample_scaled(frame['mass'], frame['rcs'], frame['doppler'], radius,
                    scale, np.random.default_rng(np.random.SeedSequence([seed, budget, index])))
                sampling_seconds = time.perf_counter() - tick
                gt, boxes = frame['gt'], frame['boxes']
                metrics = sample_metrics(points, gt, norm, mmd, device)
                # Field-based mark errors and collisions stay unchanged.
                metrics.update({k: v for k, v in baseline[index]['metrics'].items()
                                if '_mae_at_' in k or k == 'target_collision_count'})
                metrics['sampling_time_seconds'] = sampling_seconds
                foreground, clouds = foreground_metrics(points, gt, boxes, norm, device, include_points=True)
                mass = frame['mass'] * scale
                analytic = compute_ppp_count_metrics(mass, gt, boxes, make_ppp_grid_xy(*mass.shape, radius))
                metrics.update({k: v for k, v in analytic.items() if not isinstance(v, list)})
                foreground.update({k: v for k, v in analytic.items() if isinstance(v, list)})
                row = dict(index=index, scene=baseline[index]['scene'], frame=baseline[index]['frame'],
                    source=baseline[index]['source'], status='ok' if len(points) else 'empty_cloud',
                    plot='', metrics=metrics, foreground=foreground, foreground_points=clouds,
                    foreground_box_count=len(boxes))
                row = json_values(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
        write_reports(directory, rows, dict(provenance, sample_count=args.limit), device)
        elapsed = time.perf_counter() - begin
        print(f'budget={budget} seed={seed}: {elapsed:.2f}s ({elapsed/args.limit:.3f}s/frame)', flush=True)
        return dict(budget=budget, seed=seed, seconds=elapsed, frames=args.limit,
                    seconds_per_frame=elapsed / args.limit)

    from concurrent.futures import ThreadPoolExecutor
    timings = []
    tasks = [(budget, seed) for budget in args.budgets for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for timing in executor.map(run_one, tasks):
            if timing is not None:
                timings.append(timing)
            (args.output / 'timing.json').write_text(json.dumps(dict(
                setup_seconds=setup_seconds, total_seconds=time.perf_counter()-started,
                evaluations=timings), indent=2))
    plot_results(args.output)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = []
    for path in sorted(output.glob('budget_*/summary.json')):
        budget = int(path.parent.name.split('_')[1])
        results.append((budget, json.loads(path.read_text())))
    if not results:
        return
    baselines = {name: json.loads((output / f'baseline_{name}.json').read_text())
                 for name in ('dino', 'radargen')}
    manifest = json.loads((output / 'experiment.json').read_text())
    for scope in ('metrics', 'foreground'):
        keys = sorted({k for _, s in results for k, v in s[scope].items()
                       if isinstance(v, (int, float, dict)) and k != 'inference_time_seconds'})
        fig, axes = plt.subplots(int(np.ceil(len(keys)/4)), 4, figsize=(20, 3.4*int(np.ceil(len(keys)/4))), squeeze=False)
        def value(summary, key):
            v = summary[scope].get(key)
            return v.get('mean') if isinstance(v, dict) else v
        for ax, key in zip(axes.flat, keys):
            budgets = sorted({b for b, _ in results})
            means, stds = [], []
            for budget in budgets:
                values = [value(s, key) for b, s in results if b == budget and value(s, key) is not None]
                means.append(np.mean(values) if values else np.nan)
                stds.append(np.std(values) if values else np.nan)
            ax.errorbar(budgets, means, yerr=stds, label='Scaled DINO PPP; seed mean ± SD')
            gt = baselines['dino']['metrics']['gt_count']['mean']
            ax.axvline(gt, color='gray', linestyle=':', label=f'GT mean ({gt:.0f})')
            for name, marker in [('dino', 'o'), ('radargen', 's')]:
                y = value(baselines[name], key)
                if y is not None:
                    x = baselines[name]['metrics']['generated_count']['mean']
                    ax.scatter([x], [y], marker=marker, s=45, label=f'{name} original ({x:.0f})', zorder=5)
            ax.set_title(key, fontsize=9)
            ax.set_xlabel('Expected mean points/frame')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f'SS50 DINO step 19914 — {manifest["frames"]} frames; shared intensity scaling')
        fig.tight_layout(rect=(0, .035, 1, .98))
        fig.text(.02, .01, 'Original counts (mean ± frame SD): ' + '; '.join(
            f'{name}: {s["metrics"]["generated_count"]["mean"]:.1f} ± {s["metrics"]["generated_count"]["std"]:.1f}'
            for name, s in baselines.items()) + f'; GT: {gt:.1f}')
        fig.savefig(output / f'{scope}.png', dpi=140)
        fig.savefig(output / f'{scope}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    import resource
    import sys
    start = time.perf_counter()
    success = False
    try:
        main()
        success = True
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        stats = dict(success=success, wall_seconds=time.perf_counter()-start,
                     user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                     max_rss_kib=usage.ru_maxrss)
        if '--output' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--output')+1])
            if directory.is_dir():
                (directory / 'resource_usage.json').write_text(json.dumps(stats, indent=2))
        print('Process resource usage: ' + json.dumps(stats), flush=True)
