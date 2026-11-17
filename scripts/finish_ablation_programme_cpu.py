#!/usr/bin/env python3
"""Entry-node CPU coordinator: count sweeps and standard-scene videos, no GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from scripts.render_ablation_standard_scenes import probe
from scripts.run_ablation_checkpoint_programme import atomic_json


def write_index(manifest, entries, states):
    lines = ['# All ablations: first- and second-final evaluation index', '',
             'Both fixed training-budget checkpoints are included, regardless of which has better SS50. '
             'GPU jobs: one per architecture, develbooster, four GPUs, two-hour limit. '
             'ResNet waits for successful training completion. CPU count sweeps and video rendering '
             'run on the entry node; no additional GPU submissions.', '',
             '| Checkpoint | Step | SS50 | Full validation / PNGs / predictions | VoxelNeXt | Count sweep | Videos |',
             '|---|---:|---|---|---|---|---|']
    resolved_entries = []
    for entry in entries:
        path = Path(entry['bundle_root']) / 'resolved_checkpoint.json'
        resolved = json.loads(path.read_text()) if path.exists() else entry
        resolved_entries.append(resolved)
        full = Path(entry['bundle_root']) / 'metrics'
        small = Path(entry['ss50_output'])
        detector = Path(resolved.get('cached_detector_metrics', ''))
        if not detector.is_file():
            candidates = sorted(Path(entry['detector_root']).glob('attempt_*/metrics.json'))
            detector = candidates[-1] if candidates else None
        count = Path('output/paper_material/count_sweep_evaluation') / f"ss50_{entry['count_slug']}_count_sweep"
        def link(path, label):
            return f'[{label}]({path.resolve()})' if path.is_file() else 'pending'
        cells = [entry['label'], str(resolved.get('step') or 'after training'),
                 link(small / 'summary.json', 'Summary'), link(full / 'summary.json', 'Summary'),
                 link(detector, 'Metrics') if detector else 'pending',
                 link(count / 'results.md', 'Results'), states[entry['key']]['videos']]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Full prediction archives are `metrics/predictions.h5`; PNGs are below `metrics/plots/`. '
              'A partial full-validation summary can exist before completion; inspect its complete flag. '
              'Count sweeps use targets 100–2000 in steps of 100, three seeds (42/43/44), and fixed maps. '
              'VoxelNeXt uses the frozen epoch-400 detector, z=1 flat-up, physical marks/absolute Doppler '
              'and the matched three-class scoring protocol.', '', '## Standard-scene videos', '',
              'Each video: GT · selected ablation checkpoint · DINO 15k · RadarGen; Doppler, RCS '
              'and heatmap. Ten scenes per checkpoint, 80 videos overall; prior verified videos reused.', '']
    scenes = Path(manifest['scene_list']).read_text().split()
    for entry in resolved_entries:
        lines += [f"### {entry['label']}", '']
        if 'model_id' not in entry:
            lines += ['Waiting for checkpoint resolution after training.', '']
            continue
        for scene in scenes:
            video = Path(manifest['scene_root']) / scene / f"video_{entry['model_id']}_dino_2773475_step15000_radargen/comparison.mp4"
            lines.append(f'- [{scene}]({video})' if video.exists() else f'- {scene}: pending')
        lines.append('')
    Path('docs/ablations/all_checkpoint_evaluations_index.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    work = Path(manifest['cpu_root'])
    work.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    entries = [entry for model in manifest['models'] for entry in model['checkpoints']]
    states = {e['key']: dict(count_sweep='waiting', videos='waiting') for e in entries}
    cores = sorted(os.sched_getaffinity(0))
    groups = queue.Queue()
    for i in range(min(4, max(1, len(cores) // 2))):
        groups.put(cores[2 * i:2 * i + 2])
    futures = {}

    def counts(entry):
        slug = entry['count_slug']
        output = Path('output/paper_material/count_sweep_evaluation') / f'ss50_{slug}_count_sweep'
        summaries = list(output.glob('budget_*/summary.json'))
        complete = (len(summaries) == 60 and (output / 'native/summary.json').exists()
                    and (output / 'results.csv').exists() and (output / 'plots/README.md').exists())
        if complete and all(json.loads(p.read_text()).get('complete') for p in summaries):
            return
        spec = work / f"{entry['key']}_count_spec.json"
        spec.write_text(json.dumps([[slug, entry['label'], Path(entry['bundle_root']).name, entry['step']]]) + '\n')
        with (work / f"{entry['key']}_counts.log").open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.ss50_ablation_count_sweeps',
                            '--workers', '24', '--specs-json', str(spec), '--usage-output',
                            str(work / f"{entry['key']}_count_resources.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def videos(entry):
        chosen = groups.get()
        try:
            for scene in Path(manifest['scene_list']).read_text().split():
                root = Path(manifest['scene_root']) / scene
                output = root / f"video_{entry['model_id']}_dino_2773475_step15000_radargen"
                video = output / 'comparison.mp4'
                if video.exists():
                    try:
                        probe(video)
                        continue
                    except (ValueError, KeyError, subprocess.CalledProcessError):
                        pass
                with (work / f"{entry['key']}_{scene}_video.log").open('w') as log:
                    subprocess.run(['taskset', '-c', ','.join(map(str, chosen)), sys.executable,
                                    '-u', 'evaluation/scripts/scene_comparison.py', 'render',
                                    '--scene-dir', str(root), '--models', entry['model_id'],
                                    'dino_2773475_step15000', 'radargen', '--output-dir', str(output),
                                    '--marks', 'doppler', 'rcs', 'heatmap', '--formats', 'mp4',
                                    '--fps', '10', '--skip-overlays'],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                probe(video)
        finally:
            groups.put(chosen)

    def ready(entry, kind):
        if kind == 'count_sweep':
            summary = Path(entry['bundle_root']) / 'metrics/summary.json'
            if not summary.exists() or not json.loads(summary.read_text()).get('complete', False):
                return False
            try:
                import h5py
                with h5py.File(Path(entry['bundle_root']) / 'metrics/predictions.h5') as handle:
                    return bool(handle.attrs.get('complete'))
            except OSError:
                return False
        return all((Path(manifest['scene_root']) / scene / 'predictions' /
                    f"{entry['model_id']}{suffix}.h5").is_file()
                   for scene in Path(manifest['scene_list']).read_text().split()
                   for suffix in ('', '_dense'))

    with ThreadPoolExecutor(max_workers=1) as count_pool, ThreadPoolExecutor(max_workers=4) as video_pool:
        while True:
            for entry in entries:
                resolved = Path(entry['bundle_root']) / 'resolved_checkpoint.json'
                if not resolved.exists():
                    continue
                resolved_entry = json.loads(resolved.read_text())
                for kind, pool, function in [('count_sweep', count_pool, counts), ('videos', video_pool, videos)]:
                    key = (entry['key'], kind)
                    if states[entry['key']][kind] == 'waiting' and ready(resolved_entry, kind):
                        futures[key] = pool.submit(function, resolved_entry)
                        states[entry['key']][kind] = 'queued_or_running'
                    future = futures.get(key)
                    if future is not None and future.done() and states[entry['key']][kind] == 'queued_or_running':
                        try:
                            future.result()
                            states[entry['key']][kind] = 'complete'
                        except BaseException as error:
                            states[entry['key']][kind] = 'failed'
                            states[entry['key']][kind + '_error'] = repr(error)
                        print(entry['key'], kind, states[entry['key']][kind], flush=True)
            atomic_json(work / 'status.json', states)
            write_index(manifest, entries, states)
            terminal = [state[k] in ('complete', 'failed') for state in states.values()
                        for k in ('count_sweep', 'videos')]
            if all(terminal):
                break
            time.sleep(30)
    if any(state[k] == 'failed' for state in states.values() for k in ('count_sweep', 'videos')):
        raise SystemExit('Some CPU stages failed; see status.json')
    with (work / 'paired_ss50_comparison.log').open('w') as log:
        subprocess.run([sys.executable, '-u', '-m', 'scripts.compare_ablation_first_second_ss50'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
