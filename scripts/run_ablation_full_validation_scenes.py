#!/usr/bin/env python3
"""Four independent GPU workers: full annotated validation + three scene shards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def bind_worker_gpu(local_rank):
    """Select one GPU explicitly: this cluster can expose all GPUs to each task."""
    allocated = os.environ.get('SLURM_STEP_GPUS', '')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [v.strip() for v in (allocated or visible).split(',') if v.strip()]
    if len(devices) > 1:
        if local_rank >= len(devices):
            raise ValueError('Local rank exceeds allocated GPU count')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
    elif len(devices) == 1:
        # Respect an already isolated task device.
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
    else:
        raise ValueError('No explicit allocated GPU inventory for bundle worker')


def write_comparison(spec):
    root = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108')
    entries = [(spec['label'], Path(spec['validation_output'])),
               ('Final DINO 19,914', root / 'dino_2773475'),
               ('Best-F1 DINO 15,000 (SS50-selected)', root / 'dino_2773475_step15000'),
               ('RadarGen', root / 'radargen')]
    summaries, identities = [], []
    for _, directory in entries:
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['complete'] or summary['evaluated_samples'] != spec['keyframes']:
            raise ValueError(f'Incomplete full validation: {directory}')
        summaries.append(summary)
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        identities.append([(r['scene'], r['frame'], r['seed']) for r in rows])
    if not all(rows == identities[0] for rows in identities):
        raise ValueError('Full comparison samples/order/seeds differ')
    higher = {'distance_attr_f1', 'distance_attr_precision', 'distance_attr_recall',
              'iou_th1', 'per_box_num_points_similarity', 'ppp_hit_probability_gt_positive', 'hit_rate'}
    neutral = {'generated_count', 'gt_count', 'inference_time_seconds', 'ppp_expected_total_count',
               'target_collision_count', 'ppp_expected_count', 'gt_positive_boxes', 'evaluated_boxes'}
    lines = [f"# Full validation: {spec['label']}", '',
             'Identical annotated-frame identities, order and seeds verified. Best unrounded means are bold; '
             'descriptive counts and hardware-dependent timings are not ranked. The 15,000-step DINO '
             'reference was selected for best SS50 F1, not necessarily best full-validation F1.', '']
    for section in ('metrics', 'foreground'):
        lines += [f'## {section}', '', '| Metric | ' + ' | '.join(n for n, _ in entries) + ' |',
                  '|---|---:|---:|---:|---:|']
        for key in dict.fromkeys(k for s in summaries for k in s[section]):
            if key == 'mmd_sampling':
                continue
            values = [s[section].get(key) for s in summaries]
            means = [v.get('mean') if isinstance(v, dict) else v for v in values]
            available = [v for v in means if isinstance(v, (int, float))]
            best = ((max if key in higher else min)(available)
                    if available and key not in neutral else None)
            cells = []
            for value in means:
                cell = '—' if value is None else (str(value) if isinstance(value, int) else f'{value:.4f}')
                cells.append(f'**{cell}**' if best is not None and value == best else cell)
            lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['Legacy RadarGen target-cell MAEs differ from detection-based mark MAEs; foreground geometry '
              'denominators can differ. Training histories and compute differ. Timing is not hardware-controlled.',
              '', 'Sources:', '']
    lines += [f'- [{name}]({directory}/summary.json).' for name, directory in entries]
    report = Path(spec['work_dir']) / 'comparison.md'
    report.write_text('\n'.join(lines) + '\n')
    destination = Path(f"docs/ablations/FULL_{spec['model_id']}_comparison.md")
    destination.write_text(report.read_text())


def run_worker(spec, rank):
    import yaml

    work = Path(spec['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    status = work / f'worker_{rank}_status.json'
    record = dict(rank=rank, completed=False, scenes=[],
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))

    def save():
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status)

    save()
    try:
        if rank == 0:
            output = Path(spec['validation_output'])
            args = [sys.executable, '-u', 'evaluation/scripts/validate.py',
                    '--config', spec['validation_config']]
            if (output / 'validation_config.json').exists():
                args.append('--resume')
            subprocess.run(args, check=True)
            summary = json.loads((output / 'summary.json').read_text())
            assert summary['complete'] and summary['evaluated_samples'] == spec['keyframes']
            assert len(list((output / 'plots').rglob('*.png'))) == spec['keyframes']
            record['validation_output'] = str(output)
            write_comparison(spec)
        else:
            manifest = json.loads(Path(spec['manifest']).read_text())
            scenes = sorted(s['scene_token'] for s in manifest['body']['scenes'])
            if spec.get('reverse_scene_order', False):
                scenes.reverse()
            assigned = scenes[rank - 1::3]
            config_dir = work / f'worker_{rank}_configs'
            config_dir.mkdir(exist_ok=True)
            for scene in assigned:
                config = dict(scene=scene, dataset_config=spec['dataset_config'],
                              output_root=spec['scene_output_root'], seed=42,
                              device='cuda', fps=10, models={spec['model_id']: dict(
                                  label=spec['label'], validation_config=spec['validation_config'])})
                path = config_dir / f'{scene}.yaml'
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                # Exports are durable/resumable, with strict provenance checks.
                # Separate processes release model/dataset memory between scenes.
                scene_dir = Path(spec['scene_output_root']) / scene
                scene_dir.mkdir(parents=True, exist_ok=True)
                # Separate model jobs share GT exports. Serialize a scene's
                # export writers to avoid racing on ground_truth.partial.h5.
                with (scene_dir / '.prediction_export.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    subprocess.run([sys.executable, '-u', 'evaluation/scripts/scene_comparison.py',
                                    'predict', '--config', str(path)], check=True)
                record['scenes'].append(scene)
                save()
            record['assigned_scenes'] = assigned
        record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--rank', type=int, default=None)
    args = parser.parse_args()
    rank = args.rank if args.rank is not None else int(os.environ['SLURM_PROCID'])
    if rank not in range(4):
        parser.error('Expected four ranks, 0 through 3')
    bind_worker_gpu(int(os.environ.get('SLURM_LOCALID', rank)))
    run_worker(json.loads(Path(args.spec).read_text()), rank)


if __name__ == '__main__':
    # Direct execution must resolve project imports as well as scripts/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
