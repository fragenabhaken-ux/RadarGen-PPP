#!/usr/bin/env python3
"""One four-GPU job per architecture, covering first/second final checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def resolve(entry):
    """Resolve a dynamic final checkpoint only after its training dependency."""
    if entry.get('checkpoint'):
        checkpoint = Path(entry['checkpoint'])
        step = entry['step']
    else:
        candidates = []
        for path in (Path(entry['train_root']) / 'checkpoints').glob('epoch_*_step_*.pth'):
            match = re.search(r'_step_(\d+)\.pth$', path.name)
            if match:
                candidates.append((int(match[1]), path))
        if not candidates:
            raise ValueError('Final checkpoint unavailable')
        step, checkpoint = max(candidates)
        if step <= entry['minimum_step']:
            raise ValueError('Second-run final checkpoint must be newer than first-final')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = dict(entry, checkpoint=str(checkpoint), step=step,
                  model_id=entry['model_id_template'].format(step=step))
    atomic_json(Path(entry['bundle_root']) / 'resolved_checkpoint.json', result)
    return result


def config_for(entry, ss50=False):
    import yaml
    config = yaml.safe_load((REPO / entry['template_config']).read_text())
    config.update(dataset_config=str(Path(entry['train_root']) / 'submitted_config.yaml'),
                  model_config=str(Path(entry['train_root']) / 'submitted_config.yaml'),
                  checkpoint_path=entry['checkpoint'], tensorboard=False,
                  max_samples=50 if ss50 else None, save_plots=True,
                  save_predictions=True, plot_dpi=110, plot_gamma=0.2,
                  output_dir=entry['ss50_output'] if ss50 else str(Path(entry['bundle_root']) / 'metrics'))
    config.pop('reference_validation_records', None)
    if ss50:
        config['reference_validation_records'] = (
            '/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/'
            '2455164_PPP_validation_dinov3_2438929/metrics.jsonl')
    return config


def validation(entry, ss50=False):
    import yaml
    config = config_for(entry, ss50)
    output = Path(config['output_dir'])
    expected = 50 if ss50 else 1906
    summary = output / 'summary.json'
    if summary.exists():
        saved = json.loads(summary.read_text())
        if saved.get('complete') and saved.get('evaluated_samples') == expected:
            if Path(saved['checkpoint']).resolve() != Path(entry['checkpoint']).resolve():
                raise ValueError(f'Checkpoint mismatch: {summary}')
            return
    path = Path(entry['bundle_root']) / 'execution' / ('ss50.yaml' if ss50 else 'full.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, '-u', 'evaluation/scripts/validate.py', '--config', str(path)]
    if (output / 'validation_config.json').exists():
        command.append('--resume')
    subprocess.run(command, cwd=REPO, check=True)
    saved = json.loads(summary.read_text())
    if not saved['complete'] or saved['evaluated_samples'] != expected:
        raise ValueError(f'Incomplete validation: {output}')


def detector(entry):
    cached = entry.get('cached_detector_metrics')
    if cached and Path(cached).exists():
        saved = json.loads(Path(cached).read_text())
        if saved.get('full_archive') and saved.get('sample_count') == 1906:
            return str(cached)
    root = Path(entry['detector_root'])
    root.mkdir(parents=True, exist_ok=True)
    # Preserve incomplete attempts rather than overwriting an HDF5 writer.
    attempts = sorted(root.glob('attempt_*'))
    for attempt in attempts:
        metrics = attempt / 'metrics.json'
        if metrics.exists():
            saved = json.loads(metrics.read_text())
            if saved.get('full_archive') and saved.get('sample_count') == 1906:
                return str(metrics)
    attempt = root / f'attempt_{len(attempts):03d}'
    attempt.mkdir()
    archive = Path(entry['bundle_root']) / 'metrics' / 'predictions.h5'
    command = '''set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen/radar_detection
source jsc_jupiter/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m radar_eval.run --archive "$1" --data-root /e/scratch/nxtaim-1/public/truckscenes/man-truckscenes --side generated --height 1 --height-frame flat-up --seed 42 --start 0 --max-frames 1906 --output "$2/detections.h5"
python -m radar_eval.score --archive "$1" --detections "$2/detections.h5" --data-root /e/scratch/nxtaim-1/public/truckscenes/man-truckscenes --output "$2/metrics.json"
'''
    subprocess.run(['bash', '-c', command, '--', str(archive), str(attempt)], check=True)
    return str(attempt / 'metrics.json')


def scenes(entry, master):
    import yaml
    tokens = Path(master['scene_list']).read_text().split()
    if len(tokens) != 10:
        raise ValueError('Expected the ten standard video scenes')
    config_dir = Path(entry['bundle_root']) / 'execution' / 'scene_configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    scene_validation = config_dir / 'model.yaml'
    scene_validation.write_text(yaml.safe_dump(config_for(entry), sort_keys=False))
    for scene in tokens:
        # Existing exports encode validation-config provenance. Reuse complete
        # point/dense exports directly without calling a differently named config.
        root = Path(master['scene_root']) / scene
        files = [root / 'predictions' / f"{entry['model_id']}{suffix}.h5" for suffix in ('', '_dense')]
        if all(path.is_file() for path in files):
            import h5py
            valid = True
            for path in files:
                with h5py.File(path) as handle:
                    provenance = json.loads(handle.attrs['provenance'])
                    valid &= bool(handle.attrs.get('complete'))
                    valid &= provenance['checkpoint']['path'] == str(Path(entry['checkpoint']).resolve())
            if valid:
                continue
        config = dict(scene=scene, dataset_config='configs/PPP_train_neck.yaml',
                      output_root=master['scene_root'], seed=42, device='cuda', fps=10,
                      models={entry['model_id']: dict(label=entry['label'],
                              validation_config=str(scene_validation))})
        path = config_dir / f'{scene}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.prediction_export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([sys.executable, '-u', 'evaluation/scripts/scene_comparison.py',
                            'predict', '--config', str(path)], cwd=REPO, check=True)


def run(master, model_index, rank):
    from scripts.run_ablation_full_validation_scenes import bind_worker_gpu, write_comparison
    bind_worker_gpu(int(os.environ.get('SLURM_LOCALID', rank)))
    model = master['models'][model_index]
    entry = resolve(model['checkpoints'][rank % 2])
    status = Path(model['execution_root']) / f'worker_{rank}.json'
    record = dict(completed=False, rank=rank, checkpoint=entry['checkpoint'],
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    atomic_json(status, record)
    try:
        if rank < 2:
            validation(entry)
            work = Path(entry['bundle_root']) / 'execution'
            work.mkdir(parents=True, exist_ok=True)
            full_config = work / 'full_for_report.yaml'
            import yaml
            full_config.write_text(yaml.safe_dump(config_for(entry), sort_keys=False))
            write_comparison(dict(label=entry['label'], validation_output=str(Path(entry['bundle_root']) / 'metrics'),
                                  keyframes=1906, work_dir=str(work), model_id=entry['model_id']))
            # Detector is GPU work. Reuse complete compact-second results.
            record['detector_metrics'] = detector(entry)
            validation(entry, ss50=True)
        else:
            scenes(entry, master)
        record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(status, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--model-index', required=True, type=int)
    args = parser.parse_args()
    run(json.loads(Path(args.manifest).read_text()), args.model_index, int(os.environ['SLURM_PROCID']))


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()
