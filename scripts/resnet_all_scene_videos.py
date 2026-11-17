#!/usr/bin/env python3
"""Resume missing all-scene predictions on GPUs and comparison videos on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import yaml

from scripts.run_ablation_checkpoint_programme import atomic_json
from scripts.run_ablation_full_validation_scenes import bind_worker_gpu

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons')
WORK = Path('output/ablations/resnet_all_scenes_final_dino')
MODELS = ['resnet50_2882612_step25030', 'dino_2773475', 'radargen']
LABELS = ['C3P-dar-RN50', 'C3P-dar-Dv3 (final, step 19914)', 'RadarGen']
SUFFIX = 'video_resnet50_2882612_step25030_dino_2773475_final_radargen'


def scenes():
    manifest = json.loads(Path('evaluation/configs/FULL_ablation_2867524_bundle.json').read_text())
    data = json.loads(Path(manifest['manifest']).read_text())
    tokens = sorted(s['scene_token'] for s in data['body']['scenes'])
    if len(tokens) != 49 or len(set(tokens)) != 49:
        raise ValueError('Expected all 49 validation scenes')
    return tokens


def valid(scene, model):
    path = ROOT / scene / 'predictions' / f'{model}.h5'
    if not path.exists():
        return False
    with h5py.File(path) as f, h5py.File(ROOT / scene / 'ground_truth.h5') as gt:
        p = json.loads(f.attrs['provenance'])
        common = json.loads(gt.attrs['provenance'])
        expected = json.loads((WORK / 'plan.json').read_text())['models'][model]
        if p['checkpoint'] != expected['checkpoint']:
            raise ValueError(f'Wrong checkpoint: {path}')
        return bool(f.attrs.get('complete')) and sorted(f['frames']) == sorted(gt['frames']) and all(
            p[k] == common[k] for k in ('scene', 'frames', 'dataset_contract', 'manifest_identity'))


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    plan = dict(scenes=scenes(), models={}, frame_count=0)
    for model in MODELS:
        representative = next(ROOT.glob(f'*/predictions/{model}.h5'))
        with h5py.File(representative) as f:
            p = json.loads(f.attrs['provenance'])
        config = WORK / f'{model}.yaml'
        config.write_text(yaml.safe_dump(p['validation_config'], sort_keys=False))
        plan['models'][model] = dict(checkpoint=p['checkpoint'], label=p['label'],
                                     validation_config=str(config.resolve()))
    for scene in plan['scenes']:
        with h5py.File(ROOT / scene / 'ground_truth.h5') as f:
            plan['frame_count'] += len(f['frames'])
    atomic_json(WORK / 'plan.json', plan)
    plan['cached_scenes'] = {m: sum(valid(s, m) for s in plan['scenes']) for m in MODELS}
    atomic_json(WORK / 'plan.json', plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'models'}, indent=2))


def predict():
    rank = int(os.environ['SLURM_PROCID'])
    bind_worker_gpu(int(os.environ.get('SLURM_LOCALID', rank)))
    plan = json.loads((WORK / 'plan.json').read_text())
    directory = WORK / f'worker_{rank}'
    directory.mkdir(exist_ok=True)
    state = dict(rank=rank, completed=False, scenes=[])
    try:
        for scene in plan['scenes'][rank::4]:
            for model in MODELS:
                if valid(scene, model):
                    continue
                config = dict(scene=scene, dataset_config='configs/PPP_train_neck.yaml',
                              output_root=str(ROOT), seed=42, device='cuda', fps=10,
                              models={model: {k:plan['models'][model][k]
                                              for k in ('label', 'validation_config')}})
                path = directory / f'{scene}_{model}.yaml'
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                with (ROOT / scene / '.prediction_export.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    with (directory / f'{scene}_{model}.log').open('w') as log:
                        subprocess.run([sys.executable, '-u', 'evaluation/scripts/scene_comparison.py',
                                        'predict', '--config', str(path)], stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                if not valid(scene, model):
                    raise ValueError(f'Incomplete export: {scene}, {model}')
            state['scenes'].append(scene)
            atomic_json(directory / 'status.json', state)
        state['completed'] = True
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        atomic_json(directory / 'status.json', state)


def probe(scene):
    output = ROOT / scene / SUFFIX
    config = json.loads((output / 'render_config.json').read_text())
    with h5py.File(ROOT / scene / 'ground_truth.h5') as f:
        frames = len(f['frames'])
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate:format=duration', '-of', 'json',
        str(output / 'comparison.mp4')]))
    stream = data['streams'][0]
    if config['models'] != MODELS or int(stream['nb_frames']) != frames or stream['r_frame_rate'] != '10/1':
        raise ValueError(f'Incorrect video: {scene}')
    if abs(float(data['format']['duration']) - frames / 10) > .02:
        raise ValueError(f'Incorrect duration: {scene}')
    return dict(frames=frames, seconds=frames / 10)


def render():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tokens = scenes()
    states = {s: dict(state='waiting', output=str(ROOT / s / SUFFIX / 'comparison.mp4')) for s in tokens}
    cores = sorted(os.sched_getaffinity(0))
    workers = min(4, max(1, len(cores) // 2))
    futures = {}

    def one(scene):
        try:
            return probe(scene)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            pass
        output = ROOT / scene / SUFFIX
        with (WORK / f'{scene}_render.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'evaluation/scripts/scene_comparison.py', 'render',
                '--scene-dir', str(ROOT / scene), '--models', *MODELS, '--model-labels', *LABELS,
                '--output-dir', str(output), '--marks', 'doppler', 'rcs', 'heatmap',
                '--formats', 'mp4', '--fps', '10', '--skip-overlays'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        return probe(scene)

    deadline = time.monotonic() + 8 * 3600
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            for scene in tokens:
                if states[scene]['state'] == 'waiting' and all(valid(scene, m) for m in MODELS):
                    futures[scene] = pool.submit(one, scene)
                    states[scene]['state'] = 'queued_or_running'
                future = futures.get(scene)
                if future and future.done() and states[scene]['state'] == 'queued_or_running':
                    try:
                        states[scene].update(state='complete', **future.result())
                    except BaseException as error:
                        states[scene].update(state='failed', error=repr(error))
                    print(scene, states[scene]['state'], flush=True)
            atomic_json(WORK / 'video_status.json', states)
            lines = ['# All-scene ResNet comparisons', '',
                     '49 validation scenes. Columns: GT · C3P-dar-RN50 (2882612 / 25,030) · '
                     'final C3P-dar-Dv3 (2773475 / 19,914) · RadarGen. '
                     'Rows: Doppler, RCS, heatmap; 10 fps. Predictions are reused without resampling.', '',
                     '| Scene | Status | Video |', '|---|---|---|']
            for scene, state in states.items():
                link = f"[MP4]({state['output']})" if state['state'] == 'complete' else 'pending'
                lines.append(f"| {scene} | {state['state']} | {link} |")
            Path('docs/ablations/resnet_all_scene_videos.md').write_text('\n'.join(lines) + '\n')
            if all(s['state'] in ('complete', 'failed') for s in states.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Missing scene predictions; inspect GPU worker logs and restart renderer')
            time.sleep(10)
    if any(s['state'] != 'complete' for s in states.values()):
        raise RuntimeError('Some videos failed; inspect video_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'predict', 'render'])
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'render': render}[args.mode]()
