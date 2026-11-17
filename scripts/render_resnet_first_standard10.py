#!/usr/bin/env python3
"""Render cached first-final ResNet comparisons and copy ten verified MP4s locally."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py

from scripts.run_ablation_checkpoint_programme import atomic_json

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons')
WORK = Path('output/ablations/resnet_first_standard10')
DEST = Path('output/resnet_first_checkpoint_comparisons_10')
MODELS = ['resnet50_2856174_step12575', 'dino_2773475', 'radargen']
LABELS = ['C3P-dar-RN50 (first, step 12575)', 'C3P-dar-Dv3 (final, step 19914)', 'RadarGen']
SUFFIX = 'video_resnet50_2856174_step12575_dino_2773475_final_radargen'


def probe(path, frames):
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,r_frame_rate:format=duration', '-of', 'json', str(path)]))
    stream = result['streams'][0]
    if int(stream['nb_frames']) != frames or stream['r_frame_rate'] != '10/1':
        raise ValueError(f'Incorrect frames/FPS: {path}')
    if abs(float(result['format']['duration']) - frames / 10) > .02:
        raise ValueError(f'Incorrect duration: {path}')
    return result


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    tokens = Path('jsc_jupiter/scene_extra10_step15000_tokens.txt').read_text().split()
    if len(tokens) != 10 or len(set(tokens)) != 10:
        raise ValueError('Expected ten distinct standard scenes')
    states = {s: dict(state='pending') for s in tokens}
    for scene in tokens:
        with h5py.File(ROOT / scene / 'ground_truth.h5') as gt:
            keys = sorted(gt['frames'])
            common = json.loads(gt.attrs['provenance'])
            for model in MODELS:
                with h5py.File(ROOT / scene / 'predictions' / f'{model}.h5') as f:
                    p = json.loads(f.attrs['provenance'])
                    if not f.attrs.get('complete') or sorted(f['frames']) != keys or any(
                            p[k] != common[k] for k in ('scene', 'frames', 'dataset_contract', 'manifest_identity')):
                        raise ValueError(f'Incompatible exports: {scene}, {model}')
                    expected = 'step_12575.pth' if model == MODELS[0] else 'step_19914.pth' if model == MODELS[1] else '.safetensors'
                    if not p['checkpoint']['path'].endswith(expected):
                        raise ValueError(f'Wrong checkpoint: {scene}, {model}')
        states[scene]['frames'] = len(keys)
    atomic_json(WORK / 'status.json', states)

    def one(scene):
        directory = ROOT / scene / SUFFIX
        source = directory / 'comparison.mp4'
        config = directory / 'render_config.json'
        reuse = False
        if source.exists() and config.exists():
            try:
                saved = json.loads(config.read_text())
                reuse = saved['models'] == MODELS and saved.get('model_labels') == LABELS
                if reuse:
                    probe(source, states[scene]['frames'])
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
                reuse = False
        if not reuse:
            with (WORK / f'{scene}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', 'evaluation/scripts/scene_comparison.py', 'render',
                    '--scene-dir', str(ROOT / scene), '--models', *MODELS, '--model-labels', *LABELS,
                    '--output-dir', str(directory), '--marks', 'doppler', 'rcs', 'heatmap',
                    '--formats', 'mp4', '--fps', '10', '--skip-overlays'],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        verified = probe(source, states[scene]['frames'])
        target = DEST / f'{scene}.mp4'
        temporary = target.with_suffix('.mp4.tmp')
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise ValueError(f'Copy size mismatch: {scene}')
        temporary.replace(target)
        return dict(state='complete', frames=states[scene]['frames'], source=str(source),
                    copy=str(target.resolve()), bytes=target.stat().st_size, probe=verified)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, scene): scene for scene in tokens}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                states[scene] = future.result()
            except BaseException as error:
                states[scene].update(state='failed', error=repr(error))
            atomic_json(WORK / 'status.json', states)
            atomic_json(DEST / 'manifest.json', states)
            print(scene, states[scene]['state'], flush=True)
    if any(s['state'] != 'complete' for s in states.values()):
        raise RuntimeError('Some videos failed; inspect status.json and per-scene logs')
    print(f'All ten videos verified and copied to {DEST.resolve()}', flush=True)


if __name__ == '__main__':
    main()
