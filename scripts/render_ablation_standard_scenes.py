#!/usr/bin/env python3
"""Render the ten standard scenes, separately for each second-run ablation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons')
WORK = Path('output/ablations/standard10_second_run_videos')
MODELS = ('dit_2867524_step58057', 'dino_compact_2867528_step20595',
          'dino_random_2863217_step29915')


def probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,r_frame_rate:format=duration', '-of', 'json', str(path)]))
    stream = data['streams'][0]
    if stream['nb_frames'] != '194' or stream['r_frame_rate'] != '10/1':
        raise ValueError(f'Unexpected video frames/FPS: {path}')
    if abs(float(data['format']['duration']) - 19.4) > 0.01:
        raise ValueError(f'Unexpected video duration: {path}')
    return data


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    scenes = Path('jsc_jupiter/scene_extra10_step15000_tokens.txt').read_text().split()
    if len(scenes) != 10 or len(set(scenes)) != 10:
        raise ValueError('Expected the established ten distinct standard scenes')
    jobs = []
    for scene in scenes:
        for model in MODELS:
            directory = ROOT / scene / f'video_{model}_dino_2773475_step15000_radargen'
            jobs.append(dict(scene=scene, model=model, output=str(directory / 'comparison.mp4'),
                             state='pending'))
    lock = threading.Lock()

    def save():
        with lock:
            temporary = WORK / 'status.tmp'
            temporary.write_text(json.dumps(jobs, indent=2) + '\n')
            temporary.replace(WORK / 'status.json')

    cores = sorted(os.sched_getaffinity(0))
    workers = min(8, max(1, len(cores) // 2))

    def render(index):
        job = jobs[index]
        output = Path(job['output'])
        try:
            # Complete verified files are reused when this orchestration is restarted.
            if output.exists():
                try:
                    job['probe'] = probe(output)
                    job['state'] = 'complete'
                    save()
                    return
                except (ValueError, subprocess.CalledProcessError, KeyError):
                    pass
            job['state'] = 'running'
            save()
            chosen = cores[2 * (index % workers):2 * (index % workers) + 2]
            args = ['taskset', '-c', ','.join(map(str, chosen)), sys.executable, '-u',
                    'evaluation/scripts/scene_comparison.py', 'render',
                    '--scene-dir', str(ROOT / job['scene']), '--models', job['model'],
                    'dino_2773475_step15000', 'radargen', '--output-dir', str(output.parent),
                    '--marks', 'doppler', 'rcs', 'heatmap', '--formats', 'mp4',
                    '--fps', '10', '--skip-overlays']
            with (WORK / f"{job['scene']}_{job['model']}.log").open('w') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            job['probe'] = probe(output)
            job['state'] = 'complete'
        except BaseException as error:
            job['state'] = 'failed'
            job['error'] = repr(error)
        finally:
            save()
        print(job['scene'], job['model'], job['state'], flush=True)

    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(render, range(len(jobs))))
    lines = ['# Standard ten scenes: second-run ablation comparisons', '',
             'Columns: **GT · selected ablation · best-F1 DINO (15,000) · RadarGen**. '
             'Rows: Doppler, RCS and heatmap. Existing predictions reused; CPU-only entry-node rendering.', '',
             '| Scene | DiT 58,057 | Compact DINO 20,595 | Random DINO 29,915 |',
             '|---|---|---|---|']
    for scene in scenes:
        group = [j for j in jobs if j['scene'] == scene]
        cells = [f"[Video]({j['output']})" if j['state'] == 'complete' else j['state'] for j in group]
        lines.append('| ' + scene + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Each successful MP4 verified: 194 frames, 10 fps, 19.4 seconds.',
              'Logs and machine-readable state: `output/ablations/standard10_second_run_videos/`.']
    Path('docs/ablations/standard10_second_run_videos.md').write_text('\n'.join(lines) + '\n')
    if any(j['state'] != 'complete' for j in jobs):
        raise SystemExit('Some videos failed; inspect status.json and per-video logs')


if __name__ == '__main__':
    main()
