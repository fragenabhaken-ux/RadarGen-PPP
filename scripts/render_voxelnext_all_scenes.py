#!/usr/bin/env python3
"""Render every saved VoxelNeXt validation scene, without GPU inference."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py

from scripts.render_voxelnext_scene_video import ROOT, CASES
from scripts.run_ablation_checkpoint_programme import atomic_json

OUTPUT = Path('output/voxelnext_scene_videos_resnet')


def verified(directory, scene, expected):
    saved = json.loads((directory / 'manifest.json').read_text())
    if saved['scene'] != scene or len(saved['frames']) != expected or not saved['annotated_keyframes_only']:
        raise ValueError('Incorrect scene manifest')
    data = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate:format=duration','-of','json',str(directory/'comparison.mp4')]))
    stream = data['streams'][0]
    n,d = map(float,stream['r_frame_rate'].split('/'))
    if int(stream['nb_frames']) != expected or abs(n/d-saved['fps']) > .001:
        raise ValueError('Incorrect video frame count or FPS')
    if abs(float(data['format']['duration'])-expected/saved['fps']) > .1:
        raise ValueError('Incorrect video duration')
    return dict(frames=expected,fps=saved['fps'],seconds=float(data['format']['duration']))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'logs').mkdir(exist_ok=True)
    with h5py.File(ROOT/'detection'/CASES[0][1]/'detections.h5') as detector:
        settings=json.loads(detector.attrs['settings_json'])
    counts={}
    with h5py.File(settings['archive']) as archive:
        for key in archive['index/group'].asstr()[:]:
            scene=key.split('/')[1]
            counts[scene]=counts.get(scene,0)+1
    if len(counts)!=49 or sum(counts.values())!=1906:
        raise ValueError('Expected the full 49-scene, 1906-keyframe validation archive')
    states={s:dict(state='pending',frames=n) for s,n in sorted(counts.items())}

    def save():
        atomic_json(OUTPUT/'status.json',states)
        lines=['# VoxelNeXt scene videos: final ResNet versus RadarGen','',
            'All 49 validation scenes, 1,906 annotated keyframes. Approximately 2 fps; '
            'no inference on intervening 10 Hz frames and no temporal interpolation. '
            'Points: RCS colours; orange: ground truth; dashed green: VoxelNeXt detections '
            '(confidence ≥ 0.25). RN50 checkpoint: 2882612 / step 25,030. '
            'Both models use their own saved generated clouds and detector outputs. '
            'No new GPU job.','', '| Scene | Status | Video |','|---|---|---|']
        for scene,state in states.items():
            link=f'[MP4]({scene}/comparison.mp4)' if state['state']=='complete' else 'pending'
            lines.append(f"| {scene} | {state['state']} | {link} |")
        lines+=['','Per-scene manifest.json files record source archives, detection settings, '
                'sample identities and timestamps. Saved detections use z=1 flat-up; '
                'equivalence to the original private preprocessing remains unverified.']
        (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n')

    def one(scene):
        directory=OUTPUT/scene
        try:
            return verified(directory,scene,counts[scene])
        except (OSError,ValueError,KeyError,subprocess.CalledProcessError):
            pass
        if scene=='044c648ac12345f1aedf33c9f91cdc5a':
            source=Path('output/voxelnext_scene_video_resnet')
            try:
                verified(source,scene,counts[scene])
                directory.mkdir(exist_ok=True)
                for name in ('comparison.mp4','preview.png','manifest.json'):
                    shutil.copy2(source/name,directory/name)
                return verified(directory,scene,counts[scene])
            except (OSError,ValueError,KeyError,subprocess.CalledProcessError):
                pass
        with (OUTPUT/'logs'/f'{scene}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.render_voxelnext_scene_video',
                '--scene',scene,'--output',str(directory)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return verified(directory,scene,counts[scene])

    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,s):s for s in states}
        for future in as_completed(futures):
            scene=futures[future]
            try:
                states[scene].update(state='complete',**future.result())
            except BaseException as error:
                states[scene].update(state='failed',error=repr(error))
            save()
            print(scene,states[scene]['state'],flush=True)
    if any(s['state']!='complete' for s in states.values()):
        raise RuntimeError('Some scenes failed; inspect status.json and logs')
    print('All 49 VoxelNeXt comparison videos verified.',flush=True)


if __name__=='__main__':
    main()
