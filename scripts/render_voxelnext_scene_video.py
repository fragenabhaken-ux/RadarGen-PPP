#!/usr/bin/env python3
"""CPU-only keyframe scene video from saved RN50 and RadarGen detector results."""
import io
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen')
SCENE = '044c648ac12345f1aedf33c9f91cdc5a'
OUTPUT = Path('output/voxelnext_scene_video_resnet')
CASES = [
    ('C3P-dar-RN50', 'ablation_checkpoint_programme/resnet_second_2882612/attempt_000'),
    ('RadarGen', '2838586_radargen_nameorder_z1_flatup'),
]


def quaternion_rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    global SCENE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default=SCENE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--include-real', action='store_true', help='Add real radar and its saved detector results on the left')
    parser.add_argument('--save-frames', action='store_true', help='Also save each rendered keyframe as PNG')
    args = parser.parse_args()
    SCENE, OUTPUT = args.scene, args.output
    cases = ([('GT (real radar)', '2818444_real_baseline')] if args.include_real else []) + CASES
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        (OUTPUT / 'frames').mkdir(exist_ok=True)
    detections = [h5py.File(ROOT / 'detection' / case / 'detections.h5') for _, case in cases]
    archives = []
    encoder = None
    manifest = dict(scene=SCENE, confidence=.25, models=[], frames=[], annotated_keyframes_only=True)
    try:
        for (label, _), detector in zip(cases, detections):
            if not detector.attrs.get('complete'):
                raise ValueError('Incomplete detections')
            settings = json.loads(detector.attrs['settings_json'])
            if settings['side'] != 'real' and (settings['height'] != 1 or settings['height_frame'] != 'flat-up'):
                raise ValueError('Unexpected detector protocol')
            archives.append(h5py.File(settings['archive']))
            manifest['models'].append(dict(label=label, archive=settings['archive'],
                detections=str(Path(detector.filename).resolve()), settings=settings))
        keys = [k for k in archives[0]['index/group'].asstr()[:] if k.split('/')[1] == SCENE]
        keys.sort(key=lambda k:json.loads(archives[0][k].attrs['metadata_json'])['reference_timestamp_us'])
        timestamps = [json.loads(archives[0][k].attrs['metadata_json'])['reference_timestamp_us'] for k in keys]
        if len(keys) < 2:
            raise ValueError('Expected a sequence of annotated frames')
        fps = 1e6 / float(np.median(np.diff(timestamps)))
        manifest['fps'] = fps
        for number, key in enumerate(keys):
            metadata = json.loads(archives[0][key].attrs['metadata_json'])
            with h5py.File(archives[0][key].attrs['source']) as source:
                transform = source['metadata/global_to_flat_up'][:]
                annotations = source['metadata/bounding_boxes']
                gt_boxes = []
                for center, size, quat, category in zip(annotations['center_global'][:],
                        annotations['size_wlh'][:], annotations['rotation_wxyz'][:], annotations['category_name'][:]):
                    category = category.decode() if isinstance(category, bytes) else category
                    if category.split('.')[-1] not in ('car', 'truck', 'trailer'):
                        continue
                    width, length, _ = size
                    corners = np.array([[length/2,width/2,0],[length/2,-width/2,0],
                                        [-length/2,-width/2,0],[-length/2,width/2,0]])
                    corners = corners @ quaternion_rotation(quat).T + center
                    corners = (corners @ transform[:3,:3].T + transform[:3,3])[:,:2]
                    if np.all(corners.max(0) >= -50) and np.all(corners.min(0) <= 50):
                        gt_boxes.append(corners)
            fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 6.6), constrained_layout=True)
            try:
                for ax, archive, detector, (label, _), model_info in zip(axes, archives, detections, cases, manifest['models']):
                    frame = archive[key]
                    other = json.loads(frame.attrs['metadata_json'])
                    group = detector[key]
                    det_meta = json.loads(group.attrs['metadata_json'])
                    if any(p['sample_token'] != metadata['sample_token'] for p in (other, det_meta)):
                        raise ValueError('Sample mismatch')
                    if model_info['settings']['side'] == 'real':
                        with h5py.File(frame.attrs['source']) as source:
                            points = np.column_stack([source[f'ppp_targets/{name}'][:]
                                                      for name in ('xy', 'rcs', 'doppler')])
                    else:
                        points = frame['points'][:]
                    artist = ax.scatter(points[:,0], points[:,1], c=points[:,2], s=3,
                                        cmap='viridis', vmin=-20, vmax=35, linewidths=0)
                    for corners in gt_boxes:
                        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='#ed9835', lw=1.1))
                    inverse = np.linalg.inv(group['detector_from_flat_up'][:])
                    count = 0
                    for box, score in zip(group['boxes'][:], group['scores'][:]):
                        if score < .25:
                            continue
                        x,y,z,dx,dy,dz,yaw = box[:7]
                        corners = np.array([[dx/2,dy/2,0],[dx/2,-dy/2,0],[-dx/2,-dy/2,0],[-dx/2,dy/2,0]])
                        c,s = np.cos(yaw),np.sin(yaw)
                        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                        corners = (corners @ rotation.T + [x,y,z]) @ inverse[:3,:3].T + inverse[:3,3]
                        ax.add_patch(Polygon(corners[:,:2], closed=True, fill=False,
                                             edgecolor='#009d73', lw=1.2, linestyle='--'))
                        count += 1
                    ax.add_patch(Rectangle((-1.3,-3.5),2.6,7,facecolor='#FF1493',edgecolor='black',lw=.5,zorder=5))
                    ax.set(xlim=(-50,50),ylim=(-50,50),aspect='equal',xlabel='x [m]',ylabel='y [m]',
                           title=f'{label} | {count} VoxelNeXt detections')
                    ax.grid(alpha=.15)
                fig.colorbar(artist, ax=list(axes), label='RCS [dBsm]', shrink=.75)
                elapsed = (timestamps[number]-timestamps[0])/1e6
                fig.suptitle(f'VoxelNeXt | scene {SCENE}\n{elapsed:.1f} s | annotated keyframe {number+1}/{len(keys)} | confidence ≥ 0.25')
                fig.legend(handles=[Line2D([],[],color='#ed9835',label='Ground truth'),
                    Line2D([],[],color='#009d73',linestyle='--',label='VoxelNeXt prediction')],
                    loc='outside lower center',ncol=2)
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=110)
                payload = buffer.getvalue()
                if args.save_frames:
                    (OUTPUT/'frames'/f'frame_{int(key.rsplit("/", 1)[-1]):06d}.png').write_bytes(payload)
                if number == 0:
                    (OUTPUT/'preview.png').write_bytes(payload)
                if encoder is None:
                    encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','image2pipe',
                        '-vcodec','png','-framerate',str(fps),'-i','-','-an','-c:v','libx264',
                        '-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',str(OUTPUT/'comparison.mp4')],stdin=subprocess.PIPE)
                encoder.stdin.write(payload)
                manifest['frames'].append(dict(key=key,sample_token=metadata['sample_token'],timestamp_us=timestamps[number]))
                print(f'Rendered {number+1}/{len(keys)}',flush=True)
            finally:
                plt.close(fig)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoder failed')
        encoder = None
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,r_frame_rate:format=duration','-of','json',str(OUTPUT/'comparison.mp4')]))
        if int(probe['streams'][0]['nb_frames']) != len(keys):
            raise ValueError('Video frame count differs')
        manifest['probe'] = probe
        (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(OUTPUT.resolve()/'comparison.mp4',flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        for f in archives + detections:
            f.close()


if __name__ == '__main__':
    main()
