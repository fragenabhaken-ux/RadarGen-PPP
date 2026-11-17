"""Export aligned paper panels from archived scene predictions, without inference."""
import argparse
import json
import io
import subprocess
import tarfile
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--model', default='dino_2773475_step15000')
    parser.add_argument('--frame', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    archive = args.scene_dir / 'predictions' / (args.model + '_dense.h5')
    with h5py.File(archive, 'r') as saved:
        assert saved.attrs['complete']
        provenance = json.loads(saved.attrs['provenance'])
        record = next(r for r in provenance['frames'] if r['frame_index'] == args.frame)
        group = saved['frames'][f'{args.frame:06d}']
        assert group.attrs['reference_token'] == record['reference_sample_data_token']
        fields = {k: group[k][()] for k in ('heatmap', 'rcs', 'doppler', 'points')}
    contract = provenance['dataset_contract']
    root = Path(contract['dataset_root'])
    source = Path(contract['ppp_root']) / provenance['scene'] / f'sample_{args.frame:06d}.h5'
    # Stream the large metadata table, retaining only the eight exact camera tokens.
    wanted = set(record['camera_t0_sample_data_tokens'] + record['camera_t1_sample_data_tokens'])
    camera_records = {}
    buffer = []
    with (root / contract['dataset_version'] / 'sample_data.json').open() as stream:
        for line in stream:
            if line.strip() == '{':
                buffer = [line]
            elif buffer:
                buffer.append(line)
                if line.strip() in ('},', '}'):
                    entry = json.loads(''.join(buffer).rstrip().rstrip(','))
                    if entry['token'] in wanted:
                        camera_records[entry['token']] = entry
                        if len(camera_records) == len(wanted):
                            break
                    buffer = []
    assert set(camera_records) == wanted
    archived_images = {}
    missing = {e['filename'] for e in camera_records.values() if not (root / e['filename']).is_file()}
    camera_archive = args.scene_dir.parents[1] / 'camera_input_archives' / (provenance['scene'] + '.tar.zst')
    if missing:
        with subprocess.Popen(['zstd', '-dc', str(camera_archive)], stdout=subprocess.PIPE) as process:
            with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
                for member in stream:
                    name = member.name.removeprefix('./')
                    if name in missing:
                        archived_images[name] = stream.extractfile(member).read()
            assert process.wait() == 0
        assert set(archived_images) == missing
    cameras = []
    with h5py.File(source, 'r') as sample:
        assert sample['metadata/reference_sample_data_token'].asstr()[()] == record['reference_sample_data_token']
        for step in ('t0', 't1'):
            assert list(sample[f'metadata/camera_{step}_sample_data_tokens'].asstr()[()]) == record[f'camera_{step}_sample_data_tokens']
            for name, timestamp, token in zip(record['camera_names'], record[f'camera_{step}_timestamps_us'], record[f'camera_{step}_sample_data_tokens']):
                entry = camera_records[token]
                assert entry['timestamp'] == timestamp
                matches = [root / entry['filename']]
                filename = f'camera_{step}_{name.removeprefix("CAMERA_").lower()}.png'
                image_source = matches[0] if matches[0].is_file() else io.BytesIO(archived_images[entry['filename']])
                with Image.open(image_source) as im:
                    im.convert('RGB').save(out / filename)
                cameras.append(dict(output=filename, source=str(matches[0]), token=token, timestamp_us=timestamp))
        for name in ('appearance', 'semantics', 'velocity'):
            Image.fromarray(sample[f'conditioning/{name}'][()]).save(out / f'modality_{name}.png')
    # Same physical square and orientation for all BEV panels; no axes or padding.
    specs = [('intensity', 'heatmap', 'hot', 0, float(fields['heatmap'].max()) ** .2),
             ('rcs', 'rcs', 'viridis', -20, 66), ('doppler', 'doppler', 'coolwarm', -30, 30)]
    for label, key, cmap, lo, hi in specs:
        data = np.maximum(fields[key], 0) ** .2 if key == 'heatmap' else fields[key]
        assert np.isfinite(data).all()
        plt.imsave(out / f'heatmap_{label}.png', data, cmap=cmap, vmin=lo, vmax=hi, origin='upper')
    points = fields['points']
    radius = provenance['coordinate_range']
    for mark in ('location', 'rcs', 'doppler'):
        fig = plt.figure(figsize=(5.12, 5.12), dpi=300, facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
        style = dict(c='white') if mark == 'location' else dict(c=points[:, 2 if mark == 'rcs' else 3], cmap='viridis' if mark == 'rcs' else 'coolwarm', vmin=-20 if mark == 'rcs' else -30, vmax=66 if mark == 'rcs' else 30)
        ax.scatter(points[:, 0], points[:, 1], s=1.6, linewidths=0, **style)
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')
        ax.set_axis_off()
        fig.savefig(out / f'pointcloud_{mark}.png', facecolor='black', dpi=300)
        plt.close(fig)
    manifest = dict(scene=provenance['scene'], frame=record, checkpoint=provenance['checkpoint'], prediction_archive=str(archive), conditioning_source=str(source), cameras=cameras, sampled_point_count=len(points), rendering=dict(extent_m=[-radius,radius,-radius,radius], intensity='hot; expected points/cell; power 0.2; range 0 to frame maximum', rcs='viridis; -20 to 66 dBsm', doppler='coolwarm; -30 to 30 m/s', pointcloud='black background; white location points; same sampled points colored by RCS or Doppler'), selection=f'Frame index {args.frame} exported for illustration, without metric-based selection; DINO 2773475 step 15000 is the final DINO selection in model_training_lineage.md.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'README.md').write_text(f'''Paper material: scene {provenance['scene']}, frame {args.frame}

17 PNGs: 8 original-resolution camera views (t0 and t1), 3 computed BEV input modalities, 3 predicted DINO fields, and 3 views of the same sampled point cloud ({len(points)} points).

Model: DINO run 2773475, checkpoint step 15000. The cameras are the exact consecutive pair recorded for these modalities, approximately 100 ms apart (camera hardware timestamps differ slightly across views). Camera PNGs are decoded from original JPEGs. Modality PNGs preserve the stored uint8 RGB arrays. Heatmaps preserve the 1024×1024 prediction grid; point clouds are 1536×1536. All BEV panels cover x/y ±50 m, positive y upward. There are no axes, titles, borders, or colorbars.

Intensity uses hot with power 0.2 for visibility. RCS uses viridis, −20…66 dBsm. Doppler uses coolwarm, −30…30 m/s. Point-cloud location uses white dots on black; RCS and Doppler color the same points on black. Mark heatmaps show the full predicted fields, including low-intensity regions. See manifest.json for exact sources, camera tokens/timestamps, checkpoint, and rendering settings.
''')
    pngs = sorted(out.glob('*.png'))
    assert len(pngs) == 17
    for path in pngs:
        with Image.open(path) as im:
            im.verify()
    print(f'Validated {len(pngs)} PNGs in {out.resolve()}')


if __name__ == '__main__':
    main()
