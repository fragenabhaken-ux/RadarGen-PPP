"""Add power-0.5 and power-0.7 intensity panels from original saved fields."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/paper_material/diagram_images'))
    args = parser.parse_args()
    manifests = sorted(args.root.glob('*/manifest.json'))
    if not manifests:
        raise ValueError(f'No frame manifests in {args.root}')
    for path in manifests:
        manifest = json.loads(path.read_text())
        frame = manifest['frame']['frame_index']
        with h5py.File(manifest['prediction_archive'], 'r') as archive:
            group = archive['frames'][f'{frame:06d}']
            assert group.attrs['reference_token'] == manifest['frame']['reference_sample_data_token']
            mass = np.maximum(group['heatmap'][()], 0)
        assert np.isfinite(mass).all() and mass.max() > 0
        variants = {}
        for power in (0.5, 0.7):
            name = f'heatmap_intensity_power_{power:.1f}.png'
            plt.imsave(path.parent / name, mass ** power, cmap='hot',
                       vmin=0, vmax=float(mass.max()) ** power, origin='upper')
            with Image.open(path.parent / name) as im:
                assert im.size == (mass.shape[1], mass.shape[0])
                im.verify()
            variants[name] = dict(power=power, cmap='hot', vmin=0,
                                  vmax=float(mass.max()) ** power,
                                  normalization='(intensity / frame_maximum) ** power')
        manifest['rendering']['intensity_variants'] = variants
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        readme = path.parent / 'README.md'
        if readme.exists():
            content = readme.read_text().replace('17 PNGs:', '19 PNGs (including two additional intensity variants):')
            note = '\nAdditional intensity variants: `heatmap_intensity_power_0.5.png` and `heatmap_intensity_power_0.7.png`. Both use the original saved intensity field, the hot colormap, and the same per-frame maximum normalization. `heatmap_intensity.png` remains power 0.2. Higher powers darken low-intensity regions relative to the peak.\n'
            if 'Additional intensity variants:' not in content:
                content += note
            readme.write_text(content)
        print(f'{path.parent.name}: added and verified two variants')
    index = args.root / 'README.md'
    if index.exists():
        content = index.read_text().replace('17 PNGs', '19 PNGs')
        if 'power-0.5 and power-0.7' not in content:
            content += '\nEach folder includes additional power-0.5 and power-0.7 intensity heatmaps; the original intensity image remains power 0.2.\n'
        index.write_text(content)
    print(f'Validated {2 * len(manifests)} new PNGs across {len(manifests)} frame folders.')


if __name__ == '__main__':
    main()
