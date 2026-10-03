#!/usr/bin/env python3
"""Focused CPU-only WP2 checks; report unavailable source checks as BLOCKED."""

import argparse
import importlib.abc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'diffusers', 'huggingface_hub', 'accelerate'}:
            raise AssertionError(f'Unexpected model/runtime import: {fullname}')


sys.meta_path.insert(0, NoModels())

import h5py
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from diffusion.utils.config import SanaConfig
from diffusion.data.transforms import get_transform
from jsc_jupiter.hdf5_samples import pair_metadata
from radargen.datasets.truckscenes import TruckScenesAdapter
from radargen.ppp_targets import create_ppp_targets_from_detections
from radargen.radar_maps.creator import deduplicate_by_pixel, normalize_to_image_coords
from radargen.training.radargen_dataset import RadarGenDataset
from radargen.training.radargen_ppp_dataset import build_radargen_ppp_dataset_from_config, read_ppp_sample


class UnpublishedSampleProbe(Dataset):
    """Exercise the shared item reader, not completion/index acceptance.

    This diagnostic cannot be used as the training loader and does not claim
    that the source scene is completed. It owns no HDF5 handles or adapter.
    """
    def __init__(self, entries, resolution, normalization):
        self.entries = entries
        self.resolution = resolution
        self.normalization = normalization
        self.transform = get_transform('default_train_from_np', resolution)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, expected = self.entries[index]
        sample, _ = read_ppp_sample(path, expected, resolution=self.resolution,
                                    point_limit=self.normalization['coordinate_range'],
                                    normalization=self.normalization, map_transform=self.transform)
        return sample


def round_trip(xy, *, size, limit):
    uv = (xy + limit) / (2 * limit) * size
    cols = np.floor(uv[:, 0])
    rows = size - 1 - np.floor(uv[:, 1])
    restored = np.stack(((cols + .5) / size * 2 * limit - limit,
                         (size - rows - .5) / size * 2 * limit - limit), axis=1)
    assert (np.abs(restored - xy) <= limit / size + 1e-5).all()
    return rows.astype(int), cols.astype(int)


def synthetic_checks():
    uv = np.array([[2.1, 2.1], [2.6, 2.1], [4.1, 4.1], [4.2, 4.1], [6.2, 6.2], [6.2, 6.2]], np.float32)
    xy = uv - 4
    rcs = np.array([1, 2, 3, 3, 0, -1], np.float32)
    doppler = np.array([10, 20, -30, 40, 0, 60], np.float32)
    originals = [a.copy() for a in (xy, rcs, doppler)]
    mask, rg, dg, stats = create_ppp_targets_from_detections(xy, rcs, doppler, image_size=8, point_limit=4)
    assert (stats['stored'], stats['after_rounded'], stats['final_cell_collisions'], stats['active_cells']) == (6, 4, 1, 3)
    np.testing.assert_array_equal(np.sort(stats['retained_indices']), [1, 2, 4])
    # Synthetic helper layout only: unused columns have no physical meaning.
    pcl = np.zeros((6, 8), np.float32)
    pcl[:, :2], pcl[:, 6], pcl[:, 7] = xy, rcs, np.arange(6)
    baseline = deduplicate_by_pixel(normalize_to_image_coords(pcl, 4, 8))
    np.testing.assert_array_equal(baseline[:, 7].astype(int), stats['rounded_indices'])
    rows, cols = round_trip(xy[stats['retained_indices']], size=8, limit=4)
    np.testing.assert_array_equal(rg[rows, cols], rcs[stats['retained_indices']])
    np.testing.assert_array_equal(dg[rows, cols], doppler[stats['retained_indices']])
    assert mask[rows[2], cols[2]] == 1 and rg[rows[2], cols[2]] == dg[rows[2], cols[2]] == 0
    points = np.array([[-3.999, -3.999], [3.999, 3.999], [-1, 1], [1, -1]], np.float32)
    rows, cols = round_trip(points, size=8, limit=4)
    assert rows[0] == 7 and rows[1] == 0 and rows[2] < rows[3]
    create_ppp_targets_from_detections(points, np.ones(4, np.float32), np.ones(4, np.float32), image_size=8, point_limit=4)
    for bad in (np.array([[4, 0]], np.float32), np.array([[np.nan, 0]], np.float32), np.empty((0, 2), np.float32)):
        try:
            create_ppp_targets_from_detections(bad, np.zeros(len(bad), np.float32), np.zeros(len(bad), np.float32), image_size=8, point_limit=4)
        except ValueError:
            pass
        else:
            raise AssertionError('Boundary/nonfinite/empty source accepted')
    for before, after in zip(originals, (xy, rcs, doppler)):
        np.testing.assert_array_equal(before, after)
    print('PASS synthetic: stored=6 rounded=4 final_collisions=1 active=3; ties, mark pairs, zeros, orientation, boundaries, empty refusal')


def assert_same(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        elif isinstance(left[key], dict):
            assert_same(left[key], right[key])
        else:
            assert left[key] == right[key], key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default='configs/RadarGen_600M_512px_TS_PPP_training.yaml')
    parser.add_argument('--legacy-bev-dir', default='/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/preprocessing/smoke/bev_condition_maps')
    args = parser.parse_args()
    torch.set_num_threads(1)
    synthetic_checks()
    cfg = pyrallis.parse(SanaConfig, args=['--config_path', args.config_path])
    from scripts.train_ppp import validate_ppp_config
    validate_ppp_config(cfg)
    scene, = cfg.data.extra['ppp_scene_tokens']
    adapter = TruckScenesAdapter(dataset_dir=cfg.data.dataset_dir, dataset_version=cfg.data.extra['dataset_version'],
                                 camera_views=cfg.data.extra['camera_views'])
    token, frames = next(adapter.iter_scenes(cfg.data.extra['eval_split'], filter_fn=lambda s: s == scene))
    assert token == scene
    entries = [(Path(cfg.data.ppp_data_dir)/scene/f'sample_{i:06d}.h5',
                dict(pair_metadata(adapter, frames[i], frames[i+1], cfg.data.extra['eval_split']), scene_token=scene, frame_index=i))
               for i in (0, 50, 100)]
    probe = UnpublishedSampleProbe(entries, cfg.data.image_size, vars(adapter.normalization).copy())
    blocked = []
    try:
        completed = build_radargen_ppp_dataset_from_config(cfg)
    except ValueError as error:
        if '_SUCCESS.json' not in str(error):
            raise
        blocked.append('completed-scene loader: '+str(error))
        print('PASS strict refusal:', error)
        completed = None
    else:
        assert len(completed) == len(frames)-1
        assert completed[-1]['data_info']['frame_idx'] == len(frames)-2
        for i, sample in zip((0, 50, 100), probe):
            assert_same(completed[i], sample)
        print(f'PASS completed scene: {scene}, frames={len(frames)}, pairs={len(completed)}, final pair retained')
    baseline = object.__new__(RadarGenDataset)
    baseline.adapter = adapter
    baseline.bev_conditioning_maps_dir = args.legacy_bev_dir
    baseline.map_transform = get_transform('default_train_from_np', cfg.data.image_size)
    for index, (path, expected) in enumerate(entries):
        sample = probe[index]
        with h5py.File(path, 'r') as handle:
            xy, rcs, doppler = (handle[f'ppp_targets/{k}'][()] for k in ('xy', 'rcs', 'doppler'))
            _, _, _, stats = create_ppp_targets_from_detections(xy, rcs, doppler, image_size=512, point_limit=50)
        retained = stats['retained_indices']
        rows, cols = round_trip(xy[retained], size=512, limit=50)
        np.testing.assert_array_equal(sample['rcs_target'][0, rows, cols], rcs[retained])
        np.testing.assert_array_equal(sample['doppler_target'][0, rows, cols], doppler[retained])
        assert set(sample) == {'point_mask','rcs_target','doppler_target','bev_color_map','bev_seg_map','bev_velocity_map','data_info'}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == torch.float32 and torch.isfinite(value).all()
                assert value.shape == ((3,512,512) if key.startswith('bev_') else (1,512,512))
        print(f"PASS sample {scene}/{expected['frame_index']}: stored={stats['stored']} rounded={stats['after_rounded']} final_collisions={stats['final_cell_collisions']} active={stats['active_cells']}; physical marks exact")
        try:
            legacy = baseline._load_bev_maps(scene, expected['frame_index'])
        except FileNotFoundError as error:
            blocked.append(f'legacy conditions frame {expected["frame_index"]}: {error.filename}')
        else:
            for array, key in zip(legacy, ('bev_color_map','bev_seg_map','bev_velocity_map')):
                other = baseline._apply_transform(array)
                torch.testing.assert_close(sample[key], other, rtol=1e-6, atol=1e-6)
                print(f'PASS conditioning {expected["frame_index"]}/{key}: max_abs_diff={(sample[key]-other).abs().max().item()}')
    dataset = Subset(completed, [0,50,100]) if completed is not None else probe
    reference = list(DataLoader(dataset, batch_size=2, num_workers=0))
    multi = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True, multiprocessing_context='spawn')
    for epoch in range(2):
        actual = list(multi)
        assert len(actual) == len(reference) == 2
        for a, b in zip(reference, actual):
            assert_same(a,b)
    print('PASS batching: batch sizes 2,1; workers=0 versus 2 (spawn, persistent), two iterations; '+('completed dataset' if completed is not None else 'item-reader probe only'))
    for reason in blocked:
        print('BLOCKED:', reason)
    print('CPU component checks complete; no models, GPUs, or training initialized.')
    return 2 if blocked else 0


if __name__ == '__main__':
    sys.exit(main())
