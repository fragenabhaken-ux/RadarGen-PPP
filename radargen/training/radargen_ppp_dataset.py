"""Completed-scene PPP loading with read-only, process-local HDF5 access."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.data.transforms import get_transform
from jsc_jupiter.hdf5_samples import PPP_POLICY, SCHEMA_VERSION, pair_metadata
from radargen.core.data_types import PPPTrainingBatch
from radargen.ppp_targets import create_ppp_targets_from_detections

LOG = logging.getLogger(__name__)
CONDITIONS = (("appearance", "bev_color_map"), ("semantics", "bev_seg_map"),
              ("velocity", "bev_velocity_map"))


def _value(dataset):
    return dataset.asstr()[()] if h5py.check_string_dtype(dataset.dtype) else dataset[()]


def validate_ppp_metadata(handle, expected, *, resolution, point_limit, normalization):
    """Validate selected metadata and six consumed datasets; never read depth/boxes."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"{handle.filename}: {message}")

    meta = handle['metadata']
    for key, value in dict(expected, schema_version=SCHEMA_VERSION, ppp_filtering_policy=PPP_POLICY,
                           map_resolution=resolution, point_limit=point_limit,
                           conditioning_coordinate_range=point_limit, nsweeps=1,
                           range_filter='abs(x) < point_limit AND abs(y) < point_limit',
                           frustum_camera_views=expected['camera_names']).items():
        require(key in meta and np.array_equal(_value(meta[key]), value), f"metadata/{key} mismatch")
    norm = json.loads(meta['normalization_json'].asstr()[()])
    require(norm == normalization, "normalization provenance mismatch")
    producer = json.loads(meta['configuration_json'].asstr()[()])
    for key, value in dict(dataset_version=expected['dataset_version'], resolution=resolution,
                           point_limit=point_limit, nsweeps=1).items():
        require(producer.get(key) == value, f"producer configuration/{key} mismatch")
    for key in ('camera_views', 'filter_camera_views'):
        require((producer.get(key) or expected['camera_names']) == expected['camera_names'],
                f"producer configuration/{key} mismatch")
    for name, _ in CONDITIONS:
        ds = handle[f'conditioning/{name}']
        require(ds.shape == (resolution, resolution, 3) and ds.dtype == np.uint8,
                f"conditioning/{name} shape/dtype mismatch")
    targets = handle['ppp_targets']
    for key, value in dict(xy_frame='RadarGen_vehicle_flat_up', xy_units='m',
                           rcs_units='dBsm', doppler_units='m/s',
                           doppler_definition='sensor_velocity_norm_times_sign_velocity_dot_sensor_position').items():
        require(targets.attrs.get(key) == value, f"ppp_targets attribute {key} mismatch")
    n = targets['rcs'].shape[0] if targets['rcs'].ndim == 1 else -1
    for name, shape in (('xy', (n, 2)), ('rcs', (n,)), ('doppler', (n,))):
        ds = targets[name]
        require(ds.shape == shape and ds.dtype == np.float32, f"ppp_targets/{name} shape/dtype mismatch")
    return n


def read_ppp_sample(path, expected, *, resolution, point_limit, normalization, map_transform):
    """Materialize one validated sample, close the file, then transform/rasterize."""
    with h5py.File(path, 'r') as handle:
        validate_ppp_metadata(handle, expected, resolution=resolution,
                              point_limit=point_limit, normalization=normalization)
        conditions = {key: handle[f'conditioning/{name}'][()] for name, key in CONDITIONS}
        xy, rcs, doppler = (handle[f'ppp_targets/{key}'][()] for key in ('xy', 'rcs', 'doppler'))
    if len(rcs) == 0:
        LOG.warning("Empty PPP frame: scene=%s frame=%s N_stored=0; policy unresolved",
                    expected['scene_token'], expected['frame_index'])
    mask, rcs_grid, doppler_grid, stats = create_ppp_targets_from_detections(
        xy, rcs, doppler, image_size=resolution, point_limit=point_limit)
    output = {key: map_transform(array) for key, array in conditions.items()}
    for key, tensor in output.items():
        if tensor.shape != (3, resolution, resolution) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f'{path}: invalid transformed condition {key}')
    output.update(point_mask=torch.from_numpy(mask[None]), rcs_target=torch.from_numpy(rcs_grid[None]),
                  doppler_target=torch.from_numpy(doppler_grid[None]),
                  data_info=dict(img_hw=torch.tensor([resolution, resolution], dtype=torch.float32),
                                 aspect_ratio=torch.tensor(1.0), scene_token=expected['scene_token'],
                                 ref_sensor_token=expected['reference_sample_data_token'],
                                 frame_idx=expected['frame_index']))
    return output, stats


class RadarGenPPPDataset(Dataset):
    """Index a whole requested split or an explicit completed-scene subset.

    Initialization reads metadata/dataset headers for selected pairs, not image,
    detection, depth, or annotation arrays. Files must remain immutable afterward.
    Empty frames explicitly stop initialization; no training policy is assumed.
    """
    load_text_feat = False
    load_vae_feat = False

    def __init__(self, adapter, *, ppp_data_dir, split, processing_signature,
                 scene_tokens=None, resolution=512):
        if split not in ('train', 'val') or resolution != 512:
            raise ValueError('Initial PPP loader requires train/val split and resolution 512')
        if not isinstance(processing_signature, str) or len(processing_signature) != 64 or any(
                c not in '0123456789abcdef' for c in processing_signature):
            raise ValueError('An independently verified SHA256 processing signature is required')
        if scene_tokens is not None and (not isinstance(scene_tokens, list) or not scene_tokens or
                any(not isinstance(s, str) for s in scene_tokens) or len(set(scene_tokens)) != len(scene_tokens)):
            raise ValueError('scene_tokens must be a nonempty list of distinct scene tokens or None')
        self.resolution = resolution
        self.point_limit = adapter.normalization.coordinate_range
        self.normalization = vars(adapter.normalization).copy()
        self.map_transform = get_transform('default_train_from_np', resolution)
        self.frame_index = []
        self.empty_frame_count = 0
        root = Path(ppp_data_dir).expanduser().resolve()
        wanted = set(scene_tokens) if scene_tokens is not None else None
        found = set()
        for scene, frames in adapter.iter_scenes(split, filter_fn=lambda token: wanted is None or token in wanted):
            found.add(scene)
            directory = root / scene
            if len(frames) < 2 or directory.is_symlink() or not directory.is_dir():
                raise ValueError(f'{scene}: missing/invalid scene or fewer than two adapter frames')
            marker_path = directory / '_SUCCESS.json'
            if marker_path.is_symlink() or not marker_path.is_file():
                raise ValueError(f'{scene}: missing/invalid _SUCCESS.json; incomplete scenes are not accepted')
            marker = json.loads(marker_path.read_text())
            pairs = len(frames) - 1
            if not isinstance(marker, dict) or any(marker.get(k) != v for k, v in dict(
                    scene=scene, signature=processing_signature, expected_frames=len(frames), expected_pairs=pairs).items()):
                raise ValueError(f'{scene}: incompatible processing signature or adapter frame/pair counts')
            names = {f'sample_{i:06d}.h5' for i in range(pairs)}
            if {p.name for p in directory.iterdir()} != names | {'_SUCCESS.json'}:
                raise ValueError(f'{scene}: exact sample inventory mismatch')
            sizes = {}
            for i in range(pairs):
                path = directory / f'sample_{i:06d}.h5'
                if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f'{path}: missing, linked, or empty sample')
                sizes[path.name] = path.stat().st_size
            if marker.get('files') != sizes:
                raise ValueError(f'{scene}: completion-record file sizes/inventory mismatch')
            for i in range(pairs):
                expected = dict(pair_metadata(adapter, frames[i], frames[i+1], split), scene_token=scene, frame_index=i)
                path = directory / f'sample_{i:06d}.h5'
                with h5py.File(path, 'r') as handle:
                    count = validate_ppp_metadata(handle, expected, resolution=resolution,
                                                  point_limit=self.point_limit, normalization=self.normalization)
                self.empty_frame_count += count == 0
                stat = path.stat()
                self.frame_index.append((path, expected, stat.st_size, stat.st_mtime_ns))
        if wanted is not None and found != wanted:
            raise ValueError(f'Selected scenes are absent from adapter split {split}: {sorted(wanted-found)}')
        if not self.frame_index:
            raise ValueError(f'No completed PPP pairs selected for split {split}')
        if self.empty_frame_count:
            LOG.error('Selected PPP inventory contains %d empty frames; training policy unresolved', self.empty_frame_count)
            raise ValueError(f'{self.empty_frame_count} empty PPP frames; define an explicit training policy before loading')
        self.ori_imgs_nums = len(self.frame_index)

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, index) -> PPPTrainingBatch:
        path, expected, size, mtime = self.frame_index[index]
        stat = path.stat()
        if path.is_symlink() or (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f'{path}: indexed immutable sample changed')
        sample, _ = read_ppp_sample(path, expected, resolution=self.resolution,
                                    point_limit=self.point_limit, normalization=self.normalization,
                                    map_transform=self.map_transform)
        return sample


def build_radargen_ppp_dataset_from_config(config, **kwargs):
    """Construct a metadata-only TruckScenes adapter and completed PPP dataset."""
    from radargen.core.normalization import NormalizationConfig
    from radargen.core.protocols import DatasetConfig
    from radargen.datasets.truckscenes import TruckScenesAdapter
    from radargen.datasets.truckscenes.config import TRUCKSCENES_REFERENCE_SENSOR
    from truckscenes import TruckScenes

    extra = config.data.extra
    if extra.get('dataset_name') != 'truckscenes':
        raise ValueError('PPP dataset factory currently supports truckscenes only')
    if config.data.load_text_feat or config.data.load_vae_feat:
        raise ValueError('PPP cannot load precomputed text/VAE features')
    resolution = kwargs.get('resolution', config.data.image_size)
    if resolution != config.data.image_size:
        raise ValueError('Dataset and requested resolution must agree')
    norm = NormalizationConfig(**{k: extra[k] for k in vars(NormalizationConfig())})
    trucksc = TruckScenes(extra['dataset_version'], str(Path(config.data.dataset_dir).expanduser()), verbose=False)
    adapter = TruckScenesAdapter(trucksc, config=DatasetConfig(
        name='truckscenes', camera_views=extra['camera_views'],
        reference_sensor=TRUCKSCENES_REFERENCE_SENSOR, normalization=norm,
        data_root=trucksc.dataroot, version=trucksc.version))
    return RadarGenPPPDataset(adapter, ppp_data_dir=config.data.ppp_data_dir,
                             split=extra['eval_split'], resolution=resolution,
                             processing_signature=extra.get('ppp_processing_signature'),
                             scene_tokens=extra.get('ppp_scene_tokens'))
