"""Synthetic CPU manifest tests; no pretrained assets or production scan."""

import copy
import datetime
import json
import importlib.abc
import runpy
import sys
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import pyrallis
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from diffusion.utils.config import SanaConfig
from diffusion.data.wids.wids import DistributedRangedSampler
from jsc_jupiter.hdf5_samples import _store, pair_metadata, PPP_POLICY, SCHEMA_VERSION
from radargen.core.normalization import NormalizationConfig
from radargen.training.ppp_manifest import (prepare_ppp_manifest, load_ppp_manifest,
                                          _digest)
from radargen.training.radargen_ppp_dataset import (build_radargen_ppp_dataset_from_config, validate_radargen_ppp_dataset_from_config)
from radargen.training.ppp_checkpoint import load_ppp_checkpoint


class SyntheticAdapter:
    normalization = NormalizationConfig()
    reference_sensor = 'LIDAR_LEFT'
    camera_views = ['CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_FRONT']

    def __init__(self):
        self.trucksc = self
        self.version = 'v1.2-trainval'
        self.dataroot = 'unused'

    def get(self, table, token):
        return dict(sample_token=token, timestamp=123, is_key_frame=True)

    def iter_scenes(self, split, filter_fn):
        # Non-lexical order proves the manifest preserves adapter order.
        for scene in ('scene_b', 'scene_a'):
            if filter_fn(scene):
                yield scene, [dict(LIDAR_LEFT=f'{scene}_{i}', CAMERA_RIGHT_FRONT=f'{scene}_{i}',
                                  CAMERA_LEFT_FRONT=f'{scene}_{i}_left') for i in range(3)]


def distributed_read(config, rendezvous, rank, queue):
    torch.set_num_threads(1)
    os.environ['RANK'] = str(rank)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=60))
    try:
        # Startup must not fall back to adapter validation or HDF5 opens.
        with patch('h5py.File', side_effect=AssertionError('startup opened HDF5')), \
             patch('truckscenes.TruckScenes', side_effect=AssertionError('startup loaded database')):
            dataset = load_ppp_manifest(config)
        sampler = DistributedRangedSampler(dataset, rank=rank, num_replicas=2)
        from accelerate.data_loader import prepare_data_loader
        loader = prepare_data_loader(DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0),
                                     device=torch.device('cpu'), num_processes=1, process_index=0,
                                     put_on_device=True, rng_types=[])
        frames = [(b['data_info']['scene_token'][0], b['data_info']['frame_idx'].item()) for b in loader]
        queue.put((rank, dataset.manifest_identity, list(sampler), frames))
    finally:
        dist.destroy_process_group()


class ManifestChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
            self.config = pyrallis.load(SanaConfig, stream)
        self.config.data.type = 'RadarGenPPPDatasetWrapper'
        self.config.data.dataset_dir = str(self.root/'raw')
        self.config.data.ppp_data_dir = str(self.root/'hdf5')
        self.config.data.ppp_manifest_path = str(self.root/'manifest.json')
        self.config.data.extra = dict(dataset_name='truckscenes', dataset_version='v1.2-trainval',
            eval_split='train', ppp_scene_tokens=None, ppp_processing_signature='a'*64,
            camera_views=SyntheticAdapter.camera_views, **vars(NormalizationConfig()))
        self.adapter = SyntheticAdapter()
        for scene, frames in self.adapter.iter_scenes('train', lambda _: True):
            directory = Path(self.config.data.ppp_data_dir)/scene
            directory.mkdir(parents=True)
            for i in range(2):
                path = directory/f'sample_{i:06d}.h5'
                expected = dict(pair_metadata(self.adapter, frames[i], frames[i+1], 'train'),
                                scene_token=scene, frame_index=i)
                with h5py.File(path, 'w') as handle:
                    meta = handle.create_group('metadata')
                    values = dict(expected, schema_version=SCHEMA_VERSION, ppp_filtering_policy=PPP_POLICY,
                        map_resolution=512, point_limit=50., conditioning_coordinate_range=50., nsweeps=1,
                        range_filter='abs(x) < point_limit AND abs(y) < point_limit',
                        frustum_camera_views=self.adapter.camera_views,
                        normalization_json=json.dumps(vars(self.adapter.normalization)),
                        configuration_json=json.dumps(dict(dataset_version=self.adapter.version,
                            resolution=512, point_limit=50., nsweeps=1, camera_views=self.adapter.camera_views,
                            filter_camera_views=self.adapter.camera_views)))
                    for key, value in values.items():
                        _store(meta, key, value)
                    cond = handle.create_group('conditioning')
                    for name in ('appearance', 'semantics', 'velocity'):
                        cond.create_dataset(name, data=np.full((512,512,3), i+1, np.uint8))
                    targets = handle.create_group('ppp_targets')
                    for key, value in dict(xy_frame='RadarGen_vehicle_flat_up', xy_units='m', rcs_units='dBsm',
                        doppler_units='m/s', doppler_definition='sensor_velocity_norm_times_sign_velocity_dot_sensor_position').items():
                        targets.attrs[key] = value
                    targets.create_dataset('xy', data=np.array([[i, 0]], np.float32))
                    targets.create_dataset('rcs', data=np.array([1], np.float32))
                    targets.create_dataset('doppler', data=np.array([-2], np.float32))
            self.refresh_marker(scene)

    def refresh_marker(self, scene):
        directory = Path(self.config.data.ppp_data_dir)/scene
        marker = dict(scene=scene, signature='a'*64, expected_frames=3, expected_pairs=2,
                      files={p.name:p.stat().st_size for p in directory.glob('*.h5')})
        (directory/'_SUCCESS.json').write_text(json.dumps(marker))

    def fresh(self):
        # Only database/enumerator are synthetic; all inventory/header rules are real.
        with patch('truckscenes.TruckScenes', return_value=self.adapter), \
             patch('radargen.datasets.truckscenes.TruckScenesAdapter', return_value=self.adapter):
            return validate_radargen_ppp_dataset_from_config(self.config)

    def prepare(self, rebuild=False):
        with patch('truckscenes.TruckScenes', return_value=self.adapter), \
             patch('radargen.datasets.truckscenes.TruckScenesAdapter', return_value=self.adapter):
            return prepare_ppp_manifest(self.config, rebuild=rebuild)

    def read_document(self):
        return json.loads(Path(self.config.data.ppp_manifest_path).read_text())

    def write_document(self, document, recompute=True):
        if recompute:
            document['identity'] = _digest(document['body'])
        Path(self.config.data.ppp_manifest_path).write_text(json.dumps(document))

    def test_equivalence_and_no_scan_startup(self):
        handles = h5py.h5f.get_obj_count()
        fresh = self.fresh()
        identity = self.prepare()
        with patch('h5py.File', side_effect=AssertionError('HDF5 opened at startup')), \
             patch('truckscenes.TruckScenes', side_effect=AssertionError('database instantiated')), \
             patch('radargen.datasets.truckscenes.TruckScenesAdapter', side_effect=AssertionError('adapter instantiated')), \
             patch('pathlib.Path.iterdir', side_effect=AssertionError('inventory enumerated')):
            cached = build_radargen_ppp_dataset_from_config(self.config)
        self.assertEqual(len(cached), len(fresh))
        self.assertEqual(cached.frame_index, fresh.frame_index)
        self.assertEqual(cached.manifest_identity['sha256'], identity)
        self.assertEqual(cached.target_counts, fresh.target_counts)
        self.assertEqual(cached.scene_records, fresh.scene_records)
        for i in range(len(fresh)):
            left, right = fresh[i], cached[i]
            for key in left:
                if key != 'data_info':
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual(left['data_info']['scene_token'], right['data_info']['scene_token'])
            self.assertEqual(left['data_info']['frame_idx'], right['data_info']['frame_idx'])
        self.assertEqual(h5py.h5f.get_obj_count(), handles)

    def test_contract_compatibility_and_selection(self):
        identity = self.prepare()
        changed = copy.deepcopy(self.config)
        changed.train.train_batch_size = 99
        changed.train.optimizer['lr'] = 0.8
        changed.train.num_epochs = 1
        changed.train.training_hours = 0.1
        self.assertEqual(load_ppp_manifest(changed).manifest_identity['sha256'], identity)
        for field, value in [('eval_split', 'val'), ('dataset_version', 'other'),
                ('ppp_processing_signature', 'b'*64), ('coordinate_range', 60.),
                ('camera_freq', 2.), ('camera_views', list(reversed(self.adapter.camera_views))),
                ('ppp_scene_tokens', ['scene_b']), ('ppp_subset_indices', [0])]:
            bad = copy.deepcopy(self.config)
            bad.data.extra[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'prepare_ppp_manifest.py'):
                load_ppp_manifest(bad, config_path='custom.yaml')
        for field in ('dataset_dir', 'ppp_data_dir'):
            bad = copy.deepcopy(self.config)
            setattr(bad.data, field, str(self.root/'other'))
            with self.assertRaisesRegex(ValueError, 'contract'):
                load_ppp_manifest(bad)
        self.config.data.extra['ppp_scene_tokens'] = ['scene_a']
        self.config.data.extra['ppp_subset_indices'] = [1]
        self.prepare(rebuild=True)
        selected = load_ppp_manifest(self.config)
        self.assertEqual(len(selected), 2)  # Training applies sample Subset after loading.
        self.assertEqual({r[1]['scene_token'] for r in selected.frame_index}, {'scene_a'})

    def test_malformed_and_missing_manifests(self):
        with self.assertRaisesRegex(ValueError, 'custom.yaml'):
            load_ppp_manifest(self.config, config_path='custom.yaml')
        self.prepare()
        original = self.read_document()
        mutations = [
            lambda d: d.update(version=True),
            lambda d: d['body'].update(sample_count=99),
            lambda d: d['body'].update(empty_frame_count=1),
            lambda d: d['body']['samples'][0].update(path='../sample.h5'),
            lambda d: d['body']['samples'][0].update(target_count=0),
            lambda d: d['body']['samples'][0].update(size=True),
            lambda d: d['body']['samples'][0]['expected'].update(frame_index=1),
            lambda d: d['body']['samples'][0]['expected'].pop('camera_t1_timestamps_us'),
            lambda d: d['body']['samples'][0]['expected'].update(reference_timestamp_us='123'),
            lambda d: d['body']['samples'][0]['expected'].update(radar_names=['UNKNOWN']),
            lambda d: d['body']['scenes'][0].update(expected_pairs=9),
            lambda d: d['body']['scenes'].reverse(),
            lambda d: d['body']['samples'].reverse(),
            lambda d: d['body']['scenes'].append(d['body']['scenes'][0]),
        ]
        for change in mutations:
            bad = copy.deepcopy(original); change(bad)
            self.write_document(bad)
            with self.assertRaisesRegex(ValueError, 'manifest unavailable/incompatible'):
                load_ppp_manifest(self.config)
        self.write_document(original)
        bad = copy.deepcopy(original); bad['identity'] = '0'*64
        self.write_document(bad, recompute=False)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_ppp_manifest(self.config)
        for malformed in ('{"format":1,"format":2}', '{}', 'not JSON', '{"x":NaN}'):
            Path(self.config.data.ppp_manifest_path).write_text(malformed)
            with self.assertRaises(ValueError):
                load_ppp_manifest(self.config)

    def test_rebuild_atomic_failure_preservation(self):
        identity = self.prepare()
        destination = Path(self.config.data.ppp_manifest_path)
        original = destination.read_bytes()
        with patch('radargen.training.radargen_ppp_dataset.validate_radargen_ppp_dataset_from_config',
                   side_effect=AssertionError('must fail before scanning')):
            with self.assertRaisesRegex(FileExistsError, '--rebuild'):
                prepare_ppp_manifest(self.config)
        self.assertEqual(self.prepare(rebuild=True), identity)
        with patch('radargen.training.radargen_ppp_dataset.validate_radargen_ppp_dataset_from_config',
                   side_effect=ValueError('synthetic validation failure')):
            with self.assertRaisesRegex(ValueError, 'validation failure'):
                prepare_ppp_manifest(self.config, rebuild=True)
        self.assertEqual(destination.read_bytes(), original)
        with patch('radargen.training.ppp_manifest.os.replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                self.prepare(rebuild=True)
        self.assertEqual(destination.read_bytes(), original)
        with patch('radargen.training.ppp_manifest.os.fsync', side_effect=OSError('write failed')):
            with self.assertRaisesRegex(OSError, 'write failed'):
                self.prepare(rebuild=True)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(list(destination.parent.glob('.manifest.json.*.tmp')), [])

    def test_accessed_files_and_completion_changes(self):
        self.prepare(); cached = load_ppp_manifest(self.config)
        path = cached.frame_index[0][0]
        # Fast startup deliberately does not stat every sample.
        stat = path.stat(); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1))
        load_ppp_manifest(self.config)
        with self.assertRaisesRegex(ValueError, 'changed'):
            cached[0]
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        # Existing target payload checks still run when size/mtime are preserved.
        with h5py.File(path, 'r+') as handle:
            handle['ppp_targets/xy'][0, 0] = np.nan
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'finite floating-point'):
            cached[0]
        with h5py.File(path, 'r+') as handle:
            handle['ppp_targets/xy'][0, 0] = 0.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        # Metadata/item validation still runs even if size/mtime are preserved.
        with h5py.File(path, 'r+') as handle:
            handle['metadata/split'][()] = 'val'
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'metadata/split mismatch'):
            cached[0]
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            cached[0]
        marker = Path(self.config.data.ppp_data_dir)/'scene_a'/'_SUCCESS.json'
        marker.write_text(marker.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'completion record changed'):
            load_ppp_manifest(self.config)

    def test_preserves_fresh_inventory_and_empty_frame_checks(self):
        self.prepare(); original = Path(self.config.data.ppp_manifest_path).read_bytes()
        extra = Path(self.config.data.ppp_data_dir)/'scene_b'/'unexpected.h5'
        extra.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'exact sample inventory mismatch'):
            self.prepare(rebuild=True)
        extra.unlink()
        marker = Path(self.config.data.ppp_data_dir)/'scene_b'/'_SUCCESS.json'
        saved = marker.read_text(); bad = json.loads(saved); bad['files']['sample_000000.h5'] += 1
        marker.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'file sizes/inventory mismatch'):
            self.prepare(rebuild=True)
        marker.write_text(saved)
        path = marker.parent/'sample_000000.h5'
        with h5py.File(path, 'r+') as handle:
            del handle['ppp_targets/rcs']
            handle['ppp_targets'].create_dataset('rcs', data=np.array([1], np.float64))
        self.refresh_marker('scene_b')
        with self.assertRaisesRegex(ValueError, 'ppp_targets/rcs shape/dtype mismatch'):
            self.prepare(rebuild=True)
        self.assertEqual(Path(self.config.data.ppp_manifest_path).read_bytes(), original)
        with h5py.File(path, 'r+') as handle:
            for name, shape in (('xy', (0,2)), ('rcs', (0,)), ('doppler', (0,))):
                del handle[f'ppp_targets/{name}']
                handle['ppp_targets'].create_dataset(name, shape=shape, dtype=np.float32)
        self.refresh_marker('scene_b')
        with self.assertRaisesRegex(ValueError, 'empty PPP frames'):
            self.prepare(rebuild=True)
        self.assertEqual(Path(self.config.data.ppp_manifest_path).read_bytes(), original)

    def test_distributed_reading_and_sampler_without_double_sharding(self):
        self.prepare()
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        processes = [context.Process(target=distributed_read,
            args=(self.config, str(self.root/'rendezvous'), rank, queue)) for rank in range(2)]
        for process in processes:
            process.start()
        try:
            results = sorted([queue.get(timeout=90), queue.get(timeout=90)])
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(results[0][2], [0, 1])
            self.assertEqual(results[1][2], [2, 3])
            self.assertEqual(results[0][3], [('scene_b', 0), ('scene_b', 1)])
            self.assertEqual(results[1][3], [('scene_a', 0), ('scene_a', 1)])
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate(); process.join()
            queue.close()
        dataset = load_ppp_manifest(self.config)
        with patch('diffusion.data.wids.wids.dist.is_initialized', return_value=True), \
             patch('diffusion.data.wids.wids.dist.get_rank', return_value=0):
            parts = [list(DistributedRangedSampler(dataset, num_replicas=4, rank=r)) for r in range(4)]
        self.assertEqual(parts, [[0], [1], [2], [3]])

    def test_preparation_cli_is_cpu_only(self):
        class NoModels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'transformers', 'diffusers', 'huggingface_hub', 'accelerate'}:
                    raise AssertionError(f'Unexpected model/runtime import: {fullname}')
        config_path = self.root/'synthetic.yaml'
        with config_path.open('w') as stream:
            pyrallis.dump(self.config, stream)
        guard = NoModels()
        sys.meta_path.insert(0, guard)
        try:
            with patch('sys.argv', ['prepare_ppp_manifest.py', '--config_path', str(config_path)]), \
                 patch('truckscenes.TruckScenes', return_value=self.adapter), \
                 patch('radargen.datasets.truckscenes.TruckScenesAdapter', return_value=self.adapter), \
                 patch('torch.cuda._lazy_init', side_effect=AssertionError('CUDA initialized')), \
                 patch('radargen.datasets.truckscenes.reader.load_radar_multisweep',
                       side_effect=AssertionError('raw radar loaded')), \
                 patch('h5py.File', wraps=h5py.File) as opened:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path('scripts/prepare_ppp_manifest.py', run_name='__main__')
                self.assertEqual(result.exception.code, 0)
                self.assertEqual(len(opened.call_args_list), 4)
                self.assertTrue(all(call.args[1] == 'r' for call in opened.call_args_list))
        finally:
            sys.meta_path.remove(guard)
        self.assertEqual(len(load_ppp_manifest(self.config)), 4)

    def test_checkpoint_manifest_identity_and_legacy_rejection(self):
        self.prepare()
        identity = load_ppp_manifest(self.config).manifest_identity
        model = torch.nn.Linear(1, 1)
        checkpoint = self.root/'checkpoint.pth'
        contract = dict(manifest_identity=identity)
        state = dict(ppp_training=dict(version=1, contract=contract, progress={}),
                     state_dict=model.state_dict(), rank_rng_states=[{}], epoch=0)
        torch.save(state, checkpoint)
        load_ppp_checkpoint(checkpoint, model, contract=contract)
        changed = dict(manifest_identity=dict(identity, sha256='f'*64))
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            load_ppp_checkpoint(checkpoint, model, contract=changed)
        state['ppp_training']['contract'] = {}
        torch.save(state, checkpoint)
        with self.assertRaisesRegex(ValueError, 'Legacy PPP checkpoint has no validated manifest identity'):
            load_ppp_checkpoint(checkpoint, model, contract=contract)
        # Legacy standalone inspection remains supported when no manifest contract is requested.
        load_ppp_checkpoint(checkpoint, model)


if __name__ == '__main__':
    unittest.main()
