"""Versioned JSON inventories prepared by full PPP metadata/header validation.

Manifests are trusted preparation artifacts, not signed proofs of validation.
Their digest detects accidental edits and binds checkpoints to their index.
Fast loading checks structure/contract and scene completion fingerprints, not
all sample files. Immutable HDF5 data and adapter metadata are required.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tempfile

from radargen.core.normalization import NormalizationConfig
from radargen.training.radargen_ppp_dataset import CONDITIONS, RadarGenPPPDataset
from radargen.training.startup_logging import startup_log
from jsc_jupiter.hdf5_samples import PPP_POLICY, SCHEMA_VERSION

FORMAT_VERSION = 1
VALIDATION_VERSION = 1  # Bump when enumeration, metadata/header, or empty-frame rules change.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _text(value):
    return isinstance(value, str) and bool(value)


def _component(value):
    return _text(value) and re.fullmatch(r'[A-Za-z0-9_-]+', value) is not None


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def dataset_contract(config):
    """Only settings affecting dataset indexing/reading; no model/optimizer knobs."""
    # Import constants only; never instantiate the adapter/database on this path.
    from radargen.datasets.truckscenes.config import TRUCKSCENES_REFERENCE_SENSOR, TRUCKSCENES_RADAR_SENSORS
    extra = config.data.extra
    _require(isinstance(extra, dict), 'data.extra must be a dictionary')
    _require(extra.get('dataset_name') == 'truckscenes' and
             extra.get('dataset_version') == 'v1.2-trainval', 'Unsupported PPP dataset/version')
    _require(extra.get('eval_split') in ('train', 'val'), 'PPP split must be train or val')
    _require(config.data.type == 'RadarGenPPPDatasetWrapper' and config.data.image_size == 512,
             'PPP requires RadarGenPPPDatasetWrapper at resolution 512')
    _require(not config.data.load_text_feat and not config.data.load_vae_feat,
             'PPP cannot load precomputed text/VAE features')
    _require(_text(config.data.dataset_dir) and _text(config.data.ppp_data_dir), 'Dataset roots are required')
    _require(_sha(extra.get('ppp_processing_signature')), 'Verified processing SHA256 required')
    cameras = extra.get('camera_views')
    _require(isinstance(cameras, list) and bool(cameras) and all(_component(c) for c in cameras)
             and len(set(cameras)) == len(cameras), 'Distinct camera_views are required')
    scenes = extra.get('ppp_scene_tokens')
    _require(scenes is None or (isinstance(scenes, list) and bool(scenes) and
             all(_component(s) for s in scenes) and len(set(scenes)) == len(scenes)), 'Invalid scene selection')
    indices = extra.get('ppp_subset_indices')
    _require(indices is None or (isinstance(indices, list) and bool(indices) and
             all(_integer(i) for i in indices) and len(set(indices)) == len(indices)), 'Invalid sample selection')
    normalization = vars(NormalizationConfig(**{k: extra[k] for k in vars(NormalizationConfig())}))
    # Reject nonfinite JSON and invalid geometry even for direct factory callers.
    _canonical(normalization)
    _require(all(type(v) in (int, float) for v in normalization.values()) and
             normalization['coordinate_range'] > 0 and normalization['camera_freq'] > 0 and
             normalization['rcs_min'] < normalization['rcs_max'] and
             normalization['doppler_min'] < normalization['doppler_max'], 'Invalid normalization')
    return dict(dataset_root=str(Path(config.data.dataset_dir).expanduser().resolve()),
                ppp_root=str(Path(config.data.ppp_data_dir).expanduser().resolve()),
                dataset_name=extra['dataset_name'], dataset_version=extra['dataset_version'],
                split=extra['eval_split'], processing_signature=extra['ppp_processing_signature'],
                schema_version=SCHEMA_VERSION, validation_version=VALIDATION_VERSION,
                filtering_policy=PPP_POLICY, resolution=config.data.image_size,
                validation_scope='exact_inventory_completion_metadata_consumed_headers_no_payload_scan',
                normalization=normalization, camera_order=cameras,
                conditioning_order=[list(pair) for pair in CONDITIONS],
                reference_sensor=TRUCKSCENES_REFERENCE_SENSOR, radar_order=TRUCKSCENES_RADAR_SENSORS,
                illuminated_only=True, scene_tokens=None if scenes is None else sorted(scenes),
                subset_indices=indices, transform='default_train_from_np',
                xy_frame='RadarGen_vehicle_flat_up', xy_units='m', rcs_units='dBsm',
                doppler_units='m/s', nsweeps=1)


def preparation_command(config_path, *, rebuild=False):
    return ('python -u scripts/prepare_ppp_manifest.py --config_path ' +
            shlex.quote(str(config_path)) + (' --rebuild' if rebuild else ''))


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _parse_json(text):
    return json.loads(text, object_pairs_hook=_json_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Invalid JSON number: {value}')))


def _read_json(path):
    return _parse_json(Path(path).read_text())


def _validate_document(document, contract):
    """Reject incomplete/unchecked index shapes before constructing any dataset."""
    _require(isinstance(document, dict) and set(document) == {'format', 'version', 'identity', 'body'},
             'Malformed manifest envelope')
    _require(document['format'] == 'radargen_ppp_manifest' and type(document['version']) is int
             and document['version'] == FORMAT_VERSION, 'Unsupported manifest format/version')
    body = document['body']
    _require(isinstance(body, dict) and set(body) == {'contract', 'scenes', 'samples', 'sample_count', 'empty_frame_count'},
             'Malformed manifest body')
    _require(_sha(document['identity']) and document['identity'] == _digest(body), 'Manifest identity checksum mismatch')
    _require(body['contract'] == contract and _canonical(body['contract']) == _canonical(contract),
             'Manifest dataset contract differs from requested configuration')
    _require(type(body['empty_frame_count']) is int and body['empty_frame_count'] == 0,
             'Manifest must certify no empty frames')
    scenes, samples = body['scenes'], body['samples']
    _require(isinstance(scenes, list) and bool(scenes) and isinstance(samples, list) and bool(samples),
             'Manifest must contain scenes and samples')
    _require(_integer(body['sample_count'], 1) and body['sample_count'] == len(samples), 'Sample count mismatch')
    scene_keys = {'scene_token', 'expected_frames', 'expected_pairs', 'inventory_count',
                  'completion_sha256', 'completion_size', 'completion_mtime_ns'}
    seen, offset = set(), 0
    for scene in scenes:
        _require(isinstance(scene, dict) and set(scene) == scene_keys, 'Malformed scene record')
        token, pairs = scene['scene_token'], scene['expected_pairs']
        _require(_component(token) and token not in seen, 'Invalid/duplicate scene token')
        seen.add(token)
        _require(_integer(pairs, 1) and _integer(scene['expected_frames'], 2) and
                 scene['expected_frames'] == pairs+1 and _integer(scene['inventory_count'], 2) and
                 scene['inventory_count'] == pairs+1, 'Scene inventory count mismatch')
        _require(_sha(scene['completion_sha256']) and _integer(scene['completion_size'], 1) and
                 _integer(scene['completion_mtime_ns']), 'Invalid completion fingerprint')
        _require(offset+pairs <= len(samples), 'Scene samples missing')
        for i, record in enumerate(samples[offset:offset+pairs]):
            _require(isinstance(record, dict) and set(record) == {'path', 'expected', 'size', 'mtime_ns', 'target_count'},
                     'Malformed sample record')
            _require(record['path'] == f'{token}/sample_{i:06d}.h5', 'Invalid sample path/order')
            _require(_integer(record['size'], 1) and _integer(record['mtime_ns']) and
                     _integer(record['target_count'], 1), 'Invalid sample fingerprint/target count')
            _validate_expected(record['expected'], contract, token, i)
        offset += pairs
    _require(offset == len(samples), 'Unclaimed samples in manifest')
    _require(contract['scene_tokens'] is None or seen == set(contract['scene_tokens']), 'Scene selection mismatch')
    _require(contract['subset_indices'] is None or max(contract['subset_indices']) < len(samples), 'Subset index outside inventory')
    return body


def _validate_expected(expected, contract, scene, index):
    scalars = dict(scene_token=scene, frame_index=index, split=contract['split'],
                   dataset_version=contract['dataset_version'], reference_sensor=contract['reference_sensor'])
    keys = set(scalars) | {'sample_token', 'reference_sample_data_token', 'reference_timestamp_us',
                          'reference_is_key_frame', 'camera_names', 'radar_names'}
    for prefix in ('camera_t0', 'camera_t1', 'radar_t0'):
        keys.update(f'{prefix}_{suffix}' for suffix in ('sample_data_tokens', 'timestamps_us', 'sample_tokens'))
    _require(isinstance(expected, dict) and set(expected) == keys, 'Malformed expected metadata')
    _require(all(expected[k] == v and type(expected[k]) is type(v) for k, v in scalars.items()),
             'Expected metadata identity/order mismatch')
    _require(_text(expected['sample_token']) and _text(expected['reference_sample_data_token']) and
             _integer(expected['reference_timestamp_us']) and type(expected['reference_is_key_frame']) is bool,
             'Invalid reference metadata')
    _require(expected['camera_names'] == contract['camera_order'], 'Camera ordering mismatch')
    radars = expected['radar_names']
    _require(isinstance(radars, list) and all(_text(r) for r in radars) and
             radars == [r for r in contract['radar_order'] if r in radars], 'Radar ordering mismatch')
    for prefix in ('camera_t0', 'camera_t1', 'radar_t0'):
        count = len(radars) if prefix == 'radar_t0' else len(contract['camera_order'])
        for suffix in ('sample_data_tokens', 'timestamps_us', 'sample_tokens'):
            values = expected[f'{prefix}_{suffix}']
            check = _integer if suffix == 'timestamps_us' else _text
            _require(isinstance(values, list) and len(values) == count and all(check(v) for v in values),
                     f'Invalid {prefix}_{suffix}')


def _check_completions(body):
    """Small per-scene reads; no HDF5 opens, sample stats, or inventory enumeration."""
    root = Path(body['contract']['ppp_root'])
    offset = 0
    for scene in body['scenes']:
        directory = root / scene['scene_token']
        path = directory / '_SUCCESS.json'
        _require(not directory.is_symlink() and not path.is_symlink(), 'Linked scene/completion record')
        stat = path.stat()
        marker_bytes = path.read_bytes()
        _require((stat.st_size, stat.st_mtime_ns) == (scene['completion_size'], scene['completion_mtime_ns']) and
                 hashlib.sha256(marker_bytes).hexdigest() == scene['completion_sha256'],
                 f'{path}: completion record changed')
        marker = _parse_json(marker_bytes)
        pairs = scene['expected_pairs']
        _require(isinstance(marker, dict) and all(marker.get(k) == v for k, v in dict(
                 scene=scene['scene_token'], signature=body['contract']['processing_signature'],
                 expected_frames=scene['expected_frames'], expected_pairs=pairs).items()), 'Completion contract mismatch')
        sizes = {Path(r['path']).name: r['size'] for r in body['samples'][offset:offset+pairs]}
        _require(marker.get('files') == sizes, 'Completion inventory differs from manifest')
        offset += pairs


def prepare_ppp_manifest(config, *, rebuild=False):
    """Run existing full validation and atomically publish only on success."""
    from radargen.training.radargen_ppp_dataset import validate_radargen_ppp_dataset_from_config
    contract = dataset_contract(config)
    destination = getattr(config.data, 'ppp_manifest_path', None)
    _require(_text(destination), 'Set data.ppp_manifest_path before preparation')
    destination = Path(destination).expanduser().absolute()
    _require(not destination.is_symlink(), 'Manifest destination must not be a symlink')
    _require(not destination.resolve().is_relative_to(Path(contract['ppp_root'])),
             'Store the manifest outside the immutable HDF5 root')
    if destination.exists() and not rebuild:
        raise FileExistsError(f'{destination} already exists; use --rebuild to revalidate and replace explicitly')
    dataset = validate_radargen_ppp_dataset_from_config(config)
    root = Path(contract['ppp_root'])
    samples = [dict(path=str(path.relative_to(root)), expected=expected, size=size,
                    mtime_ns=mtime, target_count=count)
               for (path, expected, size, mtime), count in zip(dataset.frame_index, dataset.target_counts)]
    body = dict(contract=contract, scenes=dataset.scene_records, samples=samples,
                sample_count=len(samples), empty_frame_count=dataset.empty_frame_count)
    # Normalize numpy scalars/arrays from adapter records into non-executable JSON.
    def convert(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(f'Unsupported manifest value: {type(value).__name__}')
    body = json.loads(json.dumps(body, default=convert, allow_nan=False))
    document = dict(format='radargen_ppp_manifest', version=FORMAT_VERSION, identity=_digest(body), body=body)
    _validate_document(document, contract)
    _check_completions(body)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_canonical(document) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if rebuild:
            os.replace(temporary, destination)
        else:
            # Exclusive atomic publication: never overwrite a concurrently created manifest.
            os.link(temporary, destination)
        startup_log('manifest published', path=destination, identity=document['identity'], samples=len(samples))
    finally:
        Path(temporary).unlink(missing_ok=True)
    return document['identity']


def load_ppp_manifest(config, *, config_path='configs/RadarGen_PPP_4GPU_utilization.yaml'):
    """Load identical validated index on each rank, without the TruckScenes database."""
    destination = getattr(config.data, 'ppp_manifest_path', None)
    startup_log('manifest loading begin', path=destination)
    try:
        _require(_text(destination), 'data.ppp_manifest_path is required')
        contract = dataset_contract(config)
        document = _read_json(Path(destination).expanduser())
        body = _validate_document(document, contract)
        _check_completions(body)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError(f'PPP manifest unavailable/incompatible: {error}. '
                         f'Prepare with {preparation_command(config_path, rebuild=True)} '
                         '(use the requested training configuration; omit --rebuild for first creation).') from error
    dataset = RadarGenPPPDataset.__new__(RadarGenPPPDataset)
    dataset.resolution = contract['resolution']
    dataset.point_limit = contract['normalization']['coordinate_range']
    dataset.normalization = contract['normalization']
    from diffusion.data.transforms import get_transform
    dataset.map_transform = get_transform(contract['transform'], dataset.resolution)
    root = Path(contract['ppp_root'])
    dataset.frame_index = [(root / r['path'], r['expected'], r['size'], r['mtime_ns']) for r in body['samples']]
    dataset.target_counts = [r['target_count'] for r in body['samples']]
    dataset.scene_records = body['scenes']
    dataset.empty_frame_count = 0
    dataset.ori_imgs_nums = len(dataset.frame_index)
    dataset.manifest_identity = dict(format_version=FORMAT_VERSION, validation_version=VALIDATION_VERSION,
                                     sha256=document['identity'])
    startup_log('manifest loading end', identity=document['identity'], samples=len(dataset))
    return dataset
