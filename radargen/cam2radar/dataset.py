"""Manifest-indexed raw observations; no conditioning maps or HDF5 reads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CameraRadarDataset(Dataset):
    """Read exact sensor observations from a prepared PPP training inventory.

    Images remain full RGB uint8 (4,3,H,W); normalization/padding belongs to
    the model input path. Geometry is metric, with reference-to-camera matrices
    expressed relative to the radar vehicle-flat-up frame. Adapter creation is
    process-local so DataLoader workers do not inherit a live devkit handle.
    """

    def __init__(self, manifest_path, dataset_dir):
        document = json.loads(Path(manifest_path).read_text())
        body = document['body']
        encoded = json.dumps(body, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')
        if (document['format'] != 'radargen_ppp_manifest'
                or document['version'] != 1
                or hashlib.sha256(encoded).hexdigest() != document['identity']):
            raise ValueError('Unsupported or modified PPP inventory')
        self.records = body['samples']
        self.contract = body['contract']
        self.dataset_dir = str(dataset_dir)
        self.manifest_identity = document['identity']
        self._adapter = None
        self._pid = None
        self.point_limit = self.contract['normalization']['coordinate_range']

    @classmethod
    def from_records(cls, records, dataset_dir, normalization):
        """Use identities from the existing validated evaluation inventory."""
        result = cls.__new__(cls)
        result.records = records
        result.contract = dict(normalization=normalization)
        result.dataset_dir = str(dataset_dir)
        result.manifest_identity = None
        result.point_limit = normalization['coordinate_range']
        result._adapter = result._pid = None
        return result

    def __len__(self):
        return len(self.records)

    def _get_adapter(self):
        if self._adapter is None or self._pid != os.getpid():
            from radargen.datasets.truckscenes.adapter import TruckScenesAdapter
            first = self.records[0]['expected']
            self._adapter = TruckScenesAdapter(
                dataset_dir=self.dataset_dir, dataset_version=first['dataset_version'],
                camera_views=first['camera_names'], reference_sensor=first['reference_sensor'])
            self._pid = os.getpid()
        return self._adapter

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_adapter=None, _pid=None)
        return state

    def __getitem__(self, index):
        from radargen.radar_maps.creator import load_filtered_radar_detections
        expected = self.records[index]['expected']
        adapter = self._get_adapter()
        frame = {expected['reference_sensor']: expected['reference_sample_data_token']}
        for prefix, names in [('camera_t0', expected['camera_names']),
                              ('radar_t0', expected['radar_names'])]:
            frame.update(zip(names, expected[f'{prefix}_sample_data_tokens']))
        for path in adapter.get_camera_image_paths(frame):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f'Raw camera observation missing: {path}. Non-keyframe sweeps may '
                    'require scene archive staging; do not substitute a keyframe.')
        images = np.stack(adapter.load_camera_images(frame))
        reference = expected['reference_sample_data_token']
        ref_to_bev = adapter.get_ref_to_vehicle_flat_up_transform(reference)
        camera_to_bev = np.stack([
            ref_to_bev @ transform for transform in adapter.get_camera_extrinsics(frame, reference)])
        radar = load_filtered_radar_detections(
            adapter, frame, point_limit=self.point_limit, nsweeps=1,
            filter_camera_views=expected['camera_names'])
        return dict(
            camera_images=torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
            camera_intrinsics=torch.from_numpy(np.stack(adapter.get_camera_intrinsics(frame))).float(),
            reference_to_camera=torch.from_numpy(np.linalg.inv(camera_to_bev)).float(),
            camera_timestamps_us=torch.tensor(expected['camera_t0_timestamps_us'], dtype=torch.int64),
            detections=torch.from_numpy(radar[:, [0, 1, 6, 7]].astype(np.float32)),
            point_limit=float(self.point_limit),
            data_info=dict(scene_token=expected['scene_token'], frame_idx=expected['frame_index'],
                           ref_sensor_token=reference,
                           img_hw=torch.tensor([512., 512.]), aspect_ratio=torch.tensor(1.)),
        )
