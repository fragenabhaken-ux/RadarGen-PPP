"""Compare selected raw Camera2Radar samples to the existing PPP targets.

Run as ``python -m scripts.check_cam2radar_data`` from the repository root.
HDF5 is used only as the reference in this diagnostic, not by the dataset.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from radargen.cam2radar.dataset import CameraRadarDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='output/ppp_manifests/truckscenes_train.json')
    parser.add_argument('--dataset-dir', default='/e/scratch/nxtaim-1/public/truckscenes/man-truckscenes')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    dataset = CameraRadarDataset(args.manifest, args.dataset_dir)
    for index in args.indices:
        sample = dataset[index]
        record = dataset.records[index]
        path = Path(dataset.contract['ppp_root']) / record['path']
        with h5py.File(path, 'r') as handle:
            stored = np.column_stack([handle[f'ppp_targets/{name}'][()]
                                      for name in ('xy', 'rcs', 'doppler')])
            stored_k = handle['metadata/camera_intrinsics'][()]
            camera_to_bev = handle['metadata/reference_to_flat_up'][()] @ handle['metadata/camera_to_reference'][()]
            np.testing.assert_allclose(sample['camera_intrinsics'].numpy(), stored_k, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(sample['reference_to_camera'].numpy(), np.linalg.inv(camera_to_bev),
                                       atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(sample['detections'].numpy(), stored, rtol=1e-5, atol=1e-5)
        print(json.dumps(dict(index=index, scene=record['expected']['scene_token'],
                              frame=record['expected']['frame_index'],
                              keyframe=record['expected']['reference_is_key_frame'],
                              image_shape=list(sample['camera_images'].shape),
                              detections=len(stored), targets_match=True, geometry_matches=True)))


if __name__ == '__main__':
    main()
