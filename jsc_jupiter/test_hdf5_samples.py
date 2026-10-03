"""CPU-only serialization, callback parity and staging lifecycle checks.

Run: python -m unittest jsc_jupiter.test_hdf5_samples -v
"""

import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from jsc_jupiter.create_hdf5_samples import staged_scene
from jsc_jupiter.hdf5_samples import annotation_payload, validate_sample, write_sample


class HDF5SamplesTest(unittest.TestCase):
    def test_depth_callback_parity_and_frame_forwarding(self):
        import torch
        import radargen.bev_condition_maps.core as core
        import radargen.bev_condition_maps.creator as creator
        from contextlib import ExitStack
        depth = dict(depth=torch.ones((2, 3)), points=torch.ones((2, 3, 3)))
        captured = []
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        kwargs = dict(camera_images_t0=[image], camera_images_t1=[image],
                      camera_intrinsics=[np.eye(3)], transform_fn=lambda outputs: np.zeros((6, 3)),
                      models=dict(depth=None, segmentation=None, segmentation_processor=None, flow=None),
                      device=torch.device("cpu"), dts=[0.1], resolution=4,
                      coordinate_range=50, doppler_min=-120, doppler_max=120)
        with ExitStack() as stack:
            stack.enter_context(patch.object(core, "get_depth", return_value=depth))
            stack.enter_context(patch.object(core, "get_depth_batched", return_value=[depth]))
            stack.enter_context(patch.object(core, "segment_image", return_value=image))
            stack.enter_context(patch.object(core, "segment_images_batched", return_value=[image]))
            stack.enter_context(patch.object(core, "radial_velocity_from_flow", return_value=(None, image)))
            stack.enter_context(patch.object(core, "radial_velocity_from_flow_batched", return_value=[(None, image)]))
            stack.enter_context(patch.object(core, "get_points_mask", return_value=np.ones(6, bool)))
            stack.enter_context(patch.object(core, "points_to_bev_map", return_value=np.zeros((4, 4, 3), np.uint8)))
            for batched in (False, True):
                original = core.create_bev_maps_from_camera_data(**kwargs, use_batched_inference=batched)
                observed = core.create_bev_maps_from_camera_data(**kwargs, use_batched_inference=batched, depth_callback=captured.append)
                for first, second in zip(original, observed):
                    np.testing.assert_array_equal(first, second)
        self.assertEqual(len(captured), 2)
        self.assertIs(captured[0][0]["depth"], depth["depth"])
        adapter = Mock(reference_sensor="ref", camera_views=["camera"])
        adapter.get_camera_timestamps.side_effect = [[0], [100000]]
        callback = Mock()
        with patch.object(creator, "create_bev_maps_from_camera_data", return_value=(image, image, image)) as compute:
            creator.create_bev_maps_for_frame(adapter, {"ref": "r"}, {"ref": "s"}, {}, torch.device("cpu"), depth_callback=callback)
        self.assertIs(compute.call_args.kwargs["depth_callback"], callback)

    def test_roundtrip_empty_and_nonempty_annotations(self):
        record = dict(token="ann", translation=[1, 2, 3], size=[2, 4, 3],
                      rotation=[1, 0, 0, 0], category_name="vehicle.truck",
                      instance_token="instance", visibility_token="3", attribute_tokens=["moving"])
        for records in ([], [record]):
            db = Mock()
            db.get.side_effect = lambda table, token: (
                dict(anns=[a["token"] for a in records], timestamp=123, prev="")
                if table == "sample" else record)
            boxes = annotation_payload(db, "sample")
            radar = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 9, -10]], dtype=np.float32)
            maps = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
            targets = [np.ones((4, 4), dtype=dtype) for dtype in (np.float32, np.float32, np.float64)]
            depth = {"camera": np.ones((2, 3), dtype=np.float32)}
            with tempfile.TemporaryDirectory() as directory:
                path = write_sample(directory, "scene", 100, maps, depth, targets, radar,
                                    dict(map_resolution=4, sample_token="sample"), boxes)
                validate_sample(path, "scene", 100, ["camera"], 4, {"camera": (2, 3)}, report=False)
                with h5py.File(path) as handle:
                    np.testing.assert_array_equal(handle["ppp_targets/doppler"][()], radar[:, 7])
                    self.assertEqual(handle["radargen_targets/doppler"].dtype, np.float64)
                    self.assertEqual(json.loads(handle["metadata/bounding_boxes/raw_annotations_json"].asstr()[()]), records)
                self.assertFalse(list(path.parent.glob("*.partial")))
                with self.assertRaises(FileExistsError):
                    write_sample(directory, "scene", 100, maps, depth, targets, radar,
                                 dict(map_resolution=4, sample_token="sample"), boxes)

    def test_failed_validation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_sample(directory, "scene", 0, [np.zeros((2, 2, 3))] * 3,
                             {"camera": np.ones((2, 2))}, [np.ones((2, 2))] * 3,
                             np.zeros((0, 8), np.float32),
                             dict(map_resolution=2, sample_token="sample"), {})
            self.assertEqual(list((Path(directory) / "scene").iterdir()), [])

    def test_staging_cleanup_and_existing_root_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permanent = root / "permanent"
            for name in ("samples", "v1.2-trainval"):
                (permanent / name).mkdir(parents=True)
            source = root / "source" / "sweeps"
            source.mkdir(parents=True)
            (source / "radar.pcd").write_text("fixture")
            archive = root / "scene.tar"
            with tarfile.open(archive, "w") as stream:
                stream.add(source, arcname="sweeps")
            subprocess.run(["zstd", "-q", str(archive), "-o", str(root / "scene.tar.zst")], check=True)
            staging = root / "staging"
            for fail in (False, True):
                try:
                    with staged_scene(root, staging, permanent, "scene", "v1.2-trainval") as staged:
                        self.assertEqual((staged / "sweeps/radar.pcd").read_text(), "fixture")
                        self.assertTrue((staged / "samples").is_symlink())
                        if fail:
                            raise RuntimeError("controlled failure")
                except RuntimeError:
                    if not fail:
                        raise
                self.assertFalse((staging / "scene").exists())
                self.assertTrue((permanent / "samples").is_dir())
            (staging / "scene").mkdir()
            with self.assertRaises(FileExistsError):
                with staged_scene(root, staging, permanent, "scene", "v1.2-trainval"):
                    pass
            self.assertTrue((staging / "scene").exists())

    def test_radar_callback_observes_metric_pre_dedup_and_cannot_mutate_maps(self):
        import torch
        import radargen.radar_maps.creator as creator
        radar = np.array([[1, 2, 0, 0, 0, 0, 5, -3], [1, 2, 0, 0, 0, 0, 9, 7]], dtype=np.float32)
        adapter = Mock(reference_sensor="ref")
        adapter.load_radar_pointcloud.side_effect = lambda *args, **kwargs: radar.copy()
        adapter.filter_by_camera_frustum.side_effect = lambda pcl, *args, **kwargs: pcl
        adapter.normalization = Mock(rcs_min=-20, rcs_max=66)
        captured = []

        def observer(pcl):
            captured.append(pcl.copy())
            pcl[:] = 0  # Mutation of the observer's copy cannot affect map creation.

        def fake_pd(xy, size, **kwargs):
            return xy.copy()

        with patch.object(creator, "create_radar_pd_map", side_effect=fake_pd), \
             patch.object(creator, "create_radar_rcs_map", side_effect=lambda xy, values, *a, **kw: values.copy()), \
             patch.object(creator, "create_radar_doppler_map", side_effect=lambda xy, values, *a: values.copy()):
            original = creator.create_radar_maps_for_frame(adapter, {"ref": "token"}, torch.device("cpu"))
            observed = creator.create_radar_maps_for_frame(adapter, {"ref": "token"}, torch.device("cpu"), radar_callback=observer)
        np.testing.assert_array_equal(captured[0], radar)
        for name in ("pd_map", "rcs_map", "doppler_map"):
            np.testing.assert_array_equal(getattr(original, name), getattr(observed, name))
        np.testing.assert_array_equal(observed.doppler_map, [7])


if __name__ == "__main__":
    unittest.main()
