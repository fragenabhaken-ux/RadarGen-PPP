"""CPU-only production validation and launch-result rejection tests."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from jsc_jupiter.hdf5_4nodes_launch_check import verify_records
from jsc_jupiter.hdf5_4nodes_worker import select_scene, validate_scene_output
from jsc_jupiter.hdf5_samples import annotation_payload, pair_metadata, write_sample


class FixtureAdapter:
    camera_views = ["camera"]
    reference_sensor = "reference"

    def __init__(self):
        self.frames = [dict(camera=f"camera{i}", reference=f"reference{i}",
                            RADAR_LEFT_FRONT=f"radar{i}") for i in range(3)]
        self.records = {token: dict(sample_token=f"sample{i}", timestamp=i * 100000,
                                    is_key_frame=True)
                        for i, frame in enumerate(self.frames) for token in frame.values()}
        self.trucksc = SimpleNamespace(version="v1.2-trainval", get=self.get)

    def get(self, table, token):
        if table == "sample":
            return dict(anns=[], timestamp=0, prev="")
        return self.records[token]

    def get_camera_image_shapes(self, frame):
        return [(2, 3)]

    def iter_scenes(self, split, filter_fn):
        if split == "train" and filter_fn("scene"):
            yield "scene", self.frames


class SceneValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "scene"
        self.adapter = FixtureAdapter()
        self.split, self.frames = select_scene(self.adapter, "scene")

    def write(self, index):
        metadata = pair_metadata(self.adapter, *self.frames[index:index + 2], self.split)
        metadata["map_resolution"] = 4
        boxes = annotation_payload(self.adapter.trucksc, metadata["sample_token"])
        return write_sample(self.root, "scene", index,
                            [np.zeros((4, 4, 3), np.uint8)] * 3,
                            {"camera": np.ones((2, 3), np.float32)},
                            [np.ones((4, 4), dtype) for dtype in
                             (np.float32, np.float32, np.float64)],
                            np.ones((2, 8), np.float32), metadata, boxes)

    def validate(self, completed=False):
        return validate_scene_output(self.directory, "scene", "signature",
                                     self.adapter, self.frames, self.split, 4,
                                     completed=completed)

    def seal(self):
        marker = self.validate()
        self.marker_path = self.directory / "_SUCCESS.json"
        self.marker_path.write_text(json.dumps(marker))
        return marker

    def reject_restart(self, message):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, message):
            self.validate(completed=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_complete_new_scene_and_restart(self):
        for i in range(2):
            self.write(i)
        marker = self.seal()
        self.assertEqual(marker["expected_frames"], 3)
        self.assertEqual(marker["expected_pairs"], 2)
        self.assertEqual(self.validate(completed=True), marker)

    def test_contiguous_incomplete_scene_rejected_new_and_restart(self):
        path = self.write(0)
        with self.assertRaisesRegex(ValueError, "incomplete.*inventory"):
            self.validate()
        marker = dict(scene="scene", signature="signature", expected_frames=3,
                      expected_pairs=2, files={path.name: path.stat().st_size})
        (self.directory / "_SUCCESS.json").write_text(json.dumps(marker))
        self.reject_restart("incomplete.*inventory")

    def test_incompatible_signature_counts_and_legacy_marker(self):
        for i in range(2):
            self.write(i)
        original = self.seal()
        for changes in ({"signature": "other"}, {"scene": "other"},
                        {"expected_pairs": 1}, {"expected_frames": 2}):
            marker = dict(original, **changes)
            self.marker_path.write_text(json.dumps(marker))
            self.reject_restart("incompatible completion marker")
        legacy = {k: v for k, v in original.items()
                  if k not in ("expected_frames", "expected_pairs")}
        self.marker_path.write_text(json.dumps(legacy))
        self.reject_restart("incompatible completion marker")

    def test_nonfinite_writer_does_not_publish(self):
        metadata = pair_metadata(self.adapter, *self.frames[:2], self.split)
        metadata["map_resolution"] = 4
        boxes = annotation_payload(self.adapter.trucksc, metadata["sample_token"])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            write_sample(self.root, "scene", 0,
                         [np.zeros((4, 4, 3), np.uint8)] * 3,
                         {"camera": np.full((2, 3), np.nan, np.float32)},
                         [np.ones((4, 4), np.float32)] * 3,
                         np.ones((2, 8), np.float32), metadata, boxes)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_nonfinite_new_and_restart_rejected(self):
        for i in range(2):
            self.write(i)
        original = self.seal()
        for dataset in ("conditioning/depth/camera", "radargen_targets/rcs",
                        "ppp_targets/xy"):
            for value in (np.nan, np.inf):
                with h5py.File(self.directory / "sample_000000.h5", "r+") as handle:
                    handle[dataset][...] = value
                marker = dict(original, files={p.name: p.stat().st_size
                              for p in self.directory.glob("*.h5")})
                self.marker_path.write_text(json.dumps(marker))
                self.reject_restart("Nonfinite")
                self.marker_path.unlink()  # Only this test's temporary fixture.
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    self.validate()
                with h5py.File(self.directory / "sample_000000.h5", "r+") as handle:
                    handle[dataset][...] = 1
                self.marker_path.write_text(json.dumps(original))

    def test_metadata_mismatch_including_final_t1_rejected(self):
        for i in range(2):
            self.write(i)
        original = self.seal()
        path = self.directory / "sample_000001.h5"
        for field in ("camera_t0_sample_data_tokens", "camera_t1_sample_data_tokens",
                      "camera_t1_timestamps_us", "radar_t0_sample_data_tokens"):
            with h5py.File(path, "r+") as handle:
                ds = handle[f"metadata/{field}"]
                old = ds[()]
                ds[...] = -1 if ds.dtype.kind == "i" else ["wrong"]
            marker = dict(original, files={p.name: p.stat().st_size
                          for p in self.directory.glob("*.h5")})
            self.marker_path.write_text(json.dumps(marker))
            self.reject_restart(f"metadata/{field} mismatch")
            self.marker_path.unlink()
            with self.assertRaisesRegex(ValueError, f"metadata/{field} mismatch"):
                self.validate()
            with h5py.File(path, "r+") as handle:
                handle[f"metadata/{field}"][...] = old
            self.marker_path.write_text(json.dumps(original))

    def test_unexpected_temporary_file_rejected(self):
        for i in range(2):
            self.write(i)
        self.seal()
        (self.directory / ".sample.partial").write_text("leftover")
        self.reject_restart("unexpected inventory")


class LaunchVerificationTest(unittest.TestCase):
    def records(self):
        return [dict(rank=i, local_rank=i % 4, host=f"host{i // 4}",
                     visible_gpus=1, cuda_result=10, gpu_uuid=f"GPU-{i}",
                     allowed_cpus=list(range((i % 4) * 72, (i % 4 + 1) * 72)))
                for i in range(16)]

    def test_valid_placement(self):
        self.assertEqual(len(verify_records(self.records())), 4)

    def test_invalid_placements(self):
        cases = [("rank", 1), ("host", "host1"), ("visible_gpus", 4),
                 ("gpu_uuid", "GPU-1"), ("cuda_result", 0),
                 ("allowed_cpus", list(range(72, 144))),
                 ("allowed_cpus", list(range(71)))]
        for field, value in cases:
            records = copy.deepcopy(self.records())
            records[0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    verify_records(records)
        with self.assertRaises(ValueError):
            verify_records(self.records()[:-1])


if __name__ == "__main__":
    unittest.main()
