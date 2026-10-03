"""Phase 1 sample serialization and validation; no model imports."""

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

SCHEMA_VERSION = "1.0"
PPP_POLICY = "camera_frustum_and_strict_square_range_before_pixel_deduplication"


def _store(group, name, value):
    if isinstance(value, str):
        return group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    array = np.asarray(value)
    if array.dtype.kind in "OU":
        return group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
    # Preserve actual numeric dtypes. Empty arrays need no chunking/compression.
    options = {"compression": "lzf", "shuffle": True} if array.ndim and array.size else {}
    return group.create_dataset(name, data=array, **options)


def annotation_payload(trucksc, sample_token):
    """Original global annotations, never visibility-filtered or interpolated."""
    sample = trucksc.get("sample", sample_token)
    records = [trucksc.get("sample_annotation", token) for token in sample["anns"]]
    payload = {
        "source_sample_token": sample_token,
        "source_timestamp_us": np.int64(sample["timestamp"]),
        "center_global": np.asarray([a["translation"] for a in records], dtype=np.float64).reshape(-1, 3),
        "size_wlh": np.asarray([a["size"] for a in records], dtype=np.float64).reshape(-1, 3),
        "rotation_wxyz": np.asarray([a["rotation"] for a in records], dtype=np.float64).reshape(-1, 4),
        "raw_annotations_json": json.dumps(records),
        "attribute_tokens": np.asarray([json.dumps(a.get("attribute_tokens", [])) for a in records], dtype=object),
    }
    for field, key in (("category_name", "category_name"), ("annotation_token", "token"),
                       ("instance_token", "instance_token"), ("visibility_token", "visibility_token")):
        payload[field] = np.asarray([a[key] for a in records], dtype=object)
    # Retain interpolation sources without inventing camera-frequency annotations.
    previous = trucksc.get("sample", sample["prev"]) if sample["prev"] else None
    payload["previous_sample_token"] = sample["prev"]
    payload["previous_annotations_json"] = json.dumps(
        [trucksc.get("sample_annotation", t) for t in previous["anns"]] if previous else [])
    payload["previous_timestamp_us"] = np.int64(previous["timestamp"] if previous else -1)
    return payload


def write_sample(output_root, scene_token, frame_index, conditioning, depth,
                 targets, radar, metadata, boxes):
    """Close and validate a private temporary HDF5 file before publishing it."""
    directory = Path(output_root) / scene_token
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"sample_{frame_index:06d}.h5"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    fd, temporary = tempfile.mkstemp(prefix=".sample_", suffix=".partial", dir=directory)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            cond = handle.create_group("conditioning")
            for name, value in zip(("appearance", "semantics", "velocity"), conditioning):
                _store(cond, name, value)
            depths = cond.create_group("depth")
            for camera, value in depth.items():
                ds = _store(depths, camera, value)
                ds.attrs["units"] = "m"
                ds.attrs["representation"] = "native_t0_camera_Z"
            target_group = handle.create_group("radargen_targets")
            for name, value in zip(("point_density", "rcs", "doppler"), targets):
                _store(target_group, name, value)
            ppp = handle.create_group("ppp_targets")
            _store(ppp, "xy", radar[:, :2])
            _store(ppp, "rcs", radar[:, 6])
            _store(ppp, "doppler", radar[:, 7])
            ppp.attrs["xy_frame"] = "RadarGen_vehicle_flat_up"
            ppp.attrs["xy_units"] = "m"
            ppp.attrs["rcs_units"] = "dBsm"
            ppp.attrs["doppler_units"] = "m/s"
            ppp.attrs["doppler_definition"] = "sensor_velocity_norm_times_sign_velocity_dot_sensor_position"
            meta = handle.create_group("metadata")
            values = dict(metadata, scene_token=scene_token, frame_index=np.int64(frame_index),
                          schema_version=SCHEMA_VERSION, ppp_filtering_policy=PPP_POLICY)
            for name, value in values.items():
                _store(meta, name, value)
            box_group = meta.create_group("bounding_boxes")
            for name, value in boxes.items():
                _store(box_group, name, value)
        validate_sample(temporary, scene_token, frame_index, list(depth),
                        int(metadata["map_resolution"]), report=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def pair_metadata(adapter, t0, t1, split):
    """Expected identities from the adapter, independent of stored metadata."""
    from radargen.datasets.truckscenes.config import TRUCKSCENES_RADAR_SENSORS

    ref_token = t0[adapter.reference_sensor]
    ref = adapter.trucksc.get("sample_data", ref_token)
    cameras = list(adapter.camera_views)
    radars = [name for name in TRUCKSCENES_RADAR_SENSORS if name in t0]
    expected = dict(
        sample_token=ref["sample_token"], split=split,
        dataset_version=adapter.trucksc.version,
        reference_sensor=adapter.reference_sensor,
        reference_sample_data_token=ref_token,
        reference_timestamp_us=ref["timestamp"],
        reference_is_key_frame=ref["is_key_frame"],
        camera_names=cameras, radar_names=radars,
    )
    for prefix, names, frame in (("camera_t0", cameras, t0),
                                  ("camera_t1", cameras, t1),
                                  ("radar_t0", radars, t0)):
        records = [adapter.trucksc.get("sample_data", frame[name]) for name in names]
        expected[f"{prefix}_sample_data_tokens"] = [frame[name] for name in names]
        expected[f"{prefix}_timestamps_us"] = [r["timestamp"] for r in records]
        expected[f"{prefix}_sample_tokens"] = [r["sample_token"] for r in records]
    return expected


def validate_sample(path, scene_token, frame_index, cameras, resolution,
                    depth_shapes=None, report=True, expected_metadata=None):
    """Read back schema, identities, variable lengths and original annotations."""
    with h5py.File(path, "r") as handle:
        def require(condition, message):
            if not condition:
                raise ValueError(f"{path}: {message}")

        def equal(actual, expected, name):
            require(np.array_equal(actual, expected), f"{name} mismatch")

        require(set(handle) == {"conditioning", "radargen_targets", "ppp_targets", "metadata"},
                "Required HDF5 groups mismatch")
        meta = handle["metadata"]
        equal(meta["scene_token"].asstr()[()], scene_token, "scene_token")
        equal(meta["frame_index"][()], frame_index, "frame_index")
        equal(meta["schema_version"].asstr()[()], SCHEMA_VERSION, "schema_version")
        if expected_metadata is not None:
            for name, expected in expected_metadata.items():
                require(name in meta, f"Missing metadata/{name}")
                ds = meta[name]
                actual = ds.asstr()[()] if h5py.check_string_dtype(ds.dtype) else ds[()]
                equal(actual, expected, f"metadata/{name}")

        def finite(name, item):
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "fciub":
                require(np.isfinite(item[()]).all(), f"Nonfinite numeric values in {name}")
        handle.visititems(finite)
        for name in ("appearance", "semantics", "velocity"):
            ds = handle[f"conditioning/{name}"]
            require(ds.shape == (resolution, resolution, 3) and ds.dtype == np.uint8,
                    f"conditioning/{name} shape/dtype mismatch")
        require(set(handle["conditioning/depth"]) == set(cameras), "Depth camera coverage mismatch")
        for camera in cameras:
            ds = handle[f"conditioning/depth/{camera}"]
            require(ds.ndim == 2 and ds.dtype.kind == "f", f"depth/{camera} shape/dtype mismatch")
            if depth_shapes is not None:
                require(ds.shape == tuple(depth_shapes[camera]), f"depth/{camera} native shape mismatch")
        for name in ("point_density", "rcs", "doppler"):
            ds = handle[f"radargen_targets/{name}"]
            require(ds.shape == (resolution, resolution) and ds.dtype.kind == "f",
                    f"radargen_targets/{name} shape/dtype mismatch")
        ppp = handle["ppp_targets"]
        count = len(ppp["rcs"])
        require(ppp["rcs"].shape == (count,) and ppp["xy"].shape == (count, 2)
                and ppp["doppler"].shape == (count,), "PPP lengths/shapes mismatch")
        for ds in ppp.values():
            require(ds.dtype == np.float32 and h5py.check_dtype(vlen=ds.dtype) is None,
                    f"{ds.name} dtype mismatch")
        boxes = meta["bounding_boxes"]
        records = json.loads(boxes["raw_annotations_json"].asstr()[()])
        for field, key, width in (("center_global", "translation", 3),
                                  ("size_wlh", "size", 3), ("rotation_wxyz", "rotation", 4)):
            expected = np.asarray([a[key] for a in records], dtype=np.float64).reshape(-1, width)
            equal(boxes[field][()], expected, f"bounding_boxes/{field}")
        for field, key in (("category_name", "category_name"), ("annotation_token", "token"),
                           ("instance_token", "instance_token"), ("visibility_token", "visibility_token")):
            equal(boxes[field].asstr()[()], [a[key] for a in records], f"bounding_boxes/{field}")
        require([json.loads(s) for s in boxes["attribute_tokens"].asstr()[()]] ==
                [a.get("attribute_tokens", []) for a in records], "Bounding box attributes mismatch")
        equal(boxes["source_sample_token"].asstr()[()], meta["sample_token"].asstr()[()],
              "bounding_boxes/source_sample_token")
        if report:
            def show(name, item):
                if isinstance(item, h5py.Dataset) and name.startswith(("conditioning/", "radargen_targets/", "ppp_targets/")):
                    print(f"{name}: shape={item.shape}, dtype={item.dtype}", flush=True)
            handle.visititems(show)


def compare_radar_maps(path, baseline_root, scene_token, frame_index):
    """Print and enforce shape/dtype parity and numerical equality to legacy NPY."""
    with h5py.File(path, "r") as handle:
        for name, prefix in (("point_density", "radar_pd_map"), ("rcs", "radar_rcs_map"),
                             ("doppler", "radar_doppler_map")):
            baseline = np.load(Path(baseline_root) / f"{prefix}_{scene_token}_{frame_index}.npy")
            actual = handle[f"radargen_targets/{name}"][()]
            if actual.shape != baseline.shape or actual.dtype != baseline.dtype:
                raise AssertionError(f"{name}: HDF5 {actual.shape}/{actual.dtype}, NPY {baseline.shape}/{baseline.dtype}")
            exact = np.array_equal(actual, baseline)
            numeric = np.allclose(actual, baseline, rtol=1e-6, atol=1e-8)
            difference = float(np.max(np.abs(actual - baseline)))
            print(f"frame={frame_index} {name}: HDF5={actual.shape}/{actual.dtype}, "
                  f"NPY={baseline.shape}/{baseline.dtype}, max_abs_diff={difference}, "
                  f"exact={exact}, numerical={numeric}", flush=True)
            if not numeric:
                raise AssertionError(f"Radar target mismatch: frame {frame_index}, {name}")
