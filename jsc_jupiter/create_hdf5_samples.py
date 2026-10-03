#!/usr/bin/env python3
"""Generate HDF5 samples for the single Phase 1 TruckScenes smoke scene."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SMOKE_SCENE = "018a60086f5e441fb09f476e55948b72"


@contextmanager
def staged_scene(archive_root, camera_archive_root, staging_root, permanent_root, scene_token, version):
    """Exclusively own a new staging directory; never reuse/delete existing roots.

    Stream zstd through tarfile and reject links and unexpected trees. Symlinks
    to permanent data are created only after extraction has completed.
    """
    archives = [
        Path(archive_root) / f"{scene_token}.tar.zst",
        Path(camera_archive_root) / f"{scene_token}.tar.zst",
    ]
    staging = Path(staging_root) / scene_token
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)
    for name in ("samples", version):
        if not (Path(permanent_root) / name).is_dir():
            raise FileNotFoundError(Path(permanent_root) / name)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()  # Refuse existing directories, including previously staged scenes.
    try:
        for archive in archives:
            print(f"Extracting: {archive}", flush=True)
            with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE) as process:
                try:
                    with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
                        for member in stream:
                            relative = PurePosixPath(member.name)
                            if member.isdir() and str(relative) == ".":
                                continue
                            if (relative.is_absolute() or ".." in relative.parts or
                                    not relative.parts or relative.parts[0] != "sweeps" or
                                    not (member.isdir() or member.isfile())):
                                raise ValueError(f"Unexpected archive member: {member.name}")
                            target = staging.joinpath(*relative.parts)
                            if member.isdir():
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with stream.extractfile(member) as source, target.open("xb") as destination:
                                    shutil.copyfileobj(source, destination)
                    process.stdout.close()
                    if process.wait() != 0:
                        raise RuntimeError(f"zstd extraction failed: {archive}")
                except BaseException:
                    process.terminate()
                    raise
        for name in ("samples", version):
            (staging / name).symlink_to((Path(permanent_root) / name).resolve(), target_is_directory=True)
        yield staging
    finally:
        # shutil.rmtree unlinks the permanent-data symlinks without following them.
        shutil.rmtree(staging)
        if staging.exists():
            raise RuntimeError(f"Staging cleanup failed: {staging}")
        print(f"Removed owned staging directory: {staging}", flush=True)


def frame_metadata(adapter, t0, t1, cfg, split):
    import numpy as np
    from radargen.datasets.truckscenes.config import TRUCKSCENES_RADAR_SENSORS
    from radargen.datasets.truckscenes.reader import get_sensor_from_global
    ref_token = t0[adapter.reference_sensor]
    ref_record = adapter.trucksc.get("sample_data", ref_token)
    flat = adapter.get_ref_to_vehicle_flat_up_transform(ref_token)
    _, _, global_to_ref = get_sensor_from_global(adapter.trucksc, ref_token)
    cameras = list(adapter.camera_views)
    radars = [name for name in TRUCKSCENES_RADAR_SENSORS if name in t0]
    metadata = {
        "sample_token": ref_record["sample_token"], "split": split,
        "dataset_version": cfg["dataset_version"],
        "reference_sensor": adapter.reference_sensor, "reference_sample_data_token": ref_token,
        "reference_timestamp_us": np.int64(ref_record["timestamp"]),
        "reference_is_key_frame": bool(ref_record["is_key_frame"]),
        "camera_names": cameras, "radar_names": radars,
        "camera_intrinsics": np.asarray(adapter.get_camera_intrinsics(t0)),
        "camera_to_reference": np.asarray(adapter.get_camera_extrinsics(t0, ref_token)),
        "reference_to_flat_up": flat, "global_to_flat_up": flat @ global_to_ref,
        "map_resolution": np.int64(cfg["resolution"]), "point_limit": cfg["point_limit"],
        "conditioning_coordinate_range": adapter.normalization.coordinate_range,
        "nsweeps": np.int64(cfg["nsweeps"]), "sigma": cfg["sigma"],
        "frustum_camera_views": cfg.get("filter_camera_views") or cameras,
        "range_filter": "abs(x) < point_limit AND abs(y) < point_limit",
        "normalization_json": json.dumps(vars(adapter.normalization)),
        "configuration_json": json.dumps(cfg),
        "annotation_policy": "original_reference_parent_sample_global_annotations_no_interpolation",
    }
    for prefix, names, frame in (("camera_t0", cameras, t0), ("camera_t1", cameras, t1), ("radar_t0", radars, t0)):
        records = [adapter.trucksc.get("sample_data", frame[name]) for name in names]
        metadata[f"{prefix}_sample_data_tokens"] = [frame[name] for name in names]
        metadata[f"{prefix}_timestamps_us"] = np.asarray([r["timestamp"] for r in records], dtype=np.int64)
        metadata[f"{prefix}_sample_tokens"] = [r["sample_token"] for r in records]
    return metadata


def run(cfg):
    import numpy as np
    import torch
    from radargen.bev_condition_maps.creator import create_bev_maps_for_frame
    from radargen.bev_condition_maps.foundation_models import load_models
    from radargen.datasets.registry import get_adapter
    from radargen.datasets.truckscenes.config import TRUCKSCENES_RADAR_SENSORS
    from radargen.radar_maps.creator import create_radar_maps_for_frame
    from jsc_jupiter.hdf5_samples import annotation_payload, compare_radar_maps, validate_sample, write_sample

    scene = cfg["scene_token"]
    if scene != SMOKE_SCENE or cfg["nsweeps"] != 1:
        raise ValueError("Phase 1 requires the smoke scene and nsweeps=1")
    output = Path(cfg["output_root"]).resolve()
    staging_root = Path(cfg["staging_root"]).resolve()
    if output == staging_root or staging_root in output.parents:
        raise ValueError("Output must be outside staging")
    if (output / scene).exists():
        raise FileExistsError(f"Choose a fresh output root; refusing existing scene output: {output / scene}")
    with staged_scene(cfg["archive_root"], cfg["camera_archive_root"], staging_root, cfg["permanent_root"], scene, cfg["dataset_version"]) as root:
        adapter = get_adapter(name="truckscenes", dataset_dir=str(root),
                              dataset_version=cfg["dataset_version"], camera_views=cfg.get("camera_views"))
        selected = [(split, frames) for split in ("train", "val")
                    for token, frames in adapter.iter_scenes(split, filter_fn=lambda token: token == scene)]
        if len(selected) != 1:
            raise RuntimeError(f"Expected exactly one selected scene, found {len(selected)}")
        split, samples = selected[0]
        compare_frames = set(cfg["compare_frames"])
        if not compare_frames or not compare_frames <= set(range(len(samples) - 1)) or 100 not in compare_frames:
            raise ValueError("Comparison frames must include 100 and be valid non-final scene indices")
        # Check all cameras, including the final frame used as t1.
        import cv2
        checked_camera_paths = set()
        for frame_index, frame in enumerate(samples):
            for camera in adapter.camera_views:
                token = frame[camera]
                record = adapter.trucksc.get("sample_data", token)
                filename = record["filename"]
                if filename in checked_camera_paths:
                    continue
                image_path = root / filename
                context = (
                    f"scene={scene}, frame={frame_index}, camera={camera}, "
                    f"token={token}, path={image_path}"
                )
                if not image_path.is_file():
                    raise FileNotFoundError(context)
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Camera image cannot be decoded: {context}")
                expected = (record["height"], record["width"])
                if image.shape[:2] != expected:
                    raise RuntimeError(
                        f"Camera dimensions {image.shape[:2]} != {expected}: {context}"
                    )
                checked_camera_paths.add(filename)
                del image
        print(
            f"Camera preflight passed: {len(checked_camera_paths)} unique images",
            flush=True,
        )

        # Preflight all radar files and selected legacy comparisons before model loading.
        for frame in samples[:-1]:
            for sensor in TRUCKSCENES_RADAR_SENSORS:
                if sensor in frame:
                    filename = adapter.trucksc.get("sample_data", frame[sensor])["filename"]
                    if not (root / filename).is_file():
                        raise FileNotFoundError(root / filename)
        for index in compare_frames:
            for prefix in ("radar_pd_map", "radar_rcs_map", "radar_doppler_map"):
                baseline = Path(cfg["baseline_radar_root"]) / f"{prefix}_{scene}_{index}.npy"
                if not baseline.is_file():
                    raise FileNotFoundError(baseline)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        depth_model, seg, seg_proc, flow = load_models(device)
        models = dict(depth=depth_model, segmentation=seg, segmentation_processor=seg_proc, flow=flow)
        for index in range(len(samples) - 1):
            t0, t1 = samples[index:index + 2]
            depth = {}
            captured = []

            def capture_depth(outputs):
                if len(outputs) != len(adapter.camera_views):
                    raise ValueError("Depth camera count mismatch")
                for name, value in zip(adapter.camera_views, outputs):
                    depth[name] = value["depth"].detach().cpu().numpy().copy()

            conditioning = create_bev_maps_for_frame(
                adapter, t0, t1, models, device, cfg["resolution"],
                cfg["use_batched_inference"], depth_callback=capture_depth)
            targets = create_radar_maps_for_frame(
                adapter, t0, device, image_size=cfg["resolution"], point_limit=cfg["point_limit"],
                nsweeps=cfg["nsweeps"], sigma_range=(cfg["sigma"], cfg["sigma"]),
                filter_camera_views=cfg.get("filter_camera_views"), radar_callback=captured.append)
            if len(captured) != 1:
                raise RuntimeError("Expected one metric radar callback")
            radar = captured[0]
            metadata = frame_metadata(adapter, t0, t1, cfg, split)
            boxes = annotation_payload(adapter.trucksc, metadata["sample_token"])
            path = write_sample(output, scene, index, conditioning, depth,
                                (targets.pd_map, targets.rcs_map, targets.doppler_map), radar, metadata, boxes)
            shapes = dict(zip(adapter.camera_views, adapter.get_camera_image_shapes(t0)))
            validate_sample(path, scene, index, adapter.camera_views, cfg["resolution"],
                            depth_shapes=shapes, report=index == 0 or index in compare_frames)
            # Verify PPP serialization directly against the exact captured detections.
            import h5py
            with h5py.File(path, "r") as handle:
                for name, values in (("xy", radar[:, :2]), ("rcs", radar[:, 6]), ("doppler", radar[:, 7])):
                    np.testing.assert_array_equal(handle[f"ppp_targets/{name}"][()], values)
            if index in compare_frames:
                compare_radar_maps(path, cfg["baseline_radar_root"], scene, index)
            print(f"[{index + 1}/{len(samples) - 1}] {path}", flush=True)
    print("One-scene preprocessing and validation completed; staging removed.", flush=True)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, default=REPO_ROOT / "jsc_jupiter/truckscenes_hdf5_smoke.yaml")
    parser.add_argument("--staging_root", type=Path, help="Optional fresh staging parent; existing scene directories are never reused")
    args = parser.parse_args()
    with args.config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    if args.staging_root is not None:
        cfg["staging_root"] = str(args.staging_root)
    run(cfg)


if __name__ == "__main__":
    main()
