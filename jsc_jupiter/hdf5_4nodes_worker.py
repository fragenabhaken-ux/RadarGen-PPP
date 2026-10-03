#!/usr/bin/env python3
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import traceback

import yaml

REPO = Path("/e/project1/nxtaim-1/huber7/repos/RadarGen")
WORK = Path("/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen")
BASE = REPO / "jsc_jupiter"
OUTPUT = WORK / "preprocessing/full/hdf5_samples"
sys.path.insert(0, str(REPO))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_scene(adapter, scene):
    selected = [(split, frames) for split in ("train", "val")
                for token, frames in adapter.iter_scenes(
                    split, filter_fn=lambda token: token == scene)]
    if len(selected) != 1 or len(selected[0][1]) < 2:
        raise ValueError(f"{scene}: expected one adapter scene with at least two frames")
    return selected[0]


def validate_scene_output(directory, scene, signature, adapter, frames, split,
                          resolution, completed=False):
    """Require the adapter's full inventory and validate every consecutive pair."""
    from jsc_jupiter.hdf5_samples import pair_metadata, validate_sample

    expected_frames = len(frames)
    expected_pairs = expected_frames - 1
    if expected_pairs < 1:
        raise ValueError(f"{scene}: adapter scene has fewer than two frames")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{scene}: invalid scene directory: {directory}")
    expected = {f"sample_{i:06d}.h5" for i in range(expected_pairs)}
    allowed = expected | ({"_SUCCESS.json"} if completed else set())
    actual_names = {p.name for p in directory.iterdir()}
    if actual_names != allowed:
        raise ValueError(f"{scene}: incomplete or unexpected inventory; "
                         f"missing={sorted(allowed - actual_names)}, "
                         f"unexpected={sorted(actual_names - allowed)}")
    marker = None
    if completed:
        marker_path = directory / "_SUCCESS.json"
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError(f"{scene}: invalid completion marker")
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"{scene}: unreadable completion marker") from error
        if not isinstance(marker, dict) or marker.get("scene") != scene or marker.get("signature") != signature:
            raise ValueError(f"{scene}: incompatible completion marker (scene/signature)")
        if (marker.get("expected_frames") != expected_frames or
                marker.get("expected_pairs") != expected_pairs):
            raise ValueError(f"{scene}: incompatible completion marker (adapter frame/pair counts)")
    files = {}
    for index in range(expected_pairs):
        path = directory / f"sample_{index:06d}.h5"
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{scene}: missing, empty or linked sample: {path}")
        files[path.name] = path.stat().st_size
    if completed and marker.get("files") != files:
        raise ValueError(f"{scene}: completed scene file inventory has changed")
    for index in range(expected_pairs):
        t0, t1 = frames[index:index + 2]
        shapes = dict(zip(adapter.camera_views, adapter.get_camera_image_shapes(t0)))
        validate_sample(directory / f"sample_{index:06d}.h5", scene, index,
                        adapter.camera_views, resolution, depth_shapes=shapes,
                        report=False, expected_metadata=pair_metadata(adapter, t0, t1, split))
    return dict(scene=scene, signature=signature, files=files,
                expected_frames=expected_frames, expected_pairs=expected_pairs)


def process_scenes():
    import torch
    from radargen.datasets.registry import get_adapter

    rank = int(os.environ["SLURM_PROCID"])
    workers = int(os.environ["SLURM_NTASKS"])
    job = os.environ["SLURM_JOB_ID"]
    if workers != 16:
        raise RuntimeError(f"Expected 16 workers, got {workers}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one visible GPU; CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    runner_path = BASE / "hdf5_4nodes_runner.py"
    runner = load_module("production_runner", runner_path)
    cfg = yaml.safe_load((BASE / "truckscenes_hdf5_smoke.yaml").read_text())
    scenes = (BASE / "hdf5_4nodes_scenes.txt").read_text().splitlines()
    if len(scenes) != 480 or len(set(scenes)) != 480:
        raise RuntimeError("Expected 480 distinct scenes")

    # Identify the processing configuration independently of temporary paths.
    settings = {
        k: v for k, v in cfg.items()
        if k not in ("scene_token", "output_root", "staging_root")
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(runner_path.read_bytes())
    digest.update((BASE / "hdf5_samples.py").read_bytes())
    signature = digest.hexdigest()

    # Metadata-only adapter: identical selection to the staged production runner.
    adapter = get_adapter(name="truckscenes", dataset_dir=cfg["permanent_root"],
                          dataset_version=cfg["dataset_version"],
                          camera_views=cfg.get("camera_views"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    attempts = OUTPUT.parent / "hdf5_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    assigned = scenes[rank::workers]
    failed = []

    print(
        f"Worker {rank}: host={socket.gethostname()}, "
        f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}, scenes={len(assigned)}",
        flush=True,
    )

    for scene in assigned:
        destination = OUTPUT / scene
        attempt = None
        try:
            split, frames = select_scene(adapter, scene)
            if destination.exists() or destination.is_symlink():
                validate_scene_output(destination, scene, signature, adapter, frames,
                                      split, cfg["resolution"], completed=True)
                print(f"SKIP completed scene: {scene}", flush=True)
                continue

            attempt = Path(tempfile.mkdtemp(
                prefix=f"{job}_{rank}_{scene}_", dir=attempts
            ))
            staging = WORK / "staging/hdf5_full" / attempt.name
            scene_cfg = dict(
                cfg,
                scene_token=scene,
                output_root=str(attempt),
                staging_root=str(staging),
            )

            print(f"START scene: {scene}", flush=True)
            runner.run(scene_cfg)

            produced = attempt / scene
            marker = validate_scene_output(produced, scene, signature, adapter, frames,
                                           split, cfg["resolution"])
            (produced / "_SUCCESS.json").write_text(json.dumps(marker, sort_keys=True))
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Output appeared during processing: {destination}")
            produced.rename(destination)
            if staging.exists():
                staging.rmdir()
            print(f"DONE scene: {scene}, samples={marker['expected_pairs']}", flush=True)

        except Exception:
            failed.append(scene)
            print(f"FAILED scene: {scene}", flush=True)
            traceback.print_exc()
        finally:
            # Delete only this worker's private attempt, never published output.
            if attempt is not None:
                shutil.rmtree(attempt)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Worker {rank} finished; failed scenes: {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    wrapper = Path("/e/project1/nxtaim-1/huber7/pretrained/offline_hdf5_smoke.py")
    offline = load_module("offline_models", wrapper)

    # Retain the tested offline cache and DINO setup, then run this worker.
    def dispatch(path, run_name):
        process_scenes()

    offline.runpy.run_path = dispatch
    sys.argv = [str(wrapper)]
    offline.main()
