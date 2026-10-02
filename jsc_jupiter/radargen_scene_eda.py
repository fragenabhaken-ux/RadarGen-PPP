#!/usr/bin/env python3
"""
Small EDA for one staged RadarGen / MAN TruckScenes scene.

Outputs:
  - camera images available for one representative frame
  - radar BEV
  - radar BEV colored by RCS
  - radar BEV colored by relative x velocity
  - RCS histogram
  - relative velocity histogram
  - point count over all frames
  - point count histogram
  - selected-frame radar points as CSV
  - scene point counts as CSV
  - text summary

Run from the RadarGen environment:
    python jsc_jupiter/radargen_scene_eda.py
"""

from pathlib import Path
import sys
import csv
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path("/e/project1/nxtaim-1/huber7/repos/RadarGen")

SCENE_TOKEN = "018a60086f5e441fb09f476e55948b72"

DATASET_ROOT = (
    Path("/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/staging")
    / SCENE_TOKEN
)

OUTPUT_DIR = (
    Path("/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/eda")
    / SCENE_TOKEN
)

DATASET_VERSION = "v1.2-trainval"
NSWEEPS = 1

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert REPO_ROOT.exists(), f"Missing repo: {REPO_ROOT}"
assert DATASET_ROOT.exists(), f"Missing staged scene: {DATASET_ROOT}"
assert (DATASET_ROOT / "sweeps").exists(), "Missing sweeps/"
assert (DATASET_ROOT / "samples").exists(), "Missing samples/"
assert (DATASET_ROOT / DATASET_VERSION).exists(), f"Missing {DATASET_VERSION}/"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radargen.datasets.registry import get_adapter

print("Repository:  ", REPO_ROOT)
print("Dataset:     ", DATASET_ROOT)
print("Scene:       ", SCENE_TOKEN)
print("Output:      ", OUTPUT_DIR)
print()

adapter = get_adapter(
    name="truckscenes",
    dataset_dir=str(DATASET_ROOT),
    dataset_version=DATASET_VERSION,
)

# ---------------------------------------------------------------------
# Find target scene exactly as RadarGen sees it
# ---------------------------------------------------------------------

target_split = None
frames = None

for split in ("train", "val"):
    for scene_token, scene_frames in adapter.iter_scenes(split=split):
        if scene_token == SCENE_TOKEN:
            target_split = split
            frames = scene_frames
            break
    if frames is not None:
        break

if frames is None:
    raise RuntimeError(f"Scene {SCENE_TOKEN} not found")

print(f"Split:  {target_split}")
print(f"Frames: {len(frames)}")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def radar_to_array(obj):
    """Normalize RadarGen radar cloud to shape N x D."""
    if hasattr(obj, "points"):
        arr = np.asarray(obj.points)
    elif isinstance(obj, tuple):
        candidates = []
        for item in obj:
            try:
                x = np.asarray(item)
            except Exception:
                continue
            if x.ndim == 2:
                candidates.append(x)
        if not candidates:
            raise TypeError(
                f"Could not find a 2D radar array in tuple of "
                f"{[type(x).__name__ for x in obj]}"
            )
        arr = candidates[0]
    else:
        arr = np.asarray(obj)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D radar array, got {arr.shape}")

    # Point-cloud APIs often use D x N.
    if arr.shape[0] <= 20 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    return arr


def camera_files(frame):
    result = []
    for sensor, token in frame.items():
        if not str(sensor).startswith("CAMERA_"):
            continue
        rec = adapter.trucksc.get("sample_data", token)
        path = DATASET_ROOT / rec["filename"]
        if path.is_file():
            result.append((sensor, token, path, rec))
    return result


def save_scatter(x, y, out_path, title, c=None, colorbar_label=None):
    plt.figure(figsize=(8, 8))
    if c is None:
        plt.scatter(x, y, s=10)
    else:
        sc = plt.scatter(x, y, c=c, s=14)
        plt.colorbar(sc, label=colorbar_label)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Select representative frame with available camera image
# ---------------------------------------------------------------------

camera_candidates = []

for i, frame in enumerate(frames):
    cams = camera_files(frame)
    if cams:
        camera_candidates.append((i, frame, cams))

if not camera_candidates:
    raise RuntimeError("No camera images available for this staged scene")

frame_index, frame, cameras = camera_candidates[len(camera_candidates) // 2]

print(f"Selected frame: {frame_index}")
print(f"Available cameras for selected frame: {len(cameras)}")

# Save original camera images without recompression where possible.
for sensor, token, source, rec in cameras:
    suffix = source.suffix.lower() or ".png"
    dest = OUTPUT_DIR / f"camera_{sensor}_frame_{frame_index:04d}{suffix}"
    shutil.copy2(source, dest)
    print("Camera:", dest.name)

# ---------------------------------------------------------------------
# Selected-frame radar cloud
# ---------------------------------------------------------------------

radar_obj = adapter.load_radar_pointcloud(frame, nsweeps=NSWEEPS)
radar = radar_to_array(radar_obj)

if radar.shape[1] < 7:
    raise RuntimeError(
        "Expected at least 7 radar fields "
        "(x, y, z, vrel_x, vrel_y, vrel_z, rcs), "
        f"got {radar.shape}"
    )

x = radar[:, 0]
y = radar[:, 1]
z = radar[:, 2]
vrel_x = radar[:, 3]
vrel_y = radar[:, 4]
vrel_z = radar[:, 5]
rcs = radar[:, 6]
vrel_xy = np.sqrt(vrel_x**2 + vrel_y**2)

print(f"Radar array shape: {radar.shape}")

# Save raw selected-frame points.
point_csv = OUTPUT_DIR / f"radar_points_frame_{frame_index:04d}.csv"
with point_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    header = [
        "x", "y", "z",
        "vrel_x", "vrel_y", "vrel_z",
        "rcs",
    ]
    if radar.shape[1] > 7:
        header += [f"extra_{i}" for i in range(radar.shape[1] - 7)]
    writer.writerow(header)
    writer.writerows(radar)

save_scatter(
    x, y,
    OUTPUT_DIR / f"radar_bev_frame_{frame_index:04d}.png",
    f"Radar BEV | frame {frame_index} | n={len(radar)}",
)

save_scatter(
    x, y,
    OUTPUT_DIR / f"radar_bev_rcs_frame_{frame_index:04d}.png",
    f"Radar BEV by RCS | frame {frame_index}",
    c=rcs,
    colorbar_label="RCS",
)

save_scatter(
    x, y,
    OUTPUT_DIR / f"radar_bev_vrelx_frame_{frame_index:04d}.png",
    f"Radar BEV by relative velocity | frame {frame_index}",
    c=vrel_x,
    colorbar_label="vrel_x [m/s]",
)

plt.figure(figsize=(8, 4))
plt.hist(rcs[np.isfinite(rcs)], bins=40)
plt.xlabel("RCS")
plt.ylabel("Radar detections")
plt.title(f"RCS distribution | frame {frame_index}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"rcs_hist_frame_{frame_index:04d}.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(vrel_x[np.isfinite(vrel_x)], bins=40)
plt.xlabel("vrel_x [m/s]")
plt.ylabel("Radar detections")
plt.title(f"Relative velocity distribution | frame {frame_index}")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / f"vrelx_hist_frame_{frame_index:04d}.png",
    dpi=180,
)
plt.close()

# ---------------------------------------------------------------------
# Point counts across the whole staged scene
# ---------------------------------------------------------------------

print()
print("Reading all scene frames for point-count EDA...")

point_counts = []

for i, scene_frame in enumerate(frames):
    cloud = adapter.load_radar_pointcloud(scene_frame, nsweeps=NSWEEPS)
    arr = radar_to_array(cloud)
    point_counts.append(len(arr))

point_counts = np.asarray(point_counts, dtype=int)

with (OUTPUT_DIR / "scene_point_counts.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "radar_points"])
    writer.writerows(enumerate(point_counts))

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(point_counts)), point_counts)
plt.xlabel("Frame index")
plt.ylabel("Radar detections")
plt.title("Radar point count across staged scene")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scene_point_count_over_time.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(point_counts, bins=30)
plt.xlabel("Radar detections per frame")
plt.ylabel("Frames")
plt.title("Point-count distribution across staged scene")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scene_point_count_hist.png", dpi=180)
plt.close()

# ---------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------

summary = f"""RadarGen / MAN TruckScenes scene EDA

scene_token: {SCENE_TOKEN}
split: {target_split}
scene_frames: {len(frames)}
selected_frame: {frame_index}
selected_frame_radar_points: {len(radar)}

Selected-frame spatial extent
x_min_m: {np.nanmin(x):.6f}
x_max_m: {np.nanmax(x):.6f}
y_min_m: {np.nanmin(y):.6f}
y_max_m: {np.nanmax(y):.6f}

Selected-frame RCS
mean: {np.nanmean(rcs):.6f}
std: {np.nanstd(rcs):.6f}
min: {np.nanmin(rcs):.6f}
max: {np.nanmax(rcs):.6f}

Selected-frame relative velocity
vrel_x_mean_mps: {np.nanmean(vrel_x):.6f}
vrel_x_std_mps: {np.nanstd(vrel_x):.6f}
vrel_xy_mean_mps: {np.nanmean(vrel_xy):.6f}

Scene point counts
mean: {point_counts.mean():.6f}
median: {np.median(point_counts):.6f}
min: {point_counts.min()}
max: {point_counts.max()}
"""

(OUTPUT_DIR / "summary.txt").write_text(summary)

print()
print(summary)
print("Created files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" ", p.name)

print()
print("DONE")
print("Output directory:", OUTPUT_DIR)
