#!/usr/bin/env python3
"""
EDA for RadarGen radar target maps.

This script inspects the preprocessed radar target maps
    radar_pd_map
    radar_rcs_map
    radar_doppler_map

for one smoke-test scene and writes plots/statistics to an output folder.

It also loads the raw fused radar point cloud from the staged scene
for a few representative frames, so the original sparse detections can be
compared against RadarGen's dense map representation.

Run:
    cd /e/project1/nxtaim-1/huber7/repos/RadarGen
    source jsc_jupiter/activate_jupiter.sh
    python jsc_jupiter/radargen_radar_maps_eda.py
"""

from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path("/e/project1/nxtaim-1/huber7/repos/RadarGen")
SCENE_TOKEN = "018a60086f5e441fb09f476e55948b72"

RADAR_MAP_ROOT = Path(
    "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/preprocessing/smoke/radar_maps"
)

STAGED_SCENE_ROOT = Path(
    "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/staging"
) / SCENE_TOKEN

OUTPUT_DIR = Path(
    "/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/eda_radar_maps"
) / SCENE_TOKEN

FRAME_SELECTION = [0, 50, 100, 150, 194]
DATASET_VERSION = "v1.2-trainval"
NSWEEPS = 1

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert REPO_ROOT.exists(), f"Missing repo: {REPO_ROOT}"
assert RADAR_MAP_ROOT.exists(), f"Missing radar map root: {RADAR_MAP_ROOT}"
assert STAGED_SCENE_ROOT.exists(), f"Missing staged scene root: {STAGED_SCENE_ROOT}"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radargen.datasets.registry import get_adapter

# ---------------------------------------------------------------------
# Discover files
# ---------------------------------------------------------------------

def parse_frame_idx(path: Path, scene_token: str) -> int:
    stem = path.stem
    suffix = stem.split(f"_{scene_token}_")[-1]
    return int(suffix)

prefixes = ["radar_pd_map", "radar_rcs_map", "radar_doppler_map"]

maps_by_prefix = {}
frame_sets = []

for prefix in prefixes:
    files = sorted(RADAR_MAP_ROOT.glob(f"{prefix}_{SCENE_TOKEN}_*.npy"))
    if not files:
        raise RuntimeError(f"No files found for prefix {prefix}")
    by_idx = {parse_frame_idx(p, SCENE_TOKEN): p for p in files}
    maps_by_prefix[prefix] = by_idx
    frame_sets.append(set(by_idx.keys()))

common_frames = sorted(set.intersection(*frame_sets))
if not common_frames:
    raise RuntimeError("No common frames across the three radar-map modalities")

selected_frames = [f for f in FRAME_SELECTION if f in common_frames]
if not selected_frames:
    selected_frames = common_frames[: min(5, len(common_frames))]

# ---------------------------------------------------------------------
# Load staged scene through RadarGen adapter
# ---------------------------------------------------------------------

adapter = get_adapter(
    name="truckscenes",
    dataset_dir=str(STAGED_SCENE_ROOT),
    dataset_version=DATASET_VERSION,
)

scene_frames = None
scene_split = None

for split in ("train", "val"):
    for scene_token, frames in adapter.iter_scenes(split=split):
        if scene_token == SCENE_TOKEN:
            scene_frames = frames
            scene_split = split
            break
    if scene_frames is not None:
        break

if scene_frames is None:
    raise RuntimeError(f"Scene {SCENE_TOKEN} not found in staged scene root")

def radar_to_array(obj):
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
                f"Could not find a 2D radar array in tuple: "
                f"{[type(x).__name__ for x in obj]}"
            )
        arr = candidates[0]
    else:
        arr = np.asarray(obj)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D radar array, got {arr.shape}")

    if arr.shape[0] <= 20 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    return arr

# ---------------------------------------------------------------------
# Utility plotting helpers
# ---------------------------------------------------------------------

def save_heatmap(arr, out_path: Path, title: str, cmap: str = "viridis"):
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, origin="lower", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_scatter(x, y, out_path: Path, title: str, c=None, colorbar_label=None):
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

def finite_values(arr):
    arr = np.asarray(arr)
    return arr[np.isfinite(arr)]

# ---------------------------------------------------------------------
# Global map statistics
# ---------------------------------------------------------------------

summary_lines = []
summary_lines.append("RadarGen radar-map EDA")
summary_lines.append("")
summary_lines.append(f"scene_token: {SCENE_TOKEN}")
summary_lines.append(f"split: {scene_split}")
summary_lines.append(f"number_of_frames: {len(common_frames)}")
summary_lines.append(f"selected_frames: {selected_frames}")
summary_lines.append("")

global_stats = {}
modality_values_for_hist = {}

for prefix in prefixes:
    frame_arrays = []
    means = []
    mins = []
    maxs = []
    stds = []

    for idx in common_frames:
        arr = np.load(maps_by_prefix[prefix][idx])
        frame_arrays.append(arr)
        vals = finite_values(arr)
        means.append(float(np.mean(vals)))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))
        stds.append(float(np.std(vals)))

    stacked = np.stack(frame_arrays, axis=0)
    vals = finite_values(stacked)

    global_stats[prefix] = {
        "shape_per_frame": frame_arrays[0].shape,
        "dtype_per_frame": str(frame_arrays[0].dtype),
        "global_min": float(np.min(vals)),
        "global_max": float(np.max(vals)),
        "global_mean": float(np.mean(vals)),
        "global_std": float(np.std(vals)),
        "frame_mean_mean": float(np.mean(means)),
        "frame_mean_std": float(np.std(means)),
        "frame_min_min": float(np.min(mins)),
        "frame_max_max": float(np.max(maxs)),
        "frame_std_mean": float(np.mean(stds)),
    }

    modality_values_for_hist[prefix] = vals

summary_lines.append("Global map statistics")
for prefix in prefixes:
    stats = global_stats[prefix]
    summary_lines.append(f"[{prefix}]")
    for k, v in stats.items():
        summary_lines.append(f"{k}: {v}")
    summary_lines.append("")

# ---------------------------------------------------------------------
# Selected-frame raw radar and map visualizations
# ---------------------------------------------------------------------

raw_point_counts_selected = []

for idx in selected_frames:
    frame = scene_frames[idx]

    radar_obj = adapter.load_radar_pointcloud(frame, nsweeps=NSWEEPS)
    radar = radar_to_array(radar_obj)
    raw_point_counts_selected.append((idx, len(radar)))

    if radar.shape[1] < 7:
        raise RuntimeError(
            "Expected at least 7 radar fields "
            "(x, y, z, vrel_x, vrel_y, vrel_z, rcs)"
        )

    x = radar[:, 0]
    y = radar[:, 1]
    vrel_x = radar[:, 3]
    rcs = radar[:, 6]

    save_scatter(
        x, y,
        OUTPUT_DIR / f"raw_radar_bev_frame_{idx:04d}.png",
        f"Raw radar point cloud | frame {idx} | n={len(radar)}",
    )

    save_scatter(
        x, y,
        OUTPUT_DIR / f"raw_radar_rcs_frame_{idx:04d}.png",
        f"Raw radar point cloud by RCS | frame {idx}",
        c=rcs,
        colorbar_label="RCS",
    )

    save_scatter(
        x, y,
        OUTPUT_DIR / f"raw_radar_vrelx_frame_{idx:04d}.png",
        f"Raw radar point cloud by relative velocity | frame {idx}",
        c=vrel_x,
        colorbar_label="vrel_x [m/s]",
    )

    pd = np.load(maps_by_prefix["radar_pd_map"][idx])
    rcs_map = np.load(maps_by_prefix["radar_rcs_map"][idx])
    doppler = np.load(maps_by_prefix["radar_doppler_map"][idx])

    save_heatmap(
        pd,
        OUTPUT_DIR / f"radar_pd_map_frame_{idx:04d}.png",
        f"RadarGen point-density map | frame {idx}",
        cmap="viridis",
    )

    save_heatmap(
        rcs_map,
        OUTPUT_DIR / f"radar_rcs_map_frame_{idx:04d}.png",
        f"RadarGen RCS map | frame {idx}",
        cmap="viridis",
    )

    save_heatmap(
        doppler,
        OUTPUT_DIR / f"radar_doppler_map_frame_{idx:04d}.png",
        f"RadarGen Doppler map | frame {idx}",
        cmap="viridis",
    )

# ---------------------------------------------------------------------
# Histograms and frame-level trends
# ---------------------------------------------------------------------

for prefix in prefixes:
    vals = modality_values_for_hist[prefix]
    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=60)
    plt.xlabel(prefix)
    plt.ylabel("Pixel count")
    plt.title(f"Distribution of {prefix} values across all frames")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_hist.png", dpi=180)
    plt.close()

for prefix in prefixes:
    frame_means = []
    frame_stds = []
    for idx in common_frames:
        arr = np.load(maps_by_prefix[prefix][idx])
        vals = finite_values(arr)
        frame_means.append(np.mean(vals))
        frame_stds.append(np.std(vals))

    plt.figure(figsize=(12, 4))
    plt.plot(common_frames, frame_means)
    plt.xlabel("Frame index")
    plt.ylabel("Per-frame mean")
    plt.title(f"Per-frame mean of {prefix}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_frame_means.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(common_frames, frame_stds)
    plt.xlabel("Frame index")
    plt.ylabel("Per-frame std")
    plt.title(f"Per-frame std of {prefix}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_frame_stds.png", dpi=180)
    plt.close()

# ---------------------------------------------------------------------
# Raw point-count EDA over the whole staged scene
# ---------------------------------------------------------------------

point_counts = []
for idx, frame in enumerate(scene_frames):
    cloud = adapter.load_radar_pointcloud(frame, nsweeps=NSWEEPS)
    arr = radar_to_array(cloud)
    point_counts.append(len(arr))

point_counts = np.asarray(point_counts, dtype=int)

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(point_counts)), point_counts)
plt.xlabel("Frame index")
plt.ylabel("Radar detections")
plt.title("Raw radar point count across staged scene")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_point_counts_over_time.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(point_counts, bins=30)
plt.xlabel("Radar detections per frame")
plt.ylabel("Frames")
plt.title("Raw radar point-count distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_point_count_hist.png", dpi=180)
plt.close()

# ---------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------

with (OUTPUT_DIR / "selected_raw_point_counts.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "raw_radar_points"])
    writer.writerows(raw_point_counts_selected)

with (OUTPUT_DIR / "raw_point_counts_all_frames.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "raw_radar_points"])
    writer.writerows(enumerate(point_counts))

with (OUTPUT_DIR / "map_file_index.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "radar_pd_map", "radar_rcs_map", "radar_doppler_map"])
    for idx in common_frames:
        writer.writerow([
            idx,
            maps_by_prefix["radar_pd_map"][idx].name,
            maps_by_prefix["radar_rcs_map"][idx].name,
            maps_by_prefix["radar_doppler_map"][idx].name,
        ])

summary_lines.append("Raw radar point-count statistics")
summary_lines.append(f"mean: {point_counts.mean():.6f}")
summary_lines.append(f"median: {np.median(point_counts):.6f}")
summary_lines.append(f"min: {point_counts.min()}")
summary_lines.append(f"max: {point_counts.max()}")
summary_lines.append("")

(OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines))

print("DONE")
print("Output directory:", OUTPUT_DIR)
print("Files created:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" ", p.name)
