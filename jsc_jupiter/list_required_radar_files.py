#!/usr/bin/env python3
"""List radar-map input files without reading PCD contents.

Run with RadarGen's Python environment:
    python jsc_jupiter/list_required_radar_files.py

The CSV preserves every frame/file reference; the TXT deduplicates paths.
Includes the last frame of each scene, exactly as radar-map preprocessing does.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

# Support invocation from outside the repository as well as from its root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pyrallis

from radargen.datasets.registry import get_adapter
from radargen.datasets.truckscenes.config import TRUCKSCENES_RADAR_SENSORS
from radargen.radar_maps.config import RadarMapCreationConfig


def iter_references(adapter, splits):
    """Use preprocessing's scene/frame selection and nsweeps=1 sensor loop.

    TruckScenesAdapter.load_radar_pointcloud loops over these same sensors.
    load_radar_multisweep starts at sample_data[sensor] and reads that record's
    filename on its sole iteration when nsweeps=1; no predecessor is loaded.
    """
    for split in splits:
        for scene_token, frames in adapter.iter_scenes(split=split):
            print(f"{split}: scene {scene_token}, {len(frames)} frames", flush=True)
            for frame_index, frame in enumerate(frames):
                for sensor in TRUCKSCENES_RADAR_SENSORS:
                    if sensor not in frame:
                        continue
                    token = frame[sensor]
                    record = adapter.trucksc.get("sample_data", token)
                    relative_path = record["filename"]
                    yield {
                        "split": split,
                        "scene_token": scene_token,
                        "frame_index": frame_index,
                        "sensor": sensor,
                        "sample_data_token": token,
                        "is_key_frame": record["is_key_frame"],
                        "relative_path": relative_path,
                        "absolute_path": str(Path(adapter.trucksc.dataroot) / relative_path),
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, default=REPO_ROOT / "jsc_jupiter/truckscenes_radar_maps_jupiter.yaml")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "jsc_jupiter")
    args = parser.parse_args()
    with args.config_path.open() as stream:
        cfg = pyrallis.load(RadarMapCreationConfig, stream)
    if cfg.dataset_name != "truckscenes" or cfg.nsweeps != 1:
        parser.error("This utility requires dataset_name=truckscenes and nsweeps=1.")
    if cfg.filter_for_split is not None:
        parser.error("Complete train/val enumeration requires filter_for_split=null.")

    # Identical constructor arguments to scripts/create_radar_maps.py;
    # preserve the adapter's default illuminated_only and camera frequency.
    adapter = get_adapter(name=cfg.dataset_name, dataset_dir=str(cfg.dataset_dir), dataset_version=cfg.dataset_version)
    print(f"Config: {args.config_path}\nDataset: {cfg.dataset_dir}, {cfg.dataset_version}\n"
          f"nsweeps={cfg.nsweeps}, illuminated_only={adapter.illuminated_only}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "required_radar_files.csv"
    txt_path = args.output_dir / "required_radar_files.txt"
    columns = ["split", "scene_token", "frame_index", "sensor", "sample_data_token",
               "is_key_frame", "relative_path", "absolute_path"]
    unique = {}
    references = 0
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in iter_references(adapter, ["train", "val"]):
            writer.writerow(row)
            references += 1
            unique.setdefault(row["relative_path"], (row["absolute_path"], row["sensor"]))
    with txt_path.open("w") as stream:
        for relative_path in sorted(unique):
            stream.write(relative_path + "\n")

    directories = Counter(Path(path).parts[0] for path in unique)
    sensors = Counter(sensor for _, sensor in unique.values())
    total_size = missing = inaccessible = existing = 0
    for absolute_path, _ in unique.values():
        try:
            stat = Path(absolute_path).stat()
        except FileNotFoundError:
            missing += 1
        except OSError:
            inaccessible += 1
        else:
            existing += 1
            total_size += stat.st_size
    print(f"\nTotal frame/file references: {references:,}\nUnique PCD files: {len(unique):,}\n"
          f"Unique files under samples/: {directories['samples']:,}\n"
          f"Unique files under sweeps/: {directories['sweeps']:,}")
    for sensor in TRUCKSCENES_RADAR_SENSORS:
        print(f"Unique files for {sensor}: {sensors[sensor]:,}")
    print(f"Existing files with available size: {existing:,}\n"
          f"Total size of those files: {total_size:,} bytes ({total_size / 1024**3:.3f} GiB)\n"
          f"Required files missing: {missing:,}\n"
          f"Files whose existence/size could not be checked: {inaccessible:,}\n"
          f"CSV: {csv_path}\nDeduplicated list: {txt_path}")


if __name__ == "__main__":
    main()
