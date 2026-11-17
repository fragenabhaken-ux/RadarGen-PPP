#!/usr/bin/env python3
"""Copy scene comparison/overlay MP4s into one folder. Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

DEFAULT_SOURCE = Path(r'C:\Users\loren\Desktop\PPP\261109')
SCENE_TOKEN = re.compile(r'^[0-9a-f]{32}$', re.IGNORECASE)


def safe_name(value):
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    return value or 'unknown'


def describe(video, source):
    relative = video.relative_to(source)
    scene = next((part for part in relative.parts[:-1] if SCENE_TOKEN.fullmatch(part)), None)
    if scene is None:
        raise ValueError('no scene token in parent folders')
    metadata_path = video.parent / 'render_config.json'
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pass  # Older downloads may not include complete metadata.
    is_overlay = 'overlay' in video.stem.lower() or any('overlay' in p.lower() for p in relative.parts[:-1])
    kind = 'overlay' if is_overlay else 'comparison'
    if is_overlay:
        model = metadata.get('model')
        if not model:
            parent = video.parent.name
            model = parent.removeprefix('overlay_')
        models = [model]
    else:
        models = metadata.get('models')
        if not isinstance(models, list) or not models:
            models = [video.parent.name.removeprefix('video_')]
    return '__'.join([safe_name(scene), kind, *[safe_name(m) for m in models]]) + '.mp4'


def collect(source, output, dry_run=False):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise SystemExit(f'Source folder does not exist: {source}')
    if source == output:
        raise SystemExit('Output must be a separate folder from the source.')
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    used = set()
    videos = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4'
                    and not p.is_relative_to(output))
    for video in videos:
        # Only collect comparison and overlay videos, excluding unrelated MP4s.
        if not any(word in video.stem.lower() for word in ('comparison', 'overlay')):
            continue
        try:
            name = describe(video, source)
        except ValueError as error:
            print(f'Skipping {video}: {error}')
            continue
        if name.lower() in used:
            suffix = hashlib.sha256(video.relative_to(source).as_posix().encode()).hexdigest()[:10]
            name = f'{Path(name).stem}__{suffix}.mp4'
        used.add(name.lower())
        destination = output / name
        if destination.exists() and destination.stat().st_size == video.stat().st_size:
            skipped += 1
            continue
        if not dry_run:
            shutil.copy2(video, destination)
        copied += 1
        print(f'{"Would copy" if dry_run else "Copied"}: {name}')
    print(f'\n{"Would copy" if dry_run else "Copied"} {copied} videos; {skipped} already present.')
    print(f'Output: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='Download folder, or its scene_comparisons subfolder')
    parser.add_argument('--output', type=Path, help='Default: SOURCE/all_videos')
    parser.add_argument('--dry-run', action='store_true', help='List copies without writing files')
    args = parser.parse_args()
    collect(args.source, args.output or args.source / 'all_videos', args.dry_run)


if __name__ == '__main__':
    main()
