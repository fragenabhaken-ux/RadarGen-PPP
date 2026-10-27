"""Stage manifest-selected scene sweeps once; training workers only read files.

Builds a dataset root with permanent samples/metadata symlinks and extracted
camera/radar sweeps. Re-running resumes completed scenes under a global lock.
No training-time extraction or scene-dependent sampling is introduced.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


def stage_scene(root, scene, archive_roots):
    archives = [Path(folder) / f'{scene}.tar.zst' for folder in archive_roots]
    fingerprint = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in archives]
    marker = root / '.scenes' / f'{scene}.json'
    if marker.exists():
        if json.loads(marker.read_text()) != json.loads(json.dumps(fingerprint)):
            raise ValueError(f'Archive changed after staging: {scene}')
        return
    for archive in archives:
        with subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE) as process:
            with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
                for member in stream:
                    relative = PurePosixPath(member.name)
                    if member.isdir() and str(relative) == '.':
                        continue
                    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                            or relative.parts[0] != 'sweeps' or not (member.isdir() or member.isfile())):
                        raise ValueError(f'Unexpected archive entry: {member.name}')
                    destination = root.joinpath(*relative.parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        # Resume interrupted extraction only after byte-for-byte comparison.
                        with stream.extractfile(member) as source, destination.open('rb') as old:
                            first, second = hashlib.sha256(), hashlib.sha256()
                            while block := source.read(1024 * 1024): first.update(block)
                            while block := old.read(1024 * 1024): second.update(block)
                            if first.digest() != second.digest():
                                raise ValueError(f'Staged file differs from archive: {destination}')
                        continue
                    temporary = destination.with_name(destination.name + '.partial')
                    with stream.extractfile(member) as source, temporary.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    os.replace(temporary, destination)
            if process.wait() != 0:
                raise RuntimeError(f'zstd failed: {archive}')
    marker.write_text(json.dumps(fingerprint) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='output/ppp_manifests/truckscenes_train.json')
    parser.add_argument('--permanent-root', default='/e/scratch/nxtaim-1/public/truckscenes/man-truckscenes')
    parser.add_argument('--camera-archives', default='/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/camera_input_archives')
    parser.add_argument('--radar-archives', default='/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/radar_input_archives')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--scene', action='append', help='Restrict preparation to explicit inventory scenes')
    args = parser.parse_args()
    body = json.loads(Path(args.manifest).read_text())['body']
    scenes = sorted({r['expected']['scene_token'] for r in body['samples']})
    if args.scene:
        if set(args.scene) - set(scenes): raise ValueError('Scene outside selected inventory')
        scenes = sorted(set(args.scene))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ('samples', body['contract']['dataset_version']):
            source, target = Path(args.permanent_root).resolve() / name, root / name
            if target.is_symlink() and target.resolve() == source:
                continue
            if target.exists(): raise FileExistsError(target)
            target.symlink_to(source, target_is_directory=True)
        (root / '.scenes').mkdir(exist_ok=True)
        for scene in scenes:
            stage_scene(root, scene, (args.camera_archives, args.radar_archives))
            print(f'Staged {scene}', flush=True)


if __name__ == '__main__':
    main()
