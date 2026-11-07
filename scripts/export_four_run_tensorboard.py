#!/usr/bin/env python3
"""Package the initial and four-node DINO/DiT runs for TensorBoard.

Run with the JSC TensorBoard module loaded. Original events are copied verbatim;
the initial DiT run is recovered from its training log and prediction outputs.
"""
import argparse
import ast
import datetime
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tarfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


RUNS = {
    'DINO_1node_2658284': '2658284_PPP_train_dino_shifted_1e4',
    'DiT_1node_2658827': '2658827_PPP_train_dit_joint_dino_512_constant_1e5',
    'DINO_4nodes_2684031': '2684031_PPP_continue_dino_2658284_4nodes',
    'DiT_4nodes_2684032': '2684032_PPP_continue_dit_2658827_4nodes',
}


def scalars(writer, prefix, values, step, wall_time):
    leaves = [Summary.Value(tag=f'{prefix}/{key}', simple_value=value)
              for key, value in values.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(value)]
    writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=leaves)))


def recover(source, destination, log):
    pattern = re.compile(r'(\S+) PPP update\s+(\d+):\s*(\{.*\})\s*$')
    points = {}
    for line in log.read_text().splitlines():
        match = pattern.search(line)
        if match:
            timestamp = match[1].removeprefix('INFO:ppp_training:')
            points[int(match[2])] = (datetime.datetime.fromisoformat(timestamp).timestamp(),
                                     ast.literal_eval(match[3]))
    if not points:
        raise ValueError(f'No training metrics: {log}')
    writer = EventFileWriter(str(destination))
    try:
        for step, (timestamp, values) in sorted(points.items()):
            scalars(writer, 'train', values, step, timestamp)
        for line in (source / 'prediction_images/metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            step, index = row['step'], row['dataset_index']
            png = source / f'prediction_images/step_{step:06d}_sample_{index}.png'
            timestamp = png.stat().st_mtime
            tag = f'prediction/sample_{index}'
            scalars(writer, tag, row, step, timestamp)
            data = png.read_bytes()
            if data[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError(f'Invalid PNG: {png}')
            width, height = struct.unpack('>II', data[16:24])
            image = Summary.Image(height=height, width=width, encoded_image_string=data)
            writer.add_event(Event(wall_time=timestamp, step=step,
                                   summary=Summary(value=[Summary.Value(tag=tag, image=image)])))
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train'))
    parser.add_argument('--output', type=Path, default=Path('output/tensorboard_four_runs'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for label, name in RUNS.items():
        source, destination = args.source / name, args.output / label
        if label == 'DiT_1node_2658827':
            recover(source, destination, Path(__file__).resolve().parents[1] /
                    f'jsc_jupiter/logs/{name}.err')
            provenance = 'Recovered from Slurm stderr metrics and prediction PNG/JSONL outputs'
        else:
            shutil.copytree(source / 'tensorboard', destination)
            provenance = 'Original TensorBoard events copied verbatim'
        shutil.copy2(source / 'submitted_config.yaml', destination / 'submitted_config.yaml')
        events = EventAccumulator(str(destination), size_guidance={'scalars': 0, 'images': 0}).Reload()
        tags = events.Tags()
        losses = events.Scalars('train/loss_total')
        for tag in tags['scalars']:
            if not all(math.isfinite(row.value) for row in events.Scalars(tag)):
                raise ValueError(f'Nonfinite scalar: {label}/{tag}')
        if not losses or not tags['images']:
            raise ValueError(f'Missing losses or images: {label}')
        report[label] = dict(source=str(source), provenance=provenance,
                             loss_records=len(losses), first_step=losses[0].step,
                             last_step=losses[-1].step, scalar_tags=len(tags['scalars']),
                             image_records=sum(len(events.Images(tag)) for tag in tags['images']))
        print(label, report[label], flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'README.txt').write_text(
        'Run: tensorboard --logdir tensorboard_four_runs --port 12356\n'
        'Four separate runs share one dashboard. Steps are original optimizer update counts.\n'
        'Initial DiT events were recovered; only originally saved metrics/images are available.\n'
        'Other events are unmodified. See manifest.json for provenance and verified counts.\n')
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(args.output, arcname=args.output.name)
    print(f'Archive: {archive}', flush=True)


if __name__ == '__main__':
    main()
