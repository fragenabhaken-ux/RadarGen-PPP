#!/usr/bin/env python3
"""Read back PPP TensorBoard events and fail on missing/nonfinite metrics or images."""
import argparse
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def check_events(directory, expected_step):
    events = EventAccumulator(str(directory), size_guidance={'scalars': 0, 'images': 0}).Reload()
    for tag in ('train/loss_total', 'train/lr', 'train/grad_norm'):
        rows = events.Scalars(tag)
        if not rows or rows[-1].step != expected_step:
            raise ValueError(f'{tag}: missing final update {expected_step}')
        if not all(math.isfinite(row.value) for row in rows):
            raise ValueError(f'{tag}: nonfinite events')
    images = [tag for tag in events.Tags()['images'] if tag.startswith('prediction/')]
    if not images or not any(row.step == expected_step for tag in images for row in events.Images(tag)):
        raise ValueError(f'Missing prediction image at update {expected_step}')
    print(f'TensorBoard verification passed: {directory}; update={expected_step}; '
          f'scalars={events.Tags()["scalars"]}; images={images}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', type=Path)
    parser.add_argument('--expected-step', type=int, default=20)
    options = parser.parse_args()
    check_events(options.work_dir / 'tensorboard', options.expected_step)
