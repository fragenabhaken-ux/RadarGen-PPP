#!/usr/bin/env python3
"""Plot historical PPP step schedules against the current cosine configs.

Warmup is omitted to compare epoch-scale decay independently of GPU count and
batch size. RadarGen's historical fine-tuning warmup is omitted as well.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def cosine_curve(config):
    train = config['train']
    duration = train['num_epochs']
    args = train['lr_schedule_args']
    peak = train['optimizer']['lr']
    floor = args['min_lr_ratio']
    start = args['decay_start_fraction']
    end = args.get('decay_end_fraction', 1.0)
    xs = [duration * i / 1000 for i in range(1001)]
    ys = [peak * (floor + (1 - floor) * .5 *
                  (1 + math.cos(math.pi * max(0., min(1., (x / duration - start) / (end - start))))))
          for x in xs]
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/lr_schedule_comparison'))
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = [yaml.safe_load((root / 'configs' / filename).read_text()) for filename in
               ('PPP_train.yaml', 'PPP_train_dinov3.yaml')]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    ax = axes[0]
    ax.step([0, 4, 8, 12], [2e-5, 1e-5, 1e-6, 1e-6], where='post',
            color='#7b8794', linewidth=2, label='Old: initial + fine-tuning (12 epochs)')
    ax.step([0, 2, 4, 6], [2e-5, 1e-5, 1e-6, 1e-6], where='post',
            color='#2563eb', linestyle='--', linewidth=2, label='Old compressed to 6 epochs')
    ax.plot(*cosine_curve(configs[0]), color='#e66b16', linewidth=2.6, label='New: cosine (6 epochs)')
    ax.axvline(6, color='#e66b16', alpha=.25, linewidth=1)
    ax.set(title='RadarGen: combined history compressed to 6 epochs', xlim=(0, 12),
           ylim=(0, 2.15e-5), xticks=range(0, 13, 2))
    ax = axes[1]
    ax.step([0, 40, 65], [1e-3, 1e-4, 1e-4], where='post',
            color='#2563eb', linestyle='--', linewidth=2, label='Old: initial run only')
    ax.plot(*cosine_curve(configs[1]), color='#e66b16', linewidth=2.6, label='New: cosine')
    ax.set(title='DINOv3: initial run, 65 epochs', xlim=(0, 65), ylim=(0, 1.08e-3))
    for ax in axes:
        ax.set_xlabel('Epoch (zero-based progress)')
        ax.set_ylabel('Learning rate')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', fontsize=9)
    fig.suptitle('PPP learning-rate schedules: old vs new', fontsize=16)
    fig.text(.5, .025, 'Warm-up ramps omitted; curves show epoch-scale hold/decay. '
             'Both new runs retain 200 warm-up optimizer updates.', ha='center', fontsize=10, color='#4b5563')
    fig.tight_layout(rect=(0, .07, 1, .94))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = options.output_dir / f'old_vs_new.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
