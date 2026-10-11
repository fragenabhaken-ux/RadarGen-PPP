#!/usr/bin/env python3
"""Plot PPP loss versus the recorded optimizer learning rate.

An input file must be supplied as an argument: use --log with the Slurm
.err file from the current training implementation. Example:

    python scripts/plot_ppp_lr_range.py --log ppp_lr_range_<JOBID>.err

For an older run that wrote structured metrics instead, use:

    python scripts/plot_ppp_lr_range.py --metrics path/to/lr_range/metrics.jsonl

The log must contain PPP update records with both 'lr' and 'loss_total'.
Learning rates are never inferred from update numbers. Log-based plots show
only logged updates (normally every 20), whereas JSONL may record every update.
Interrupted files are supported; incomplete or nonfinite records are skipped.
Output defaults to <input-stem>_lr_loss.png beside the input file. Use --output
to select another path. --smoothing-beta controls EWMA over recorded points;
0 disables smoothing. No training models, CUDA, or dataset access are required.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys


def read_points(path, structured):
    points = []
    skipped = 0
    pattern = re.compile(r'PPP update\s+(\d+):\s*(\{.*\})\s*$')
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            if not structured:
                if 'PPP update' not in line:
                    continue
                match = pattern.search(line)
                if match is None:
                    skipped += 1
                    continue
            try:
                record = json.loads(line) if structured else ast.literal_eval(match.group(2))
                lr, loss = record['lr'], record['loss_total']
                if (type(lr) not in (int, float) or type(loss) not in (int, float)
                        or not math.isfinite(lr) or lr <= 0 or not math.isfinite(loss)):
                    raise ValueError('Nonfinite or invalid LR/loss')
                if structured and record.get('optimizer_applied') is False:
                    skipped += 1
                    continue
                points.append((lr, loss))
            except (ValueError, SyntaxError, KeyError, TypeError):
                skipped += 1
    if not points:
        raise ValueError("No finite update records with both 'lr' and 'loss_total'. "
                         'Older logs without LR require their metrics.jsonl via --metrics.')
    return points, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--log', type=Path, help='Required input option: Slurm .err update log')
    source.add_argument('--metrics', type=Path, help='Alternative input: older LR metrics JSONL')
    parser.add_argument('--output', type=Path, help='Output image path (default: beside input)')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Loss axis limits, e.g. --ylim 10 40')
    parser.add_argument('--smoothing-beta', type=float, default=0.995,
                        help='EWMA beta over recorded points, in [0,1); default 0.995')
    args = parser.parse_args()
    if not math.isfinite(args.smoothing_beta) or not 0 <= args.smoothing_beta < 1:
        parser.error('--smoothing-beta must be finite and in [0,1)')
    if args.ylim and (not all(math.isfinite(v) for v in args.ylim)
                      or args.ylim[0] >= args.ylim[1]):
        parser.error('--ylim requires finite MIN < MAX')
    path = args.log or args.metrics
    try:
        points, skipped = read_points(path, args.metrics is not None)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rates, losses = zip(*points)
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        ax.plot(rates, losses, alpha=0.5, label='Recorded total loss')
        if args.smoothing_beta:
            smooth = []
            value = losses[0]
            for loss in losses:
                value = args.smoothing_beta*value + (1-args.smoothing_beta)*loss
                smooth.append(value)
            ax.plot(rates, smooth, label=f'EWMA (smoothing beta={args.smoothing_beta:g})')
        ax.set(xscale='log', xlabel='Recorded optimizer learning rate', ylabel='Total PPP training loss',
               title=f'{path.name}: {len(points)} recorded updates')
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output = args.output or path.with_name(path.stem+'_lr_loss.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    print(f'Saved {output} ({len(points)} points, {skipped} skipped records, '
          f'smoothing beta={args.smoothing_beta:g})')
    if skipped:
        print('Incomplete, invalid, missing-LR or unapplied records were skipped.', file=sys.stderr)


if __name__ == '__main__':
    main()
