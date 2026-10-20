#!/usr/bin/env python3
"""Plot logged PPP training losses on a logarithmic loss axis.

Repeat --segment LABEL LOG OFFSET LAST_STEP in training order. OFFSET maps local
steps to cumulative optimizer iterations; LAST_STEP excludes abandoned updates
(use 0 for no cutoff). Saves PNG, PDF, and the plotted records as CSV.
"""
import argparse
import ast
import csv
import math
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', nargs=4, action='append', required=True,
                        metavar=('LABEL', 'LOG', 'OFFSET', 'LAST_STEP'))
    parser.add_argument('--output', type=Path, required=True, help='Output stem')
    parser.add_argument('--title', default='DiT backbone with PPP loss')
    parser.add_argument('--beta', type=float, default=0.9, help='EWMA over logged records')
    args = parser.parse_args()
    if not 0 <= args.beta < 1:
        parser.error('--beta must be in [0, 1)')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator
    pattern = re.compile(r'PPP update\s+(\d+):\s*(\{.*\})\s*$')
    fields = ['loss_total', 'loss_ppp_weighted', 'loss_rcs_weighted', 'loss_doppler_weighted']
    names = ['Total loss', 'PPP intensity', 'RCS', 'Doppler']
    colors = ['#202938', '#2563eb', '#d97706', '#059669']
    rows = []
    boundaries = []
    for label, filename, offset, last in args.segment:
        offset, last = int(offset), int(last)
        points = {}
        for line in Path(filename).read_text().splitlines():
            match = pattern.search(line)
            if not match:
                continue
            step = int(match[1])
            if last and step > last:
                continue
            record = ast.literal_eval(match[2])
            values = [record[key] for key in fields]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f'Invalid log-scale loss at {filename}:{step}')
            points[step] = values
        if not points:
            raise ValueError(f'No training updates in {filename}')
        if rows and min(points) + offset <= rows[-1]['iteration']:
            raise ValueError('Segments overlap; specify the resume cutoff and offset')
        boundaries.append((min(points) + offset, label))
        for step, values in sorted(points.items()):
            rows.append(dict(iteration=step + offset, local_step=step, phase=label,
                             source=filename, **dict(zip(fields, values))))
        print(f'{label}: {len(points)} records, local steps {min(points)}–{max(points)}')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, name, color in zip(fields, names, colors):
        x = [row['iteration'] for row in rows]
        y = [row[key] for row in rows]
        ax.plot(x, y, color=color, alpha=0.16, linewidth=0.6)
        smooth, value = [], y[0]
        for loss in y:
            value = args.beta * value + (1 - args.beta) * loss
            smooth.append(value)
        ax.plot(x, smooth, color=color, linewidth=1.6, label=name)
    for step, label in boundaries[1:]:
        ax.axvline(step, color='#64748b', linestyle='--', linewidth=0.8)
        ax.text(step, 0.98, ' ' + label, transform=ax.get_xaxis_transform(),
                va='top', fontsize=9)
    ax.set(xlabel='Cumulative optimizer iteration', ylabel='Training loss (log scale)',
           yscale='log', title=args.title)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
    ax.grid(True, which='both', alpha=0.18)
    ax.legend(loc='upper right')
    fig.text(0.5, 0.015, f'Faint: logged batch losses · Solid: EWMA (β={args.beta:g}, over logged records)',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.output.with_suffix(suffix), dpi=200)
    plt.close(fig)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.output}.{{png,pdf,csv}}; {len(rows)} records')


if __name__ == '__main__':
    main()
