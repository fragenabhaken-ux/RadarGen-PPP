#!/usr/bin/env python3
"""Render the full DiT PPP checkpoint lineage as five training panels."""
import ast
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('Original training', 'PPP_train_2289081.err', 0, 20000),
    ('Resumed training', '2341693_PPP_train_resume_2289081.err', 0, 25068),
    ('Fine-tuning', '2356875_PPP_train_finetune_2341693.err', 25068, None),
]
PATTERN = re.compile(r'PPP update\s+(\d+):\s*(\{.*\})\s*$')
BETA = 0.99


def main():
    rows, boundaries = [], []
    for label, filename, offset, cutoff in SOURCES:
        records = {}
        for line in (ROOT / 'jsc_jupiter/logs' / filename).read_text().splitlines():
            match = PATTERN.search(line)
            if match:
                step = int(match[1])
                if cutoff is None or step <= cutoff:
                    records[step] = ast.literal_eval(match[2])
        if not records:
            raise ValueError(f'No updates in {filename}')
        boundaries.append((min(records) + offset, label))
        rows.extend((step + offset, record) for step, record in sorted(records.items()))
    x = [step for step, _ in rows]
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1, 1, 1, 0.8]})
    specs = [
        ('loss_total', 'Total', '#202938', None),
        ('loss_ppp_weighted', 'PPP intensity', '#2563eb', (5, 7)),
        ('loss_rcs_weighted', 'RCS', '#d97706', (3, 6)),
        ('loss_doppler_weighted', 'Doppler', '#059669', (1, 3)),
    ]
    for panel, (key, label, color, limits) in enumerate(specs):
        y = [record[key] for _, record in rows]
        smoothed, value = [], y[0]
        for loss in y:
            value = BETA * value + (1 - BETA) * loss
            smoothed.append(value)
        axes[0].plot(x, y, color=color, alpha=0.12, linewidth=0.5)
        axes[0].plot(x, smoothed, color=color, linewidth=1.8, label=label)
        if panel:
            ax = axes[panel]
            ax.plot(x, y, color=color, alpha=0.18, linewidth=0.5)
            ax.plot(x, smoothed, color=color, linewidth=1.8)
            ax.set(yscale='log', ylim=limits, ylabel=f'{label} loss',
                   title=f'{label} loss — log scale')
            ticks = {'PPP intensity': [5, 5.5, 6, 6.5, 7],
                     'RCS': [3, 4, 5, 6], 'Doppler': [1, 1.5, 2, 2.5, 3]}[label]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set(ylabel='Training loss', ylim=(0, 40), title='All losses — linear scale')
    axes[0].legend(ncol=4, loc='upper right')
    axes[4].plot(x, [record['lr'] for _, record in rows], color='#7c3aed', linewidth=1.5)
    axes[4].set(ylabel='Learning rate', xlabel='Cumulative optimizer iteration',
                title='Recorded optimizer learning rate — log scale')
    axes[4].set_yscale('log')
    axes[4].set_ylim(bottom=7e-7)
    for ax in axes:
        ax.grid(True, which='major', alpha=0.2)
        for step, label in boundaries[1:]:
            ax.axvline(step, color='#64748b', linestyle='--', linewidth=0.8)
        ax.set_xlim(0, x[-1])
    for step, label in boundaries[1:]:
        axes[0].text(step, 1.03, label, transform=axes[0].get_xaxis_transform(), fontsize=9)
    fig.suptitle('DiT backbone with PPP loss — full training history', fontsize=16)
    fig.text(0.5, 0.012, 'Faint: logged batch losses · Solid: EWMA (β=0.99 over logged records) · Dashed: training phase boundaries',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=1.5)
    destination = ROOT / 'output/plots/dit_ppp_training_five_panels'
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png',):
        fig.savefig(destination.with_suffix('.' + extension), dpi=200)
        print(destination.with_suffix('.' + extension))
    plt.close(fig)


if __name__ == '__main__':
    main()
