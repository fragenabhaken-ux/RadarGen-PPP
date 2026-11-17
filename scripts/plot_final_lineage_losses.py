#!/usr/bin/env python3
"""Plot total training loss and learning rate across the initial and final model lineages.

Use only updates leading to the checkpoint loaded by the next stage. Resume
counters are converted to new iterations before accumulating the lineage axis.
"""
import ast
import csv
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/assets/model_training_lineage'
# run, log name, nodes, source counter (zero after weights-only loading), last step
LINEAGES = {
    'DINO': [
        ('2382473', 'PPP_train_dinov3', 1, 0, 6305),
        ('2391378', 'PPP_train_dinov3_finetune_2382473', 1, 0, 6305),
        ('2658284', 'PPP_train_dino_shifted_1e4', 1, 0, 10000),
        ('2684031', 'PPP_continue_dino_2658284_4nodes', 4, 0, 9971),
        ('2773475', 'PPP_resume_dino_2684031_optimizer_4nodes', 4, 9971, 19914),
    ],
    'DiT': [
        ('2438928', 'PPP_train', 1, 0, 12534),
        ('2516862', 'PPP_train_finetune_cont_2438928', 1, 0, 12534),
        ('2658827', 'PPP_train_dit_joint_dino_512_constant_1e5', 1, 0, 18730),
        ('2684032', 'PPP_continue_dit_2658827_4nodes', 4, 0, 17822),
        ('2773477', 'PPP_resume_dit_2684032_optimizer_4nodes', 4, 17822, 35974),
    ],
}
LINEAGES['Initial DINO'] = LINEAGES['DINO'][:2]
LINEAGES['Initial DiT'] = [
    ('2289081', 'PPP_train', 1, 0, 20000),
    ('2341693', 'PPP_train_resume_2289081', 1, 20000, 25068),
    ('2356875', 'PPP_train_finetune_2341693', 1, 0, 12534),
]
PATTERN = re.compile(r'PPP update\s+(\d+):\s*(\{.*\})\s*$')
WARMUP = {
    '2289081': 200, '2341693': 0, '2356875': 200,
    '2382473': 200, '2391378': 200, '2658284': 100, '2684031': 100, '2773475': 0,
    '2438928': 200, '2516862': 200, '2658827': 0, '2684032': 100, '2773477': 0,
}
COLORS = ['#2563eb', '#0891b2', '#d97706', '#7c3aed', '#059669']


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for model, runs in LINEAGES.items():
        fig, (ax, lr_ax) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1.35]})
        offset, rows = 0, []
        for index, (run, name, nodes, source_step, final_step) in enumerate(runs):
            filename = 'PPP_train_2289081.err' if run == '2289081' else f'{run}_{name}.err'
            path = ROOT / 'jsc_jupiter/logs' / filename
            records = {}
            for line in path.read_text().splitlines():
                match = PATTERN.search(line)
                if not match:
                    continue
                step = int(match[1])
                if not source_step < step <= final_step:
                    continue
                record = ast.literal_eval(match[2])
                loss, lr = record['loss_total'], record['lr']
                if not math.isfinite(lr) or lr < 0:
                    raise ValueError(f'Invalid learning rate in {path}, step {step}')
                if not math.isfinite(loss) or loss <= 0:
                    raise ValueError(f'Invalid total loss in {path}, step {step}')
                records[step] = (loss, lr)
            if not records:
                raise ValueError(f'No losses found in {path}')
            steps = sorted(records)
            duration = final_step - source_step
            x = [offset + step - source_step for step in steps]
            y = [records[step][0] for step in steps]
            learning_rates = [records[step][1] for step in steps]
            color = COLORS[index]
            for panel in (ax, lr_ax):
                panel.axvspan(offset, offset + duration, color=color, alpha=0.045)
            # Zero LR at the first warmup update cannot be displayed on a log axis.
            lr_ax.plot(x, [lr if lr > 0 else float('nan') for lr in learning_rates],
                       color=color, linewidth=1.5)
            warmup = WARMUP[run]
            lr_ax.text(offset + duration / 2, 1.035,
                       f'Warm-up: {warmup} iterations' if warmup else 'No warm-up',
                       transform=lr_ax.get_xaxis_transform(), ha='center', va='bottom',
                       fontsize=9, color=color)
            if warmup:
                lr_ax.axvspan(offset, offset + warmup, color=color, alpha=0.25)
            ax.plot(x, y, color=color, alpha=0.20, linewidth=0.6)
            # Same smoothing horizon in optimizer iterations despite different log cadence.
            smooth, value, previous = [], y[0], steps[0]
            for step, loss in zip(steps, y):
                decay = math.exp(-(step - previous) / 200)
                value = decay * value + (1 - decay) * loss
                smooth.append(value)
                previous = step
                rows.append(dict(run=run, nodes=nodes, logged_step=step,
                                 run_iteration=step-source_step,
                                 cumulative_iteration=offset+step-source_step,
                                 loss_total=loss, lr=records[step][1], warmup_iterations=warmup,
                                 source=str(path.relative_to(ROOT))))
            ax.plot(x, smooth, color=color, linewidth=1.8)
            ax.text(offset + duration / 2, 1.035,
                    f'{run}\n{nodes} node'+('s' if nodes != 1 else '')+f' · {duration:,} iterations',
                    transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=9, color=color, fontweight='bold')
            if index:
                for panel in (ax, lr_ax):
                    panel.axvline(offset, color='#64748b', linestyle='--', linewidth=0.9)
            if index == 2 and not model.startswith('Initial'):
                change = ('New neck/head + continuous targets\n512 → 1024 output; backbone unfrozen'
                          if model == 'DINO' else 'New joint head\n32-channel output bottleneck removed')
                ax.annotate(change, xy=(offset, 29), xytext=(offset+duration*0.12, 31),
                            arrowprops=dict(arrowstyle='->', color='#475569'),
                            fontsize=8, color='#334155',
                            bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.95))
            print(f'{model} {run}: {len(steps)} records, steps {steps[0]}–{steps[-1]}, '
                  f'axis {offset}–{offset+duration}, {nodes} nodes')
            offset += duration
        ax.set_yscale('log')
        upper = max(45, 1.08 * max(row['loss_total'] for row in rows))
        ax.set_ylim(8.5, upper)
        ax.yaxis.set_major_locator(FixedLocator([tick for tick in [9, 10, 12, 15, 20, 30, 40, 60, 100] if tick < upper]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.set(xlim=(0, offset), ylabel='Total training loss (log scale)')
        lr_ax.set(yscale='log', xlabel='Cumulative optimizer iterations across the displayed stages',
                  ylabel='Learning rate (log scale)')
        lr_ax.grid(axis='y', which='major', alpha=0.2)
        ax.grid(axis='y', which='major', alpha=0.2)
        title = (f'{model} — total loss and learning rate' if model.startswith('Initial')
                 else f'{model} — total loss and learning rate across the final training lineage')
        fig.suptitle(title, fontsize=15, y=0.98)
        fig.text(0.5, 0.025,
                 'Faint: logged batch loss · Dark: EWMA (200-iteration time constant; reset at each run)\n'
                 'Dashed: run boundaries · Dark shading in LR panel: warm-up · LR values are logged, not smoothed\n'
                 'Stage width = optimizer iterations, not elapsed time or equal data exposure; zero LR is omitted on the log axis',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.83, hspace=0.27)
        stem = DEST / f'{model.lower().replace(" ", "_")}_lineage_total_loss'
        fig.savefig(stem.with_suffix('.png'), dpi=170)
        plt.close(fig)
        with stem.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
