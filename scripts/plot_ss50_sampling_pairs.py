"""Render the five requested SS50 metric pairs as PNG and PDF figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'output/paper_material/count_sweep_evaluation/ss50_sampling_budget'
DEST = ROOT / 'paired_plots'
PAIRS = [
 ('01_cd_location_iou', 'metrics', [('chamfer_distance_xy', 'CD location ↓'), ('iou_th1', 'IoU @ 1 m ↑')]),
 ('02_recall_precision', 'metrics', [('distance_attr_recall', 'Recall ↑'), ('distance_attr_precision', 'Precision ↑')]),
 ('03_mmd_rcs_doppler', 'metrics', [('pcl_mmd_rcs', 'MMD RCS ↓'), ('pcl_mmd_doppler', 'MMD Doppler ↓')]),
 ('04_foreground_cd', 'foreground', [('per_box_chamfer_distance_xy', 'Foreground CD location ↓'), ('per_box_chamfer_distance_all_norm', 'Foreground CD full ↓')]),
 ('05_density_similarity_hit_rate', 'foreground', [('per_box_num_points_similarity', 'Density similarity ↑'), ('hit_rate', 'Hit rate ↑')]),
]


def value(summary, scope, key):
    entry = summary[scope][key]
    return entry['mean'] if isinstance(entry, dict) else entry


def main():
    DEST.mkdir(exist_ok=True)
    groups = {}
    for path in sorted(ROOT.glob('budget_*/summary.json')):
        budget = int(path.parent.name.split('_')[1])
        summary = json.loads(path.read_text())
        assert summary['complete'] and summary['evaluated_samples'] == 50
        groups.setdefault(budget, []).append(summary)
    assert len(groups) == 20 and all(len(v) == 3 for v in groups.values())
    baselines = [
        ('DINO final', 'dino', 'o', '#636363'),
        ('DINO 15K', 'dino_15k', 'D', '#d95f02'),
        ('RadarGen', 'radargen', 's', '#7570b3'),
    ]
    summaries = {key: json.loads((ROOT / f'baseline_{key}.json').read_text()) for _, key, _, _ in baselines}
    gt = value(summaries['dino'], 'metrics', 'gt_count')
    budgets = sorted(groups)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    for filename, scope, metrics in PAIRS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for ax, (key, label) in zip(axes, metrics):
            seed_values = np.array([[value(s, scope, key) for s in groups[b]] for b in budgets])
            means, sd = seed_values.mean(axis=1), seed_values.std(axis=1)
            ax.plot(budgets, means, color='#1b6ca8', lw=2, label='DINO final: scaled PPP')
            ax.fill_between(budgets, means-sd, means+sd, color='#1b6ca8', alpha=.18, label='±1 SD across 3 seeds')
            ax.axvline(gt, color='#777777', ls=':', lw=1.5, label=f'GT mean: {gt:,.2f}')
            for name, baseline, marker, color in baselines:
                summary = summaries[baseline]
                count = value(summary, 'metrics', 'generated_count')
                ax.scatter(count, value(summary, scope, key), marker=marker, color=color,
                           edgecolor='white', linewidth=.7, s=85, zorder=5,
                           label=f'{name}: {count:,.2f} points')
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.set_xlabel('Expected mean points per frame')
            ax.set_xlim(50, 2050)
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, .015))
        fig.suptitle('SS50 · DINO final checkpoint 19,914 · shared intensity scaling', fontsize=13)
        fig.tight_layout(rect=(0, .19, 1, .94))
        for extension in ['png', 'pdf']:
            fig.savefig(DEST / f'{filename}.{extension}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    text = '# SS50 paired metric plots\n\nEach figure contains two metric panels. Curves use the final DINO checkpoint (19,914), with means and ±1 population SD across three seeds. Baseline markers use original single-run metrics and realized mean counts; the vertical line marks the ground-truth mean count.\n\n'
    for filename, _, metrics in PAIRS:
        title = ' and '.join(label for _, label in metrics)
        text += f'- [{title}]({filename}.png) · [PDF]({filename}.pdf)\n'
    (DEST / 'README.md').write_text(text)
    print(f'Saved five PNG/PDF pairs in {DEST}')


if __name__ == '__main__':
    main()
