"""Generate focused SS50 comparisons from validation summaries."""
import json
from pathlib import Path

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation')
RUNS = [
    ('RadarGen', '2404702_RadarGen_validation'),
    ('Best previous DiT (2356875)', '2369104_PPP_validation_finetune_2356875'),
    ('New DiT joint DINO (2658827)', '2680507_PPP_validation_ss50_2658827'),
    ('New DINOv3 1024 (2658284)', '2680332_PPP_validation_ss50_2658284'),
    ('4-node DiT final (2684032)', '2701616_PPP_validation_ss50_2684032'),
    ('4-node DINOv3 final (2684031)', '2701615_PPP_validation_ss50_2684031'),
    ('4-node DINOv3 step 9000 (2684031)', '2701633_PPP_validation_ss50_dino_step9000'),
    ('DiT RCS L2 final (2719596)', '2735087_PPP_validation_ss50_2719596'),
    ('DINO RCS L2 final (2719598)', '2735088_PPP_validation_ss50_2719598'),
]
METRIC_KEYS = [
    'chamfer_distance_xy', 'chamfer_distance_all_norm', 'iou_th1',
    'distance_attr_recall', 'distance_attr_precision', 'distance_attr_f1',
    'pcl_mmd_xy', 'pcl_mmd_rcs', 'pcl_mmd_doppler',
    'per_box_chamfer_distance_xy', 'per_box_chamfer_distance_all_norm',
    'per_box_num_points_similarity', 'hit_rate', 'gt_count', 'generated_count',
    'count_abs_error', 'ppp_expected_total_count', 'ppp_total_count_abs_error',
    'rcs_mae_at_target_cells', 'doppler_mae_at_target_cells',
    'ppp_expected_count', 'ppp_expected_count_abs_error', 'ppp_hit_probability_gt_positive',
    *[f'invariant_mmd_{metric}_{cls}' for cls in ('car', 'truck', 'trailer')
      for metric in ('xy', 'rcs', 'doppler')],
    'inference_time_seconds', 'inference_time_seconds',
]


def generate(*, local_only=False):
    comparison = ROOT / 'SS50_metrics_comparison.md'
    local_path = Path('docs/SS50_selected_comparison.md')
    local = local_path.read_text()
    # Use only the first table as the shared metric-label template.
    table_lines = []
    for line in local.splitlines():
        if line.startswith('| '):
            table_lines.append(line)
        elif table_lines:
            break
    template = [line.split('|')[1:3] for line in table_lines][2:]
    assert len(template) == len(METRIC_KEYS) == 34
    summaries = [json.loads((ROOT / run / 'summary.json').read_text()) for _, run in RUNS]
    for summary in summaries:
        assert summary['complete'] and summary['evaluated_samples'] == 50
    identities = []
    for _, run in RUNS:
        rows = [json.loads(line) for line in (ROOT / run / 'metrics.jsonl').read_text().splitlines()]
        identities.append([(row['scene'], row['frame']) for row in rows])
    assert all(ids == identities[0] for ids in identities), 'SS50 sample identities differ'

    def table(indices, *, pairwise=False):
        lines = ['| Scope | Metric | ' + ' | '.join(RUNS[i][0] for i in indices) + ' |',
                 '| ' + ' | '.join(['---'] * (2 + len(indices))) + ' |']
        for (scope, label), key in zip(template, METRIC_KEYS):
            scope, label = scope.strip(), label.strip()
            values = []
            for i in indices:
                summary = summaries[i]
                value = summary['metrics'].get(key, summary['foreground'].get(key))
                if 'Median inference' in label and value is not None:
                    value = value.get('median')
                values.append(value)
            means = [v['mean'] if isinstance(v, dict) else v for v in values]
            available = [v for v in means if v is not None]
            ranked = sorted(set(available), reverse='↑' in label) if ('↓' in label or '↑' in label) else []
            best = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else None
            cells = []
            pair_winners = []
            if pairwise and ('↓' in label or '↑' in label):
                for pair in ((0, 5), (0, 8)):
                    candidates = [(i, means[indices.index(i)]) for i in pair]
                    if all(mean is not None for _, mean in candidates):
                        winner = (max if '↑' in label else min)(mean for _, mean in candidates)
                        pair_winners.append({i for i, mean in candidates if mean == winner})
                    else:
                        pair_winners.append(set())
            for i, value, mean in zip(indices, values, means):
                cell = '—' if value is None else (f"{value['mean']:.3f} ± {value['std']:.3f}" if isinstance(value, dict) else f'{value:.3f}')
                if pairwise:
                    if pair_winners and i in pair_winners[0]:
                        cell = f'<strong>{cell}</strong>'
                    if pair_winners and i in pair_winners[1]:
                        cell = f'<u>{cell}</u>'
                elif best is not None and mean == best:
                    cell = f'**{cell}**'
                elif second is not None and mean == second:
                    cell = f'<u>{cell}</u>'
                cells.append(cell)
            lines.append('| ' + ' | '.join([scope, label, *cells]) + ' |')
        return lines

    # Select by whole-area detection F1, consistently for both families and the new-model winner.
    score = lambda i: summaries[i]['metrics']['distance_attr_f1']['mean']
    best_dit = max((2, 4, 7), key=score)
    best_dino = max((3, 5, 8), key=score)
    best_new = max((best_dit, best_dino), key=score)
    lines = ['## Local SS50 results: focused model comparisons', '',
        'All models evaluated the same 50 validation keyframes in the same order (verified). Mean ± population standard deviation, rounded to three decimals. Except in the explicitly labeled pairwise DINO table, bold marks the best value and underline marks the second-best distinct value within each table, using unrounded means/scalars; exact ties share the same rank. ↓ lower is better; ↑ higher is better; — unavailable. Rows without arrows are not ranked.', '',
        'Selection rule: “best” means highest whole-area detection F1 among the initial, four-node continuation, and RCS L2 continuation checkpoints for each new model family. This favors balanced detection performance; no model wins every metric. The best previous DiT remains binary-target fine-tune run 2356875, not the continuous-target fine-tune run 2516862.', '',
        f'Selected new DiT: {RUNS[best_dit][0]}; selected new DINO: {RUNS[best_dino][0]}; best new model: {RUNS[best_new][0]}. These selections include both RCS L2 continuation checkpoints; the intermediate DINO step-9000 checkpoint is compared separately below.', '']
    sections = [
        ('DiT: initial run versus four-node continuation', [2, 4]),
        ('DINO: initial one-shot run versus four-node fine-tuning', [3, 5]),
        ('DINO: latest checkpoint (step 9971) versus step 9000', [5, 6]),
        ('DINO step 9000 and final checkpoint versus RadarGen', [6, 5, 0]),
        ('Final four-node DiT and DINO versus RadarGen', [4, 5, 0]),
        ('DiT: four-node checkpoint versus RCS L2 continuation', [4, 7]),
        ('DINO: four-node checkpoint versus RCS L2 continuation', [5, 8]),
        ('RCS L2 DiT and DINO versus previous four-node models and RadarGen', [7, 8, 4, 5, 0]),
        ('Best new DINO and DiT versus previous best DiT and RadarGen', [best_dino, best_dit, 1, 0]),
        ('Best new model versus RadarGen', [best_new, 0]),
    ]
    for title, indices in sections:
        lines += ['### ' + title, '', *table(indices), '']
    lines += ['### RadarGen versus four-node DINO and DINO RCS L2: independent pairwise comparisons', '',
        '**Bold** marks the better value between RadarGen and four-node DINO (2684031). '
        '<u>Underline</u> marks the better value between RadarGen and DINO RCS L2 (2719598). '
        'Each pair is compared independently using unrounded values; exact ties mark both values. '
        'A RadarGen value is both bold and underlined when it wins both comparisons. '
        'Rows without arrows and pairs with unavailable values are not highlighted.', '',
        *table([0, 5, 8], pairwise=True), '']
    lines += ['RCS L2 runs use identity-linked RCS with mean squared error and tenfold lower constant learning rates (DiT 1e-6; DINO 1e-5). These tables compare physical validation metrics, not L1 versus L2 training losses. Archived validation metrics are retained as originally computed; no historical runs were re-evaluated.', '', 'The new models use evaluator version 6 and continuous detection targets. RadarGen uses version 5; previous best DiT uses version 3 and binary targets. Mark MAE retains different target-cell versus detection-level semantics. Foreground Chamfer includes only boxes with both clouds nonempty. Previous best DiT inference timing was not recorded. Runtime excludes loading, metrics, and plotting.', '',
        'Checkpoints: initial DiT `epoch_17_step_18730.pth`; initial DINO `epoch_11_step_10000.pth`; four-node DiT `epoch_64_step_17822.pth` (validation 2701616); four-node DINO latest `epoch_45_step_9971.pth` (validation 2701615), compared with `epoch_41_step_9000.pth` (validation 2701633). RCS L2 DiT `epoch_65_step_18059.pth` (validation 2735087); RCS L2 DINO `epoch_45_step_9979.pth` (validation 2735088). DINO uses a 1024×1024 output grid; new DiT uses 512×512.', '', 'Sources:', '']
    lines += [f'- {name}: [summary]({ROOT / run / "summary.json"}).' for name, run in RUNS]
    section = '\n'.join(lines) + '\n'
    local_path.write_text('# Selected SS50 model comparison\n\n' + section)
    if not local_only:
        original = comparison.read_text()
        start = original.index('## Local SS50 results:')
        end = original.find('\n## ', start + 1)
        if end == -1:
            end = len(original)
        comparison.write_text(original[:start] + section + original[end:])
    print('Updated', local_path, 'and' if not local_only else '', comparison if not local_only else '')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Only update the repository comparison document")
    generate(local_only=parser.parse_args().local_only)
