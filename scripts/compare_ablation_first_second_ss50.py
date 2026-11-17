#!/usr/bin/env python3
"""Audit matching first/second final checkpoints; never select from full means."""
import json
from pathlib import Path


ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation')
PAIRS = [
    ('DiT without bottleneck', 'SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537',
     'SS50_ablation_dit_dcae_no_bottleneck_2867524_step58057_checkpoint_audit'),
    ('Compact DINO', 'SS50_ablation_dino_neck_compact_2849062_step10372',
     'SS50_ablation_dino_neck_compact_2867528_step20595_checkpoint_audit'),
    ('Random DINO', 'SS50_ablation_dino_random_init_2847666_step15000',
     'SS50_ablation_dino_random_init_2863217_step29915_checkpoint_audit'),
    ('ResNet-50', 'SS50_ablation_resnet50_backbone_2856174_step12575',
     'SS50_ablation_resnet50_backbone_2882612_final'),
]
HIGHER = {'distance_attr_f1', 'distance_attr_precision', 'distance_attr_recall', 'iou_th1',
          'per_box_num_points_similarity', 'ppp_hit_probability_gt_positive', 'hit_rate'}
NEUTRAL = {'generated_count', 'gt_count', 'inference_time_seconds', 'ppp_expected_total_count',
           'target_collision_count', 'ppp_expected_count', 'gt_positive_boxes', 'evaluated_boxes'}


def main():
    import numpy as np
    lines = ['# SS50: first versus second final checkpoints', '',
             'Same 50 scene/frame/seed tuples in the same order are required. Best unrounded '
             'means are bold. Lower is better except F1/precision/recall/IoU, similarity and '
             'hit probabilities/rates. Descriptive counts and timing are not ranked.', '']
    results = []
    for label, first, second in PAIRS:
        lines += [f'## {label}', '']
        if not (ROOT / second / 'summary.json').exists():
            lines += ['Second-final SS50 pending.', '']
            continue
        summaries = [json.loads((ROOT / name / 'summary.json').read_text()) for name in (first, second)]
        rows = [[json.loads(line) for line in (ROOT / name / 'metrics.jsonl').read_text().splitlines()]
                for name in (first, second)]
        identities = [[(r['scene'], r['frame'], r['seed']) for r in records] for records in rows]
        if identities[0] != identities[1] or len(rows[0]) != 50:
            raise ValueError(f'Unmatched SS50 records: {label}')
        if not all(s['complete'] and s['evaluated_samples'] == 50 for s in summaries):
            raise ValueError(f'Incomplete SS50: {label}')
        f1 = [s['metrics']['distance_attr_f1']['mean'] for s in summaries]
        result = dict(model=label, first_f1=f1[0], second_f1=f1[1], delta_f1=f1[1] - f1[0],
                      earlier_higher_f1=f1[0] > f1[1], paired_intervals={})
        lines += [f"F1: **{f1[0]:.4f} → {f1[1]:.4f}**; delta {f1[1]-f1[0]:+.4f}.", '',
                  '### Paired uncertainty check', '',
                  'Second minus first; 95% percentile intervals from 10,000 paired frame bootstrap '
                  'resamples (seed 42). Exploratory, without multiple-comparison correction.', '',
                  '| Metric | Mean difference | 95% interval |', '|---|---:|---|']
        sample_indices = np.random.default_rng(42).integers(0, 50, size=(10000, 50))
        for key in ('distance_attr_f1', 'chamfer_distance_xy', 'count_abs_error',
                    'rcs_mae_at_detections', 'doppler_mae_at_detections'):
            delta = np.array([b['metrics'][key] - a['metrics'][key] for a, b in zip(*rows)])
            low, high = np.quantile(delta[sample_indices].mean(axis=1), [.025, .975])
            result['paired_intervals'][key] = dict(delta=float(delta.mean()), low=float(low), high=float(high))
            lines.append(f'| {key} | {delta.mean():+.4f} | [{low:+.4f}, {high:+.4f}] |')
        results.append(result)
        for section in ('metrics', 'foreground'):
            lines += ['', f'### {section}', '', '| Metric | First final | Second final | Difference |',
                      '|---|---:|---:|---:|']
            for key in dict.fromkeys(k for s in summaries for k in s[section]):
                if key == 'mmd_sampling':
                    continue
                values = [s[section].get(key) for s in summaries]
                means = [v.get('mean') if isinstance(v, dict) else v for v in values]
                if not all(isinstance(v, (int, float)) for v in means):
                    continue
                best = (max if key in HIGHER else min)(means) if key not in NEUTRAL else None
                cells = [f'**{v:.4f}**' if best is not None and v == best else f'{v:.4f}' for v in means]
                lines.append('| ' + key + ' | ' + ' | '.join(cells) + f' | {means[1]-means[0]:+.4f} |')
        lines += ['', f'- [First summary]({ROOT / first}/summary.json).',
                  f'- [Second summary]({ROOT / second}/summary.json).', '']
    out = Path('output/ablations/first_second_checkpoint_audit')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    Path('docs/ablations/SS50_first_vs_second_final_checkpoints.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
