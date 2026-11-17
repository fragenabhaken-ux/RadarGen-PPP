"""Compare complete, matched full-validation results for DINO step 15000 and RadarGen."""
import json
from pathlib import Path
import tempfile

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108')
OUTPUT = Path(__file__).resolve().parents[1] / 'docs/FULL_validation_step15000_vs_radargen.md'
RUNS = [('DINO finetuning step 15000 (2773475)', 'dino_2773475_step15000'), ('RadarGen', 'radargen')]


def generate():
    summaries = []
    lines = ['# Full validation: DINO step 15000 versus RadarGen', '',
             'All annotated validation keyframes, without SS50 subsampling. No PNGs are generated.', '',
             '| Model | Evaluated | Expected | Complete |', '| --- | --- | --- | --- |']
    for label, name in RUNS:
        path = ROOT / name / 'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else None
        summaries.append(summary)
        lines.append(f"| {label} | {summary['evaluated_samples'] if summary else 0} | {summary['expected_samples'] if summary else 'pending'} | {bool(summary and summary['complete'])} |")
    if all(s and s['complete'] for s in summaries):
        identities, inventories = [], []
        for (_, name), summary in zip(RUNS, summaries):
            directory = ROOT / name
            provenance = json.loads((directory / 'validation_config.json').read_text())
            if provenance.get('max_samples') is not None or not provenance.get('keyframes_only') or provenance.get('reference_validation_records'):
                raise ValueError(f'Not unrestricted full-keyframe validation: {name}')
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            count = summary['evaluated_samples']
            if count != summary['expected_samples'] or len(rows) != count or len(provenance['selected_indices']) != count:
                raise ValueError(f'Incomplete inventory: {name}')
            if provenance['selected_indices'] != sorted(provenance['selected_indices']):
                raise ValueError(f'Unexpected selection order: {name}')
            identities.append([(v['scene'], v['frame'], v['seed']) for v in rows])
            inventories.append(provenance['inventory_sha256'])
        if identities[0] != identities[1] or inventories[0] != inventories[1]:
            raise ValueError('Validation frames, seeds, or inventories differ')
        lines += ['', f'Verified identical {len(identities[0])} frames, order, seeds, and dataset inventory.', '',
                  'Mean ± population standard deviation where available. **Bold** marks the better mean; ↓ lower is better, ↑ higher is better. — unavailable. Metric names retain the evaluator keys.', '',
                  '| Scope | Metric | DINO step 15000 | RadarGen |', '| --- | --- | --- | --- |']
        for scope in ('metrics', 'foreground'):
            keys = list(dict.fromkeys(k for s in summaries for k in s[scope]))
            for key in keys:
                if key == 'mmd_sampling':
                    continue
                values = [s[scope].get(key) for s in summaries]
                means = [v.get('mean') if isinstance(v, dict) else v for v in values]
                lower = any(token in key for token in ('chamfer', 'mmd', 'abs_error', 'mae', 'time_seconds'))
                higher = key in ('iou_th1', 'distance_attr_precision', 'distance_attr_recall', 'distance_attr_f1', 'hit_rate', 'per_box_num_points_similarity', 'ppp_hit_probability_gt_positive')
                available = [v for v in means if v is not None]
                ranked = (lower or higher) and len(available) == 2
                best = (min(available) if lower else max(available)) if ranked else None
                cells = []
                for value, mean in zip(values, means):
                    cell = '—' if mean is None else f"{value['mean']:.6f} ± {value['std']:.6f}" if isinstance(value, dict) else f'{value:.6f}'
                    cells.append(f'**{cell}**' if ranked and mean == best else cell)
                arrow = ' ↓' if lower else ' ↑' if higher else ''
                lines.append('| ' + ' | '.join([scope, key + arrow, *cells]) + ' |')
                if any(isinstance(v, dict) and 'median' in v for v in values):
                    medians = [v.get('median') if isinstance(v, dict) else None for v in values]
                    lines.append('| ' + ' | '.join([scope, key + ' (median) ↓', *['—' if v is None else f'{v:.6f}' for v in medians]]) + ' |')
        lines += ['', 'Foreground Chamfer excludes boxes where either cloud is empty. Class-wise MMD uses the existing every-third-nonempty-frame policy, up to 200 frames per class and side. PPP detection-level mark MAE and RadarGen target-cell mark MAE have different target semantics. Runtime excludes loading, metrics, and rendering.']
    else:
        lines += ['', 'Validation is pending or incomplete. Metric comparisons will be generated automatically after completion; partial results are not compared against the complete baseline.']
    lines += ['', '## Sources', '']
    for label, name in RUNS:
        directory = ROOT / name
        lines.append(f'- {label}: [summary]({directory}/summary.json), [provenance]({directory}/validation_config.json), [predictions]({directory}/predictions.h5).')
    with tempfile.NamedTemporaryFile('w', dir=OUTPUT.parent, delete=False) as stream:
        stream.write('\n'.join(lines) + '\n')
        temporary = Path(stream.name)
    temporary.replace(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    generate()
