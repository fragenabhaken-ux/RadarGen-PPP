"""Compare full-keyframe validation runs, never mixing partial or SS50 results."""
import argparse
import json
from pathlib import Path
import tempfile

from compare_ss50_selected import METRIC_KEYS

RUNS = [
    ('RadarGen', 'radargen'),
    ('DINO four-node (2684031)', 'dino_2684031'),
    ('DINO final (2744861)', 'dino_2744861'),
    ('DiT four-node (2684032)', 'dit_2684032'),
    ('DiT final (2744862)', 'dit_2744862'),
]


def generate(root, output):
    summaries, identities, inventories = [], [], []
    lines = ['# Full validation metrics comparison', '',
             'All annotated validation keyframes; no SS50 subsampling. Prediction archives are stored separately for each checkpoint.', '',
             '| Model | Evaluated | Expected | Status |', '| --- | --- | --- | --- |']
    for label, name in RUNS:
        directory = root / name
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        summaries.append(summary)
        count = summary['evaluated_samples'] if summary else 0
        expected = summary['expected_samples'] if summary else '—'
        complete = bool(summary and summary['complete'])
        lines.append(f'| {label} | {count} | {expected} | {"Complete" if complete else "Pending / incomplete"} |')
        if complete:
            provenance = json.loads((directory / 'validation_config.json').read_text())
            if (provenance.get('max_samples') is not None or not provenance.get('keyframes_only') or
                    provenance.get('reference_validation_records')):
                raise ValueError(f'Not an unrestricted full-keyframe run: {directory}')
            if (count != expected or count != len(provenance['selected_indices']) or
                    provenance['selected_indices'] != sorted(provenance['selected_indices'])):
                raise ValueError(f'Incomplete or subsampled keyframe inventory: {directory}')
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            if len(rows) != count:
                raise ValueError(f'Metric records disagree with summary: {directory}')
            identities.append([(r['scene'], r['frame']) for r in rows])
            inventories.append(provenance['inventory_sha256'])
    if not all(s and s['complete'] for s in summaries):
        lines += ['', 'Metric tables will appear once all five runs complete and their frame identities match.']
    else:
        if any(ids != identities[0] for ids in identities) or len(set(inventories)) != 1:
            raise ValueError('Full-validation frame identities or inventories differ')
        table_lines = []
        template_path = Path(__file__).resolve().parents[1] / 'docs/SS50_selected_comparison.md'
        for line in template_path.read_text().splitlines():
            if line.startswith('| '):
                table_lines.append(line.split('|')[1:3])
            elif table_lines:
                break
        template = table_lines[2:]
        assert len(template) == len(METRIC_KEYS)
        lines += ['', f'All five models evaluated the same {len(identities[0])} annotated validation frames in the same order (verified).', '',
                  'Mean ± population standard deviation; scalar hit rates, class-wise MMD, and median runtime. **Bold marks first place only** within each table using unrounded values; exact ties share first place. ↓ lower is better; ↑ higher is better; — unavailable. Rows without arrows are not ranked.', '',
                  'Class-wise MMD retains the existing evaluator policy: every third nonempty frame, up to 200 frames per class and side. Foreground Chamfer requires nonempty clouds on both sides; other foreground metrics retain their existing denominators.']
        for title, indices in [('DINO: four-node versus final', (1, 2)),
                               ('DiT: four-node versus final', (3, 4)),
                               ('DINO: four-node versus final versus RadarGen', (0, 1, 2)),
                               ('DiT: four-node versus final versus RadarGen', (0, 3, 4))]:
            lines += ['', f'## {title}', '', '| Scope | Metric | ' + ' | '.join(RUNS[i][0] for i in indices) + ' |',
                      '| ' + ' | '.join(['---'] * (2 + len(indices))) + ' |']
            for (scope, metric), key in zip(template, METRIC_KEYS):
                scope, metric = scope.strip(), metric.strip()
                values = [summaries[i]['metrics'].get(key, summaries[i]['foreground'].get(key)) for i in indices]
                if 'Median inference' in metric:
                    values = [v.get('median') if isinstance(v, dict) else v for v in values]
                means = [v.get('mean') if isinstance(v, dict) else v for v in values]
                available = [v for v in means if v is not None]
                ranked = ('↓' in metric or '↑' in metric) and len(available) >= 2
                best = (min if '↓' in metric else max)(available) if ranked else None
                cells = []
                for value, mean in zip(values, means):
                    cell = '—' if mean is None else (f"{value['mean']:.3f} ± {value['std']:.3f}" if isinstance(value, dict) else f'{value:.3f}')
                    if ranked and mean == best:
                        cell = f'**{cell}**'
                    cells.append(cell)
                lines.append('| ' + ' | '.join([scope, metric, *cells]) + ' |')
    lines += ['', '## Artifacts', '']
    for label, name in RUNS:
        directory = root / name
        lines.append(f'- {label}: [{directory.name}/summary.json]({directory}/summary.json), [predictions.h5]({directory}/predictions.h5), [validation_config.json]({directory}/validation_config.json).')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as stream:
        stream.write('\n'.join(lines) + '\n')
        temporary = Path(stream.name)
    temporary.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.root, args.output))
