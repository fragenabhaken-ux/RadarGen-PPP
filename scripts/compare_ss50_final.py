"""Compare one final SS50 run against archived four-node and RadarGen results."""
import argparse
import json
from pathlib import Path
from compare_ss50_selected import METRIC_KEYS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [
        (args.label, args.new_run),
        ('4-node DINO (2684031)', ROOT / '2701615_PPP_validation_ss50_2684031'),
        ('4-node DiT (2684032)', ROOT / '2701616_PPP_validation_ss50_2684032'),
        ('RadarGen', ROOT / '2404702_RadarGen_validation'),
    ]
    summaries, identities = [], []
    for _, directory in runs:
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['complete'] or summary['evaluated_samples'] != 50:
            raise ValueError(f'Incomplete SS50 run: {directory}')
        summaries.append(summary)
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        identities.append([(r['scene'], r['frame']) for r in rows])
    if any(ids != identities[0] for ids in identities):
        raise ValueError('SS50 sample identities or order differ')
    template = []
    for line in Path('docs/SS50_selected_comparison.md').read_text().splitlines():
        if line.startswith('| '):
            template.append(line.split('|')[1:3])
        elif template:
            break
    template = template[2:]
    if len(template) != len(METRIC_KEYS):
        raise ValueError('Metric template mismatch')
    lines = [f'# {args.label}: SS50 comparison', '',
             'All four runs evaluated the same 50 frames in the same order (verified).',
             'Mean ± population standard deviation. ↓ lower is better; ↑ higher is better.',
             'RCS L2 training losses are not compared with earlier L1 losses; these are physical validation metrics.', '',
             '| Scope | Metric | ' + ' | '.join(label for label, _ in runs) + ' |',
             '| ' + ' | '.join(['---'] * 6) + ' |']
    for (scope, label), key in zip(template, METRIC_KEYS):
        cells = []
        for summary in summaries:
            value = summary['metrics'].get(key, summary['foreground'].get(key))
            if 'Median inference' in label and isinstance(value, dict):
                value = value.get('median')
            cells.append('—' if value is None else
                         f"{value['mean']:.3f} ± {value['std']:.3f}" if isinstance(value, dict)
                         else f'{value:.3f}')
        lines.append('| ' + ' | '.join([scope.strip(), label.strip(), *cells]) + ' |')
    lines += ['', 'Result directories:', ''] + [f'- {label}: `{directory}`' for label, directory in runs]
    args.output.write_text('\n'.join(lines) + '\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
