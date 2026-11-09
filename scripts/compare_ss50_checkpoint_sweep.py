"""Rank a fixed SS50 checkpoint sweep against its previous-training baseline."""
import argparse
import json
from pathlib import Path
import tempfile

from compare_ss50_selected import METRIC_KEYS


def generate(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['entries']
    results, identities, signatures = [], [], []
    for entry in entries:
        directory = Path(entry['results'])
        path = directory / 'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else None
        if summary and summary['complete']:
            config = json.loads((directory / 'validation_config.json').read_text())
            checkpoint = Path(entry['checkpoint']).resolve()
            stat = checkpoint.stat()
            if (summary['evaluated_samples'] != 50 or summary['expected_samples'] != 50 or
                    config['max_samples'] != 50 or config['validation_version'] != 6 or
                    Path(config['checkpoint_path']).resolve() != checkpoint or
                    config['checkpoint_size'] != stat.st_size or config['checkpoint_mtime_ns'] != stat.st_mtime_ns):
                raise ValueError(f'Unmatched SS50 checkpoint/provenance: {directory}')
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            if len(rows) != 50:
                raise ValueError(f'Incomplete SS50 records: {directory}')
            identities.append([(row['scene'], row['frame']) for row in rows])
            signatures.append((config['inventory_sha256'], config['seed'], config['selection_seed']))
        results.append(summary)
    if identities and (any(ids != identities[0] for ids in identities) or len(set(signatures)) != 1):
        raise ValueError('SS50 frame identities, inventory, or seeds differ')
    complete = all(s and s['complete'] for s in results)
    available = [i for i, s in enumerate(results) if s and s['complete']]
    score = lambda i: results[i]['metrics']['distance_attr_f1']['mean']
    ranking = sorted(available, key=score, reverse=True)
    lines = [manifest.get('title', '# DINO four-node fine-tuning: all-checkpoint SS50 comparison'), '',
             f"Training run: `{manifest.get('training_run', 2744861)}`. {manifest.get('baseline_description', 'Previous-training baseline: four-node DINO run `2684031`, final step 9,971.')}", '',
             '**Selection criterion:** highest whole-area distance-and-attribute detection F1 on the fixed SS50 benchmark. Exact ties share first place. Other metrics show trade-offs and do not change this selection rule.', '',
             'New evaluations produce metrics only: no figures, PNGs, TensorBoard output, or prediction exports. Existing matched final-checkpoint and baseline evaluations are reused.']
    if complete:
        winners = [i for i in ranking if score(i) == score(ranking[0])]
        tuned = [i for i in ranking if entries[i]['kind'] == 'finetuned']
        best_tuned = [i for i in tuned if score(i) == score(tuned[0])]
        baseline = next(i for i, entry in enumerate(entries) if entry['kind'] == 'baseline')
        delta = score(tuned[0]) - score(baseline)
        lines += ['', '**Best overall on SS50:** ' + ', '.join(entries[i]['label'] for i in winners) + f' (F1 = {score(winners[0]):.6f}).',
                  '', '**Best fine-tuned checkpoint:** ' + ', '.join(entries[i]['label'] for i in best_tuned) + f' (F1 = {score(best_tuned[0]):.6f}).',
                  '', f'Best fine-tuned versus previous-training final: F1 difference {delta:+.6f} ({delta * 100:+.3f} percentage points).',
                  '', 'All entries evaluated the same 50 frames in the same order, with the same inventory and seeds (verified). The best checkpoint is selected on SS50; it has not been selected using the full validation set.']
    else:
        lines += ['', f'**Sweep incomplete: {len(available)}/{len(entries)} evaluations complete.** Rankings below are provisional; the overall winner will be declared after every checkpoint finishes.']
    lines += ['', '## Ranking by whole-area F1', '',
              '| Rank | Checkpoint | Epoch | Step | F1 ↑ | Precision ↑ | Recall ↑ | CD location ↓ | Foreground hit rate ↑ |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    previous_score, rank = None, 0
    for position, i in enumerate(ranking, 1):
        entry, summary = entries[i], results[i]
        if score(i) != previous_score:
            rank, previous_score = position, score(i)
        metrics = summary['metrics']
        f1 = f"{score(i):.6f}"
        if rank == 1:
            f1 = f'**{f1}**'
        lines.append('| ' + ' | '.join([str(rank), entry['label'], str(entry['epoch']), str(entry['step']), f1,
            f"{metrics['distance_attr_precision']['mean']:.6f}", f"{metrics['distance_attr_recall']['mean']:.6f}",
            f"{metrics['chamfer_distance_xy']['mean']:.6f}", f"{summary['foreground']['hit_rate']:.6f}"]) + ' |')
    for i, entry in enumerate(entries):
        if i not in available:
            summary = results[i]
            status = f"Incomplete ({summary['evaluated_samples']}/50)" if summary else 'Pending'
            lines.append(f"| — | {entry['label']} | {entry['epoch']} | {entry['step']} | {status} | — | — | — | — |")
    lines += ['', '## All metrics', '',
              'Mean ± population standard deviation; hit rate, class-wise MMD, and median runtime are scalars. Bold marks first place within each metric using unrounded values. Rows without arrows are not ranked; — unavailable or pending. Rankings remain provisional until completion.', '',
              '| Scope | Metric | ' + ' | '.join(entry['label'] for entry in entries) + ' |',
              '| ' + ' | '.join(['---'] * (2 + len(entries))) + ' |']
    template = []
    for line in (Path(__file__).resolve().parents[1] / 'docs/SS50_selected_comparison.md').read_text().splitlines():
        if line.startswith('| '):
            template.append(line.split('|')[1:3])
        elif template:
            break
    template = template[2:]
    assert len(template) == len(METRIC_KEYS)
    for (scope, metric), key in zip(template, METRIC_KEYS):
        scope, metric = scope.strip(), metric.strip()
        values = [s['metrics'].get(key, s['foreground'].get(key)) if i in available else None for i, s in enumerate(results)]
        if 'Median inference' in metric:
            values = [v.get('median') if isinstance(v, dict) else v for v in values]
        means = [v.get('mean') if isinstance(v, dict) else v for v in values]
        valid = [v for v in means if v is not None]
        ranked = ('↓' in metric or '↑' in metric) and len(valid) >= 2
        best = (min if '↓' in metric else max)(valid) if ranked else None
        cells = []
        for value, mean in zip(values, means):
            cell = '—' if mean is None else f"{value['mean']:.3f} ± {value['std']:.3f}" if isinstance(value, dict) else f'{value:.3f}'
            if ranked and mean == best:
                cell = f'**{cell}**'
            cells.append(cell)
        lines.append('| ' + ' | '.join([scope, metric, *cells]) + ' |')
    if manifest.get('single_table', False):
        start = lines.index('## Ranking by whole-area F1')
        end = lines.index('## All metrics')
        del lines[start:end]
    lines += ['', '## Sources', '']
    for entry in entries:
        lines.append(f"- {entry['label']}: [{entry['results']}/summary.json]({entry['results']}/summary.json); checkpoint `{entry['checkpoint']}`.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as stream:
        stream.write('\n'.join(lines) + '\n')
        temporary = Path(stream.name)
    temporary.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.manifest, args.output))
