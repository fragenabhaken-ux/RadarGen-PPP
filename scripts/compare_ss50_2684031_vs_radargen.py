"""Compare every run 2684031 checkpoint tied for most wins against RadarGen."""
import json
from pathlib import Path

from compare_ss50_selected import METRIC_KEYS, ROOT


def generate():
    manifest = json.loads(Path('evaluation/configs/SS50_dino_2684031_checkpoint_sweep/manifest.json').read_text())
    baseline_dir = ROOT / '2404702_RadarGen_validation'
    baseline = json.loads((baseline_dir / 'summary.json').read_text())
    template = []
    for line in Path('docs/SS50_selected_comparison.md').read_text().splitlines():
        if line.startswith('| '):
            template.append([v.strip() for v in line.split('|')[1:3]])
        elif template:
            break
    template = template[2:]
    assert len(template) == len(METRIC_KEYS)

    def raw(summary, key, label):
        value = summary['metrics'].get(key, summary['foreground'].get(key))
        if 'Median inference' in label and isinstance(value, dict):
            return value.get('median')
        return value

    def mean(value):
        return value.get('mean') if isinstance(value, dict) else value

    def identities(directory):
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 50
        return [(r['scene'], r['frame']) for r in rows]

    reference = identities(baseline_dir)
    assert baseline['complete'] and baseline['evaluated_samples'] == 50
    step15_dir = ROOT / 'SS50_dino_2773475_checkpoint_sweep/step_015000'
    step15 = json.loads((step15_dir / 'summary.json').read_text())
    assert step15['complete'] and step15['evaluated_samples'] == 50
    assert identities(step15_dir) == reference
    candidates = []
    for entry in manifest['entries']:
        if entry['kind'] != 'finetuned':
            continue
        directory = Path(entry['results'])
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['complete'] and summary['evaluated_samples'] == 50
        assert identities(directory) == reference, directory
        wins = losses = ties = 0
        for (_, label), key in zip(template, METRIC_KEYS):
            a, b = mean(raw(summary, key, label)), mean(raw(baseline, key, label))
            if not ('↑' in label or '↓' in label) or a is None or b is None:
                continue
            if a == b:
                ties += 1
            elif (a < b if '↓' in label else a > b):
                wins += 1
            else:
                losses += 1
        candidates.append((wins, summary['metrics']['distance_attr_f1']['mean'], entry, summary, losses, ties))
    max_wins = max(c[0] for c in candidates)
    selected = sorted((c for c in candidates if c[0] == max_wins), key=lambda c: c[2]['step'])
    tied_steps = ', '.join(str(c[2]['step']) for c in selected)
    lines = ['# Run 2684031: checkpoints with most metric wins versus RadarGen', '',
             f'Steps **{tied_steps}** tie for the most wins: **{max_wins} of 27 comparable metric rows** each. Each checkpoint is compared independently with RadarGen below.', '',
             'Selection searched all 10 saved checkpoints of run 2684031, excluding the initial-training baseline. '
             'Each directional row with values for both models counts equally, using unrounded means/scalars. '
             'Rows without a direction and unavailable RadarGen PPP metrics are excluded. '
             'Mean and median runtime count as two rows, matching the existing comparison table.', '',
             'All evaluations contain the same 50 frames in the same order (verified). Selection is on SS50. '
             'RadarGen uses evaluator version 5 and DINO version 6; mark MAE retains detection-level versus target-cell semantics. '
             'Win counts describe the recorded metrics and do not imply statistical significance.', '',
             'Mean ± population standard deviation. Bold compares run 2684031 with RadarGen; underline compares step 15000 (run 2773475) with run 2684031. '
             'Each pair is ranked independently using unrounded values; exact ties mark both. A run 2684031 value can be both bold and underlined. '
             '↓ lower is better; ↑ higher is better; — unavailable.']
    for wins, f1, entry, summary, losses, ties in selected:
        lines += ['', f"## Step {entry['step']} versus RadarGen", '',
                  f"Epoch {entry['epoch']}; checkpoint `{Path(entry['checkpoint']).name}`. "
                  f'**{wins} wins, {losses} losses, {ties} ties**; whole-area DA F1 {f1:.6f}.', '',
                  f"| Scope | Metric | DINO step {entry['step']} (2684031) | RadarGen | DINO step 15000 (2773475) |", '| --- | --- | --- | --- | --- |']
        for (scope, label), key in zip(template, METRIC_KEYS):
            values = [raw(s, key, label) for s in (summary, baseline, step15)]
            means = [mean(v) for v in values]
            directional = '↓' in label or '↑' in label
            winners = []
            for pair in ((0, 1), (0, 2)):
                valid = directional and all(means[i] is not None for i in pair)
                best = (min if '↓' in label else max)(means[i] for i in pair) if valid else None
                winners.append({i for i in pair if means[i] == best} if valid else set())
            cells = []
            for i, (v, avg) in enumerate(zip(values, means)):
                cell = '—' if avg is None else f"{v['mean']:.3f} ± {v['std']:.3f}" if isinstance(v, dict) else f'{v:.3f}'
                if i in winners[0]:
                    cell = f'<strong>{cell}</strong>'
                if i in winners[1]:
                    cell = f'<u>{cell}</u>'
                cells.append(cell)
            lines.append('| ' + ' | '.join([scope, label, *cells]) + ' |')
    lines += ['', '## Sources', '', f'- RadarGen: [summary]({baseline_dir}/summary.json).',
              f'- DINO step 15000 (2773475): [summary]({step15_dir}/summary.json); checkpoint `epoch_68_step_15000.pth`.']
    for _, _, entry, _, _, _ in selected:
        lines.append(f"- DINO step {entry['step']}: [summary]({entry['results']}/summary.json); checkpoint `{entry['checkpoint']}`.")
    lines.append('- Selection candidates: [sweep manifest](../evaluation/configs/SS50_dino_2684031_checkpoint_sweep/manifest.json).')
    output = Path('docs/SS50_dino_2684031_most_wins_vs_radargen.md')
    output.write_text('\n'.join(lines) + '\n')
    print(output)
    print(f"Compared steps {tied_steps}: {max_wins} wins each")


if __name__ == '__main__':
    generate()
