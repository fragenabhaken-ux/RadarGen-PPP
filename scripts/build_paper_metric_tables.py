#!/usr/bin/env python3
"""Build source-backed full-validation LaTeX tables with current model names."""
import json
from pathlib import Path

ROOT = Path('/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen')
OUT = Path('docs/ablations')


def load(path):
    return json.loads(Path(path).read_text())


def rows(include_first=False):
    entries = []
    def add(name, validation, detector, parameters, resolution, tag):
        directory = ROOT / 'validation' / validation
        summary = load(directory / 'summary.json')
        assert summary['complete'] and summary['evaluated_samples'] == 1906, directory
        det = load(ROOT / 'detection' / detector / 'metrics.json')
        assert det['full_archive'] and det['sample_count'] == 1906
        settings = json.loads(det['settings']['settings_json'])
        assert settings['height'] == 1 and settings['height_frame'] == 'flat-up'
        entries.append(dict(name=name, summary=summary, detector=det,
                            parameters=parameters, resolution=resolution, tag=tag,
                            source=str(directory / 'summary.json'),
                            detector_source=str(ROOT / 'detection' / detector / 'metrics.json')))
    add('RadarGen (released checkpoint)', 'FULL_validation_20261108/radargen',
        '2838586_radargen_nameorder_z1_flatup', 751.085100, 512, 'radargen')
    configurations = [
        ('C3P-dar-SANA (ours)', 'dit_dcae_no_bottleneck',
         2849053, 28537, 2867524, 58057, 'dit', 761.393162, 512),
        ('C3P-dar-Dv3 (ours), random init.', 'dino_random_init',
         2847666, 15000, 2863217, 29915, 'random', 239.863331, 1024),
        ('C3P-dar-Dv3 (ours), compact neck', 'dino_neck_compact',
         2849062, 10372, 2867528, 20595, 'compact', 80.610851, 1024),
        ('C3P-dar-RN50 (ours)', 'resnet50_backbone',
         2856174, 12575, 2882612, 25030, 'resnet', 239.031907, 1024)]
    for name, architecture, j1, s1, j2, s2, slug, params, resolution in configurations:
        if include_first:
            add(name + ', 12h', f'FULL_ablation_bundles/{architecture}_{j1}_step{s1}/metrics',
                f'ablation_checkpoint_programme/{slug}_first_{j1}/attempt_000',
                params, resolution, f'{slug}_first')
        bundle = f'{architecture}_{j2}_step{s2}' if slug != 'resnet' else f'{architecture}_{j2}_final'
        detector = (f'ablation_checkpoint_programme/{slug}_second_{j2}/attempt_000'
                    if slug != 'compact' else '2895299_dino_compact_2867528_step20595_z1_flatup')
        add(name + (', 24h' if include_first else ''), f'FULL_ablation_bundles/{bundle}/metrics',
            detector, params, resolution, f'{slug}_second')
        if slug == 'dit':
            add('C3P-dar-Dv3 (ours)', 'FULL_validation_20261108/dino_2773475',
                '2838595_dino_2773475_final_z1_flatup', 239.863331, 1024, 'dv3_reference')
    identities = []
    for entry in entries:
        records = Path(entry['source']).parent / 'metrics.jsonl'
        identities.append([(r['scene'], r['frame'], r['seed'])
                           for r in map(json.loads, records.read_text().splitlines())])
    assert all(v == identities[0] for v in identities)
    hashes = [e['detector']['sample_tokens_sha256'] for e in entries]
    assert len(set(hashes)) == 1
    return entries


def value(row, scope, key):
    if scope == 'efficiency':
        return row[key]
    if scope == 'detector':
        return row['detector']['summary'].get(key)
    if scope == 'class_ap':
        return row['detector']['classes'][key]['AP']
    v = row['summary'][scope].get(key)
    return v.get('mean') if isinstance(v, dict) else v


def render(entries, all_checkpoints=False):
    # Published values are supplied by the user; never included in rankings.
    paper = {
        'chamfer_distance_xy': (1.68, .39), 'distance_attr_recall': (.23, .12),
        'distance_attr_precision': (.26, .12), 'distance_attr_f1': (.24, .12),
        'pcl_mmd_xy': (.056, .062), 'iou_th1': (.31, .11),
        'pcl_mmd_rcs': (.09, .15), 'pcl_mmd_doppler': (.31, .74),
        'per_box_chamfer_distance_xy': (.95, .65),
        'per_box_num_points_similarity': (.51, .41), 'hit_rate': (.66, None),
        'invariant_mmd_xy_car': (.037, None), 'invariant_mmd_rcs_car': (.006, None),
        'invariant_mmd_doppler_car': (.014, None), 'invariant_mmd_xy_truck': (.024, None),
        'invariant_mmd_rcs_truck': (.031, None), 'invariant_mmd_doppler_truck': (.060, None),
        'invariant_mmd_xy_trailer': (.0069, None), 'invariant_mmd_rcs_trailer': (.022, None),
        'invariant_mmd_doppler_trailer': (.046, None), 'mAP': (.11, None), 'NDS': (.30, None)}
    text = ['% Requires: \\usepackage{booktabs,graphicx}',
            '% Generated from complete 1,906-frame summaries; rankings use unrounded means.',
            '% Names: C3P-dar-SANA, C3P-dar-Dv3, C3P-dar-RN50.', '']
    suffix = '_all_checkpoints' if all_checkpoints else ''

    def start(caption, label):
        text.extend([r'\begin{table*}[t]', r'\centering', r'\scriptsize' if all_checkpoints else r'\small',
                     r'\setlength{\tabcolsep}{5pt}', '\\caption{' + caption + '}',
                     '\\label{' + label + suffix + '}'])

    def finish(note):
        text.extend([r'\smallskip', r'\begin{minipage}{\textwidth}', r'\footnotesize', note,
                     r'\end{minipage}', r'\end{table*}', ''])

    def section(title, scope, columns):
        # columns: label, source key, decimals, rank direction (min/max/None)
        text.extend(['', '\\textbf{' + title + '}\\par\\smallskip',
                     r'\resizebox{\textwidth}{!}{%',
                     '\\begin{tabular}{l' + 'r' * len(columns) + '}', r'\toprule',
                     'Model & ' + ' & '.join(c[0] for c in columns) + r' \\', r'\midrule'])
        published = []
        for _, key, digits, _ in columns:
            if key not in paper:
                published.append('--')
            else:
                mean, sd = paper[key]
                published.append('$' + f'{mean:.{digits}f}' +
                                 (f' \\pm {sd:.{digits}f}' if sd is not None else '') + '$')
        text.append(r'RadarGen (paper)~\cite{borreda2026radargen} & ' +
                    ' & '.join(published) + r' \\')
        text.append(r'\midrule')
        ranks = []
        for _, key, _, direction in columns:
            vals = [value(row, scope, key) for row in entries]
            ranks.append(sorted(set(v for v in vals if v is not None), reverse=direction == 'max')
                         if direction else [])
        for row in entries:
            cells = []
            for index, (_, key, digits, direction) in enumerate(columns):
                mean = value(row, scope, key)
                if mean is None:
                    cells.append('--')
                    continue
                displayed = f'{mean:.{digits}f}'
                if ranks[index] and mean == ranks[index][0]:
                    displayed = r'\mathbf{' + displayed + '}'
                elif len(ranks[index]) > 1 and mean == ranks[index][1]:
                    displayed = r'\underline{' + displayed + '}'
                sd = None
                if scope in ('metrics', 'foreground'):
                    raw = row['summary'][scope].get(key)
                    if isinstance(raw, dict):
                        sd = raw.get('std')
                cells.append('$' + displayed + (f' \\pm {sd:.{digits}f}' if sd is not None else '') + '$')
            text.append(row['name'] + ' & ' + ' & '.join(cells) + r' \\')
        text.extend([r'\bottomrule', r'\end{tabular}%', '}', r'\medskip'])

    common = (r'Bold marks the best locally evaluated generated method by mean; underlining marks the second-best distinct mean. '
              r'Published RadarGen values are separate reference results and excluded from rankings. '
              r'A dash denotes an unavailable or inapplicable metric. '
              r'Standard deviations describe variation across frames or boxes, not confidence intervals or training-seed variation. ')
    budget = ('First- and second-allocation finals are both shown; 12h and 24h denote the allocated training budgets. '
              if all_checkpoints else 'Ablation rows use the final second-allocation checkpoints. ')
    checkpoint_note = (r'The C3P-dar-Dv3 reference uses run 2773475, step 19,914, not its best-SS50-F1 intermediate checkpoint. '
                       r'Training histories differ across architectures. ')
    start('Point-cloud generation results on MAN TruckScenes. Means $\\pm$ standard deviations across '
          '1,906 validation frames. Observed count: $1123.8 \\pm 316.9$ points/frame. '
          'DA denotes distance-and-attribute matching.', 'tab:generation_results')
    section('Count fidelity', 'metrics', [
        ('Mean generated count','generated_count',1,None), ('Mean expected count','ppp_expected_total_count',1,None),
        ('Count MAE $\\downarrow$','count_abs_error',1,'min'),
        ('Expected-count MAE $\\downarrow$','ppp_total_count_abs_error',1,'min')])
    section('Point-cloud geometry and correspondence', 'metrics', [
        ('Chamfer XY $\\downarrow$','chamfer_distance_xy',3,'min'),
        ('Chamfer all, norm. $\\downarrow$','chamfer_distance_all_norm',4,'min'),
        ('DA Recall $\\uparrow$','distance_attr_recall',3,'max'),
        ('DA Precision $\\uparrow$','distance_attr_precision',3,'max'),
        ('DA F1 $\\uparrow$','distance_attr_f1',3,'max'),
        ('Location MMD $\\downarrow$','pcl_mmd_xy',4,'min')])
    section('Spatial coverage', 'metrics', [('BEV IoU @ 1\\,m $\\uparrow$','iou_th1',3,'max')])
    section('Marks', 'metrics', [('RCS MMD $\\downarrow$','pcl_mmd_rcs',3,'min'),
        ('RCS MAE (dBsm) $\\downarrow$','rcs_mae_at_detections',2,'min'),
        ('Doppler MMD $\\downarrow$','pcl_mmd_doppler',3,'min'),
        ('Doppler MAE ($\\mathrm{m\\,s^{-1}}$) $\\downarrow$','doppler_mae_at_detections',2,'min')])
    finish(common + budget + checkpoint_note + r'Generated counts refer to sampled clouds; expected counts integrate PPP intensity. '
           r'Mark MAEs are evaluated at individual GT detections. Compatibility target-cell aliases are omitted because they duplicate these values.')
    start('Foreground evaluation on annotated object footprints. Per-box values are means $\\pm$ standard deviations; '
          'class-specific MMD and sampled hit rate are aggregate scores.', 'tab:foreground_results')
    section('Foreground geometry, count fidelity, and coverage', 'foreground', [
        ('Chamfer XY $\\downarrow$','per_box_chamfer_distance_xy',3,'min'),
        ('Chamfer all, norm. $\\downarrow$','per_box_chamfer_distance_all_norm',4,'min'),
        ('Density similarity $\\uparrow$','per_box_num_points_similarity',3,'max'),
        ('Object hit rate $\\uparrow$','hit_rate',4,'max')])
    section('Object-level PPP predictions', 'foreground', [
        ('Mean expected count','ppp_expected_count',3,None),
        ('Expected-count MAE $\\downarrow$','ppp_expected_count_abs_error',3,'min'),
        ('Hit probability, GT-positive $\\uparrow$','ppp_hit_probability_gt_positive',3,'max')])
    section('Class-specific foreground MMD', 'foreground', [
        (f'{kind.title()} {attribute.upper() if attribute in ("xy","rcs") else "Doppler"} $\\downarrow$',
         f'invariant_mmd_{attribute}_{kind}',4,'min')
        for kind in ('car','truck','trailer') for attribute in ('xy','rcs','doppler')])
    valid = '; '.join(row['name'] + ': ' + str(row['summary']['foreground']['per_box_chamfer_distance_xy']['valid_boxes'])
                      for row in entries)
    finish(common + r'Count similarity and expected-count errors use 11,891 boxes; analytical hit probabilities use '
           r'10,874 GT-positive boxes. Foreground Chamfer excludes boxes with an empty GT or generated cloud, so '
           r'its conditional means have different denominators. Valid-box counts: ' + valid + '. '
           r'Class MMD follows the existing evaluator: every third nonempty frame, up to 200 frames per class and side.')
    start('Frozen VoxelNeXt downstream detection and predictor computational cost.', 'tab:detection_efficiency')
    section('VoxelNeXt detection', 'detector', [
        ('mAP $\\uparrow$','mAP',4,'max'), ('NDS $\\uparrow$','NDS',4,'max'),
        ('mATE $\\downarrow$','mATE',3,'min'), ('mASE $\\downarrow$','mASE',3,'min'),
        ('mAOE $\\downarrow$','mAOE',3,'min'), ('mAVE $\\downarrow$','mAVE',3,'min'),
        ('mAAE $\\downarrow$','mAAE',3,'min')])
    section('VoxelNeXt class AP', 'class_ap', [('Car AP $\\uparrow$','car',4,'max'),
        ('Truck AP $\\uparrow$','truck',4,'max'), ('Trailer AP $\\uparrow$','trailer',4,'max')])
    section('Prediction-network size and output grid', 'efficiency', [
        ('Parameters (M) $\\downarrow$','parameters',1,'min'), ('Output grid side','resolution',0,None)])
    section('Generation latency (not hardware-controlled)', 'metrics', [
        ('Mean latency (s/frame)','inference_time_seconds',3,None)])
    finish(common + r'Parameter counts exclude external conditioning/text encoders and BEV preprocessing. '
           r'Latency uses precomputed conditioning and is not hardware-controlled; it is not ranked. '
           r'VoxelNeXt uses the same frozen epoch-400 checkpoint, physical archived clouds, $z=1$ in flat-up '
           r'coordinates before radar-frame conversion, and matched three-class scoring on 1,906 frames. '
           r'Equivalence to the original private preprocessing remains unverified. '
           r'The earlier local real-radar baseline is omitted because it used a different height convention.')
    return '\n'.join(text) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    for include_first in (False, True):
        entries = rows(include_first)
        filename = 'paper_metric_tables_all_checkpoints.tex' if include_first else 'paper_metric_tables.tex'
        (OUT / filename).write_text(render(entries, include_first))
        sources.append(dict(file=filename, rows=[{k:v for k,v in row.items()
                            if k not in ('summary','detector')} for row in entries]))
        print('Wrote', OUT / filename, flush=True)
    (OUT / 'paper_metric_table_sources.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
