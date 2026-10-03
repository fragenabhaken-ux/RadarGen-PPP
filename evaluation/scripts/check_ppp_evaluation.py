#!/usr/bin/env python3
"""PPP preflight: count empties and check analytical integration; no sampled metrics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from evaluation.config import EvaluationConfig
    from evaluation.evaluator import Evaluator
    from evaluation.ppp_metrics import make_ppp_grid_xy, compute_ppp_count_metrics, aggregate_ppp_metrics
    from evaluation.aggregation import create_aggregator
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='evaluation/configs/truckscenes_eval_ppp_smoke.yaml')
    args = parser.parse_args()
    config = EvaluationConfig.from_yaml(args.config)
    if len(config.models) != 1 or config.models[0].name != 'radargen_ppp':
        raise ValueError('PPP preflight requires exactly one PPP model')
    evaluator = Evaluator(config)
    model = evaluator.models[0]
    samples = evaluator._gather_samples()
    if not samples:
        raise ValueError('No samples selected')
    aggregator = create_aggregator()
    empty = 0
    for current, following, boxes, scene, frame in samples:
        gt = evaluator._get_evaluation_gt(current, following, scene, frame)
        points = model.predict_point_cloud(evaluator.adapter, current, following, scene, frame)
        fields = model.get_last_ppp_fields(scene, frame)
        mass = fields['cell_mass_grid']
        metrics = compute_ppp_count_metrics(mass, gt, boxes, make_ppp_grid_xy(*mass.shape, fields['coordinate_range']))
        aggregate_ppp_metrics(aggregator, metrics)
        empty += len(points) == 0
        print(json.dumps(dict(scene=scene, frame=frame, gt_count=len(gt), sampled_count=len(points),
                              **metrics)), flush=True)
    print(f'PPP analytical preflight passed: samples={len(samples)}, empty_clouds={empty}', flush=True)
    if empty:
        print('Sampled-metric evaluation is blocked until an explicit empty-cloud policy is defined.', flush=True)
        return 2
    print('No empty clouds in the selected set. Sampled metrics are not tested by this preflight.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
