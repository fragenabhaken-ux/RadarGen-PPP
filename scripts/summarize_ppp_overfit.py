#!/usr/bin/env python3
"""Summarize deterministic tiny-subset evaluations; no convergence claim."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir')
    args=parser.parse_args()
    directory=Path(args.work_dir)/'overfit_diagnostics'
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line]
    steps=sorted({r['step'] for r in rows})
    if not steps:
        raise ValueError('No diagnostics recorded')
    keys=('loss_total','loss_ppp_raw','loss_rcs_raw','loss_doppler_raw','expected_count',
          'active_cell_count','continuous_gt_count','count_abs_error_active',
          'rcs_mae_at_targets','doppler_mae_at_targets','mass_fraction_at_target_cells')
    grouped={step:{(r['scene'],r['frame']):r for r in rows if r['step']==step} for step in steps}
    identities=set(grouped[steps[0]])
    if any(set(group)!=identities for group in grouped.values()):
        raise ValueError('Incomplete or inconsistent diagnostic sample sets')
    means={key:[float(np.mean([r[key] for r in grouped[step].values()])) for step in steps] for key in keys}
    for key in keys:
        print(f'{key}: {means[key][0]:.6g} -> {means[key][-1]:.6g}')
    print(f'Observed steps: {steps[0]} -> {steps[-1]}; samples: {len(identities)}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for key in ('loss_ppp_raw','loss_rcs_raw','loss_doppler_raw'):
        axes[0,0].plot(steps,means[key],label=key)
    for key in ('expected_count','active_cell_count','continuous_gt_count'):
        axes[0,1].plot(steps,means[key],label=key)
    for key in ('rcs_mae_at_targets','doppler_mae_at_targets'):
        axes[1,0].plot(steps,means[key],label=key)
    axes[1,1].plot(steps,means['mass_fraction_at_target_cells'],label='Mass fraction at GT cells')
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(directory/'learning_curves.png',dpi=140); plt.close(fig)
    print('Review per-sample images and curves; no automatic overfit pass threshold is imposed.')


if __name__=='__main__':
    main()
