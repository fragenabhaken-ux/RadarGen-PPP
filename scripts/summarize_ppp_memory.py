#!/usr/bin/env python3
"""Summarize training GPU peaks and a provisional batch-one extrapolation."""
import argparse
import json
import math
from pathlib import Path


def summarize(rows, headroom=0.15):
    training = [r for r in rows if r['phase'] == 'train']
    steady = [r for r in training if r['update'] >= 6] or [r for r in training if r['update'] >= 2]
    if not steady:
        raise ValueError('Need at least two optimizer updates for a memory summary')
    batches = {r['microbatch_size'] for r in steady}
    if batches != {1}:
        raise ValueError('This extrapolation requires measured microbatch one')
    total = min(r['gpu_total_gib'] for r in steady)
    fixed = max(r['gpu_step_start_allocated_gib'] for r in steady)
    # Deliberately scale all step-dependent memory, including gradient buffers
    # and optimizer workspaces. This is a rough starting point, not a fit to
    # measured activation growth or a promise of the maximum fitting batch.
    extra = max(r['gpu_peak_allocated_gib'] - r['gpu_step_start_allocated_gib'] for r in steady)
    external = max(max(0, r['gpu_total_gib'] - r['gpu_free_gib'] - r['gpu_reserved_gib']) for r in steady)
    candidate = max(0, math.floor((total * (1 - headroom) - external - fixed) / extra)) if extra > 0 else None
    return dict(
        measured_microbatch=1, measured_accumulation=steady[-1]['accumulation'],
        optimizer_updates=len(training), steady_updates=len(steady),
        total_gib=total, step_start_allocated_gib=fixed, step_extra_allocated_gib=extra,
        peak_allocated_gib=max(r['gpu_peak_allocated_gib'] for r in training),
        peak_reserved_gib=max(r['gpu_peak_reserved_gib'] for r in training),
        estimated_non_torch_gib=external, headroom_fraction=headroom,
        provisional_microbatch_candidate=candidate,
        extrapolated_allocated_gib={str(b): fixed + b * extra for b in (1, 2, 4, 8, 16)},
        limitation='Single-batch extrapolation only; gradients/workspaces also contribute to the '
                   'scaled term. Validate larger batches with full data, diagnostics and DDP before use.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir')
    args = parser.parse_args()
    root = Path(args.work_dir)
    rows = [json.loads(line) for line in (root / 'gpu_memory.jsonl').read_text().splitlines() if line]
    summary = summarize(rows)
    (root / 'gpu_memory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
