#!/usr/bin/env python3
"""Full-data four-GPU memory probe followed by a fresh DiT LR sweep."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def probe_summary(rows, batch, returncode):
    training = [r for r in rows if r['phase'] == 'train']
    result = dict(batch_per_gpu=batch, world_size=4, effective_batch=batch*4,
                  returncode=returncode, successful_updates=len(training), safe=False)
    if returncode == 0 and len(training) == 5:
        total = min(r['gpu_total_gib'] for r in training)
        allocated = max(r['gpu_peak_allocated_gib'] for r in training)
        reserved = max(r['gpu_peak_reserved_gib'] for r in training)
        device_used = max(r['gpu_total_gib'] - r['gpu_free_gib'] for r in training)
        result.update(rank0_peak_allocated_gib=allocated, rank0_peak_reserved_gib=reserved,
                      rank0_device_used_gib=device_used, gpu_total_gib=total,
                      safe=max(reserved, device_used) <= 0.85 * total)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--backbone-checkpoint', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside the supplied Slurm four-GPU allocation')
    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.config).read_text())
    (root / 'submitted_config.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    launch = ['srun', '--ntasks=1', '--cpus-per-task=288', '--cpu-bind=verbose', '--kill-on-bad-exit=1',
              sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4',
              'scripts/train_ppp.py']
    attempts = []
    selected = None
    for batch in (8, 4, 2):
        directory = root / f'batch_probe_b{batch}'
        directory.mkdir(exist_ok=True)
        config = copy.deepcopy(base)
        config['train'].update(train_batch_size=batch, lr_schedule='constant',
                               lr_schedule_args={'num_warmup_steps':0}, training_hours=0.15)
        config['train']['optimizer']['lr'] = 1e-6
        config['train']['extra'].update(lr_range_test=False, prediction_interval=0)
        config['work_dir'] = config['train']['work_dir'] = str(directory)
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f'Probing batch {batch}/GPU: five full DDP optimizer updates; logs {directory}', flush=True)
        with (directory/'train.out').open('w') as out, (directory/'train.err').open('w') as err:
            result = subprocess.run(launch + ['--config_path', str(path), '--work_dir', str(directory),
                                    '--init-backbone-from', args.backbone_checkpoint, '--max-updates', '5'],
                                    stdout=out, stderr=err)
        memory_path = directory/'gpu_memory.jsonl'
        rows = [json.loads(line) for line in memory_path.read_text().splitlines() if line] if memory_path.exists() else []
        summary = probe_summary(rows, batch, result.returncode)
        attempts.append(summary)
        (root/'batch_probe_summary.json').write_text(json.dumps(dict(attempts=attempts), indent=2)+'\n')
        print('Batch probe: '+json.dumps(summary), flush=True)
        if summary['safe']:
            selected = batch
            break
        if result.returncode:
            errors = (directory/'train.err').read_text()
            if 'out of memory' not in errors.lower():
                print(errors[-12000:], flush=True)
                raise RuntimeError('Batch probe failed for a reason other than CUDA OOM')
    if selected is None:
        raise RuntimeError('No tested batch retained 15% GPU memory headroom')
    config = copy.deepcopy(base)
    config['train']['train_batch_size'] = selected
    directory = root/'sweep'
    directory.mkdir(exist_ok=True)
    config['work_dir'] = config['train']['work_dir'] = str(directory)
    path = directory/'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = dict(selected_batch_per_gpu=selected, world_size=4, effective_batch=selected*4,
                   output_resolution=base['model']['image_size'] * base['model'].get('extra', {}).get('output_scale', 1), memory_headroom_fraction=0.15, attempts=attempts,
                   note='Five successful DDP updates verified all ranks; memory fields report rank zero.')
    (root/'batch_probe_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Starting fresh 600-update LR sweep, 1e-7 -> 1e-2; '+json.dumps(summary), flush=True)
    with (root/'lr_range.out').open('w') as out, (root/'lr_range.err').open('w') as err:
        result = subprocess.run(launch + ['--config_path', str(path), '--work_dir', str(directory),
                                '--init-backbone-from', args.backbone_checkpoint], stdout=out, stderr=err)
    plot = subprocess.run([sys.executable, 'scripts/plot_ppp_lr_range.py', '--log', str(root/'lr_range.err'),
                           '--output', str(root/'lr_range/lr_loss.png'), '--smoothing-beta', '0.95'])
    (root/'run_status.json').write_text(json.dumps(dict(sweep_exit_code=result.returncode,
                                                       plot_exit_code=plot.returncode), indent=2)+'\n')
    return result.returncode or plot.returncode


if __name__ == '__main__':
    raise SystemExit(main())
