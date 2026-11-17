#!/usr/bin/env python3
"""Confirm entry-node batch estimate on four GPUs; run a fresh 30-minute LR sweep."""
import argparse
import ast
import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import yaml


def sweep_updates(timestamps, window_seconds):
    if len(timestamps) < 3:
        raise ValueError('Need at least three DDP optimizer timestamps to calibrate sweep length')
    deltas = [(b-a).total_seconds() for a,b in zip(timestamps,timestamps[1:])]
    if min(deltas) <= 0:
        raise ValueError('Optimizer timestamps must increase')
    # Mean observed cadence; guard caps training even if throughput changes.
    seconds_per_update = sum(deltas)/len(deltas)
    return max(2,int(window_seconds/seconds_per_update)),seconds_per_update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--recommendation',required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires a four-GPU Slurm allocation')
    root=args.work_dir;root.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(Path(args.config).read_text())
    recommendation=json.loads(Path(args.recommendation).read_text())
    direct = base['model']['model'] in ('DinoPPP', 'ResNetPPP')
    if not direct and (base['model']['model'] != 'RadarGenPPP_600M_P1_D28'
                       or base['model']['extra'].get('decoder_type') != 'dc_ae_no_bottleneck'):
        raise ValueError('Unsupported ablation predictor')
    if recommendation['model'] != base['name'].removesuffix('_lr_range'):
        raise ValueError('Entry-node batch recommendation belongs to a different ablation')
    (root/'submitted_config.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
    (root/'entry_batch_recommendation.json').write_text(json.dumps(recommendation,indent=2)+'\n')
    launch=['srun','--ntasks=1','--cpus-per-task=288','--cpu-bind=verbose','--kill-on-bad-exit=1',
            sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
            'scripts/train_ppp.py']
    batch=recommendation['selected_batch_per_gpu']; attempts=[]; selected=None;cadence=None
    while batch>=1:
        directory=root/f'ddp_probe_b{batch}';directory.mkdir(exist_ok=False)
        config=copy.deepcopy(base)
        config['train'].update(train_batch_size=batch,lr_schedule='constant',
                               lr_schedule_args={'num_warmup_steps':0},training_hours=0.1)
        config['train']['optimizer']['lr']=1e-6
        config['train']['extra']['lr_range_test']=False
        config['work_dir']=config['train']['work_dir']=str(directory)
        path=directory/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
        print(f'Four-GPU confirmation: batch {batch}/GPU, five updates; {directory}',flush=True)
        with (directory/'train.out').open('w') as out,(directory/'train.err').open('w') as err:
            result=subprocess.run(launch+['--config_path',str(path),'--work_dir',str(directory),
                                          '--max-updates','5'],stdout=out,stderr=err)
        memory=directory/'gpu_memory.jsonl'
        rows=[json.loads(l) for l in memory.read_text().splitlines() if l] if memory.exists() else []
        rows=[r for r in rows if r['phase']=='train']
        peak=max((max(r['gpu_peak_allocated_gib'],r['gpu_peak_reserved_gib'])*1024**3 for r in rows),default=None)
        attempt=dict(batch_per_gpu=batch,returncode=result.returncode,successful_updates=len(rows),
                     rank0_peak_bytes=peak,safe=result.returncode==0 and len(rows)==5 and peak<90_000_000_000)
        attempts.append(attempt);(root/'ddp_probe_summary.json').write_text(json.dumps(attempts,indent=2)+'\n')
        print(json.dumps(attempt),flush=True)
        errors=(directory/'train.err').read_text()
        if attempt['safe']:
            timestamps=[dt.datetime.fromisoformat(m[1]) for m in re.finditer(
                r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+[+]00:00) PPP update \d+:',errors)]
            updates,cadence=sweep_updates(timestamps,1800)
            selected=batch;break
        if result.returncode and 'out of memory' not in errors.lower():
            raise RuntimeError(f'DDP probe failed for a non-memory reason: see {directory}/train.err')
        batch=batch-4 if batch>4 else batch-1
    if selected is None:
        raise RuntimeError('No batch passed the four-GPU memory confirmation')
    config=copy.deepcopy(base)
    config['train']['train_batch_size']=selected
    config['train']['lr_schedule_args']['updates']=updates
    config['train']['training_hours']=0.5
    directory=root/'sweep';directory.mkdir(exist_ok=False)
    config['work_dir']=config['train']['work_dir']=str(directory)
    path=directory/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    calibration=dict(batch_per_gpu=selected,world_size=4,effective_batch=4*selected,
                     seconds_per_update=cadence,planned_updates=updates,training_window_seconds=1800,
                     lr_start=1e-7,lr_end=1e-2,warmup_updates=0,
                     note='Update count estimated from DDP cadence; 1800-second guard can stop before endpoint.')
    (root/'lr_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print('Starting fresh LR sweep: '+json.dumps(calibration),flush=True)
    started=time.monotonic()
    with (root/'lr_range.out').open('w') as out,(root/'lr_range.err').open('w') as err:
        result=subprocess.run(launch+['--config_path',str(path),'--work_dir',str(directory)],stdout=out,stderr=err)
    metrics=[]
    for line in (root/'lr_range.err').read_text().splitlines():
        match=re.search(r'PPP update\s+(\d+):\s*(\{.*\})\s*$',line)
        if match:
            record=ast.literal_eval(match[2]);record['update']=int(match[1]);metrics.append(record)
    plots=root/'lr_range';plots.mkdir(exist_ok=True)
    (plots/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    plot=subprocess.run([sys.executable,'scripts/plot_ppp_lr_range.py','--log',str(root/'lr_range.err'),
                         '--output',str(plots/'lr_loss.png'),'--smoothing-beta','0.95'])
    status=dict(sweep_exit_code=result.returncode,plot_exit_code=plot.returncode,
                process_elapsed_seconds=time.monotonic()-started,logged_updates=len(metrics),
                last_lr=metrics[-1]['lr'] if metrics else None,
                reached_lr_endpoint=bool(metrics and metrics[-1]['lr']>=0.999e-2))
    (root/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('Result: '+json.dumps(status),flush=True)
    return result.returncode or plot.returncode


if __name__=='__main__':
    raise SystemExit(main())
