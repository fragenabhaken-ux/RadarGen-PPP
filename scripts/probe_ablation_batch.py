#!/usr/bin/env python3
"""Short single-GPU, real-data optimizer probe; conservatively extrapolate to DDP."""
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def recommend(rows, parameter_bytes, total_bytes, external_bytes, limit_bytes=90_000_000_000):
    """Fit an upper linear envelope; add 10% uncertainty and DDP/context allowance."""
    points = [(r['batch_per_gpu'], max(r['peak_allocated_bytes'], r['peak_reserved_bytes'])) for r in rows]
    if len(points) < 2:
        raise ValueError('At least two successful batch sizes are needed')
    slope = max((yb-ya)/(xb-xa) for xa,ya in points for xb,yb in points if xb>xa)
    if slope <= 0:
        raise ValueError('Memory slope must be positive')
    intercept = max(y-slope*x for x,y in points)
    ddp_bytes = 2 * parameter_bytes + 2 * 1024**3
    cap = min(limit_bytes, 0.90 * total_bytes)
    def predicted(batch):
        return 1.10 * (intercept+slope*batch) + ddp_bytes + external_bytes
    batch = min(64, math.floor(((cap-ddp_bytes-external_bytes)/1.10-intercept)/slope))
    batch = (batch // 4)*4 if batch >= 4 else batch
    if batch < 1:
        raise ValueError('No batch fits the conservative per-GPU budget')
    return dict(selected_batch_per_gpu=batch, world_size=4, effective_batch=4*batch,
                predicted_per_gpu_bytes=predicted(batch), per_gpu_limit_bytes=cap,
                slope_bytes_per_sample=slope, intercept_bytes=intercept,
                ddp_allowance_bytes=ddp_bytes, uncertainty_fraction=0.10,
                external_bytes=external_bytes,
                note='Extrapolation, not four-GPU verification; DDP replicates model/state on each GPU.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', nargs='+', type=int, default=[1,4,8])
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--dedicated-target', action='store_true',
                        help='Estimate for exclusive Slurm GPUs; exclude unrelated entry-node usage')
    args = parser.parse_args()
    if len(set(args.batches)) < 2 or min(args.batches) < 1 or args.updates < 2:
        parser.error('Use two or more positive batches and at least two optimizer updates')
    args.output.mkdir(parents=True, exist_ok=False)
    import pyrallis
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusion.utils.config import SanaConfig
    from radargen.training.ppp_initialization import initialize_dino_ppp_model, initialize_ppp_model
    from radargen.training.radargen_ppp_dataset import build_radargen_ppp_dataset_from_config
    from radargen.training.ppp_collate import collate_ppp_samples
    from radargen.losses.ppp_loss import PPPLoss
    from scripts.train_ppp import training_microbatch, validate_ppp_config, load_local_text_provider
    with open(args.config) as handle:
        config = pyrallis.load(SanaConfig, handle)
    validate_ppp_config(config)
    direct = config.model.model in ('DinoPPP', 'ResNetPPP')
    if not direct and (config.model.model != 'RadarGenPPP_600M_P1_D28'
                       or config.model.extra.get('decoder_type') != 'dc_ae_no_bottleneck'):
        raise ValueError('Unsupported ablation predictor')
    accelerator = Accelerator(mixed_precision='bf16')
    if accelerator.device.type != 'cuda' or accelerator.num_processes != 1:
        raise RuntimeError('Run this short probe on exactly one CUDA GPU')
    set_seed(config.train.seed)
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv'],text=True)
    (args.output/'gpu_inventory.csv').write_text(inventory)
    (args.output/'source_config.yaml').write_text(Path(args.config).read_text())
    free,total = torch.cuda.mem_get_info()
    external = total-free
    print(f'Building {config.model.model}; initialization settings: {config.model.extra}', flush=True)
    if direct:
        components = initialize_dino_ppp_model(config, device=accelerator.device)
        text_provider = None
    else:
        config.work_dir = config.train.work_dir = str(args.output)
        components = initialize_ppp_model(config, device=accelerator.device,
            null_embed_path='output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth')
        text_provider = load_local_text_provider(config, accelerator.device,
            '/e/project1/nxtaim-1/huber7/pretrained/gemma-2-2b-it')
    model = components.model.train()
    encoder = components.condition_encoder
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if config.name == 'ablation_dino_random_init':
        assert parameters == trainable == 239863331, (parameters, trainable)
    parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    optargs = dict(config.train.optimizer); optargs['lr']=1e-6
    if direct:
        optimizer_type = getattr(torch.optim,optargs.pop('type'))
        optimizer = optimizer_type((p for p in model.parameters() if p.requires_grad), **optargs)
    else:
        from diffusion.utils.optimizer import build_optimizer
        optimizer = build_optimizer(model,optargs)
    print('Loading real continuous-target samples from training manifest.', flush=True)
    dataset = build_radargen_ppp_dataset_from_config(config, config_path=args.config)
    indices = [i*len(dataset)//max(args.batches) for i in range(max(args.batches))]
    samples = [dataset[i] for i in indices]
    resolution = config.model.image_size * getattr(model, 'output_scale', 1)
    ppp_loss = PPPLoss(out_img_size=(resolution,resolution))
    rows=[]
    backbone = model.backbone if direct else model.dit
    backbone_weight = next(p for p in backbone.parameters() if p.requires_grad)
    before = backbone_weight.detach().clone()
    for batchsize in args.batches:
        batch = collate_ppp_samples(samples[:batchsize])
        batch = {k:v.to(accelerator.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        torch.cuda.empty_cache()
        measurements=[]
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started=time.monotonic()
            loss,logs = training_microbatch(model,encoder,batch,text_provider,ppp_loss,config,accelerator)
            accelerator.backward(loss)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.gradient_clip)
            if not torch.isfinite(norm): raise ValueError('Nonfinite gradient norm')
            optimizer.step(); torch.cuda.synchronize()
            measurement=dict(batch_per_gpu=batchsize,update=update+1,loss_total=float(loss.detach()),
                elapsed_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            measurements.append(measurement)
            with (args.output/'updates.jsonl').open('a') as handle: handle.write(json.dumps(measurement)+'\n')
            print(json.dumps(measurement),flush=True)
            del loss,logs
        # Last two updates include populated AdamW state; retain maximum across all updates.
        row=dict(batch_per_gpu=batchsize,successful_updates=args.updates,
                 peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measurements),
                 peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measurements),
                 steady_update_seconds=max(r['elapsed_seconds'] for r in measurements[1:]))
        rows.append(row)
        optimizer.zero_grad(set_to_none=True)
        del batch
        gc.collect()
    if torch.equal(before,backbone_weight.detach()): raise AssertionError('Backbone weights did not update')
    result=recommend(rows,parameter_bytes,total,0 if args.dedicated_target else external)
    result.update(model=config.name,parameters=parameters,trainable_parameters=trainable,
                  observed_entry_external_bytes=external,gpu_total_bytes=total,
                  dedicated_target=args.dedicated_target,
                  batches=rows,sample_indices=indices,optimizer=config.train.optimizer['type'],precision='bf16',
                  output_resolution=resolution,backbone_update_verified=True,limit_convention='90 GB decimal per GPU')
    (args.output/'batch_recommendation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RECOMMENDATION '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
