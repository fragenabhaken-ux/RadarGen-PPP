#!/usr/bin/env python3
"""Direct PPP training; --validate-only imports no model/data runtime."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyrallis

from diffusion.utils.config import SanaConfig


def validate_ppp_config(config: SanaConfig) -> None:
    """Check the roadmap's configuration contract without runtime initialization."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0

    require(config.data.type == "RadarGenPPPDatasetWrapper", "data.type must be RadarGenPPPDatasetWrapper")
    require(bool(config.data.dataset_dir) and bool(config.data.ppp_data_dir), "data.dataset_dir and data.ppp_data_dir are required")
    require(not config.data.load_text_feat and not config.data.load_vae_feat, "PPP requires load_text_feat=false and load_vae_feat=false")
    data = config.data.extra
    require(isinstance(data, dict), "data.extra must be a dictionary")
    require(data.get("dataset_name") == "truckscenes", "data.extra.dataset_name must be truckscenes")
    require(data.get("dataset_version") == "v1.2-trainval", "data.extra.dataset_version must be v1.2-trainval")
    require(data.get("eval_split") in ("train", "val"), "data.extra.eval_split must be train or val")
    scenes = data.get("ppp_scene_tokens")
    require(scenes is None or (isinstance(scenes, list) and bool(scenes) and
            all(isinstance(s, str) and bool(s) for s in scenes) and len(set(scenes)) == len(scenes)),
            "ppp_scene_tokens must be null or a nonempty list of distinct scene tokens")
    signature = data.get("ppp_processing_signature")
    require(isinstance(signature, str) and len(signature) == 64 and
            all(c in "0123456789abcdef" for c in signature), "ppp_processing_signature must be a SHA256 hex string")
    require(isinstance(data.get("camera_views"), list) and bool(data["camera_views"]), "data.extra.camera_views must be a nonempty list")
    for key in ("coordinate_range", "camera_freq"):
        require(positive(data.get(key)), f"data.extra.{key} must be positive and finite")
    for prefix in ("rcs", "doppler"):
        bounds = [data.get(f"{prefix}_{suffix}") for suffix in ("min", "max")]
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds), f"{prefix} bounds must be finite numbers")
        require(bounds[0] < bounds[1], f"{prefix}_min must be less than {prefix}_max")
    require(config.model.model == "RadarGenPPP_600M_P1_D28", "model.model must be RadarGenPPP_600M_P1_D28")
    require(config.data.image_size == config.model.image_size == 512, "PPP data/model image_size must be 512")
    model = config.model.extra
    require(isinstance(model, dict), "model.extra must be a dictionary")
    require(model.get("num_maps") == model.get("num_conditions") == 3, "PPP requires three modalities and three conditions")
    require(model.get("fixed_timestep") == 0.0, "model.extra.fixed_timestep must be 0.0")
    require(config.model.mixed_precision == "bf16", "PPP mixed_precision must be bf16")
    require(config.vae.vae_type == "dc-ae" and config.vae.weight_dtype == "float32", "PPP requires dc-ae with float32 weights")
    require(config.vae.vae_latent_dim == config.vae.vae_downsample_rate == 32, "PPP requires 32 latent channels and downsample rate 32")
    require(bool(config.vae.vae_pretrained), "vae.vae_pretrained is required")
    losses = config.train.extra
    require(isinstance(losses, dict), "train.extra must be a dictionary")
    for key in ("ppp_weight", "rcs_weight", "doppler_weight"):
        require(positive(losses.get(key)), f"train.extra.{key} must be positive and finite")
    require(losses.get("normalize_ppp_by_gt_count") is True, "normalize_ppp_by_gt_count must be true")
    require(losses.get("velocity_loss_mode") == "signed_l1", "velocity_loss_mode must be signed_l1 until a circular period is established")
    for key in ("train_batch_size", "gradient_accumulation_steps", "num_epochs"):
        require(getattr(config.train, key) > 0, f"train.{key} must be positive")
    require(config.train.num_workers >= 0, "train.num_workers must be nonnegative")
    require(not config.train.use_fsdp, "PPP initially supports the DDP configuration only")
    require(bool(config.work_dir), "root-level work_dir is required")
    require(positive(config.train.gradient_clip), "train.gradient_clip must be positive and finite")
    for key in ("log_interval", "save_model_steps", "save_model_epochs"):
        require(getattr(config.train, key) > 0, f"train.{key} must be positive")
    require(config.train.skip_step == 0, "PPP uses checkpoint progress for resume; train.skip_step must be zero")
    require(not config.train.online_metric, "PPP online evaluation belongs to a later work package")


def compute_ppp_losses(decoded, batch, ppp_loss, settings):
    """Raw decoded fields -> WP3 FP32 links/reductions, applied exactly once."""
    from radargen.losses.mark_losses import masked_rcs_l1, masked_doppler_loss
    if decoded.ndim != 5 or decoded.shape[1:3] != (3, 1):
        raise ValueError('Decoded PPP fields must have shape (B,3,1,H,W)')
    mask, rcs, doppler = (batch[k] for k in ('point_mask', 'rcs_target', 'doppler_target'))
    if any(t.shape != decoded[:, 0].shape for t in (mask, rcs, doppler)):
        raise ValueError('PPP targets must match decoded (B,1,H,W) fields')
    raw = dict(ppp=ppp_loss(decoded[:, 0], mask),
               rcs=masked_rcs_l1(decoded[:, 1], rcs, mask),
               doppler=masked_doppler_loss(decoded[:, 2], doppler, mask,
                                           settings['velocity_loss_mode']))
    weighted = {name: value * settings[f'{name}_weight'] for name, value in raw.items()}
    total = sum(weighted.values())
    if not total.isfinite():
        raise ValueError('Nonfinite weighted PPP loss')
    logs = dict(loss_total=total.detach())
    logs.update({f'loss_{name}_raw': value.detach() for name, value in raw.items()})
    logs.update({f'loss_{name}_weighted': value.detach() for name, value in weighted.items()})
    return total, logs


def training_microbatch(model, condition_encoder, batch, text_provider, ppp_loss, config, accelerator):
    """One direct forward. Only conditions enter the frozen encoder."""
    import random
    with accelerator.autocast():
        conditions = condition_encoder([batch[k] for k in
                                        ('bev_color_map', 'bev_seg_map', 'bev_velocity_map')])
        batch_size = batch['point_mask'].shape[0]
        latent_size = config.model.image_size // config.vae.vae_downsample_rate
        if any(c.shape != (batch_size, 32, latent_size, latent_size) for c in conditions):
            raise ValueError('Unexpected scaled condition latent shapes')
        # Baseline independently drops each complete conditioning batch at p=0.1.
        conditions = [c * 0.0 if random.random() < 0.1 else c for c in conditions]
        y, mask = text_provider(batch_size, accelerator.device)
        decoded = model(conditions, y, mask=mask, data_info=batch.get('data_info'))
    return compute_ppp_losses(decoded, batch, ppp_loss, config.train.extra)


def assert_optimizer_membership(model, encoder, optimizer):
    intended = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != intended:
        raise ValueError('PPP optimizer must contain each trainable DiT/decoder parameter exactly once')
    if set(actual) & {id(p) for p in encoder.parameters()}:
        raise ValueError('Frozen conditioning encoder must not be optimized')


def load_local_text_provider(config, device, model_directory):
    """Preserve live baseline Gemma encoding; no cached-null training substitution."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from radargen.training.ppp_initialization import encode_ppp_empty_text
    if config.text_encoder.text_encoder_name != 'gemma-2-2b-it' or config.text_encoder.chi_prompt:
        raise ValueError('PPP currently supports the baseline Gemma empty-prompt branch without chi_prompt')
    if not Path(model_directory).is_dir():
        raise ValueError(f'Local Gemma assets unavailable: {model_directory}; downloads are disabled')
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    tokenizer.padding_side = 'right'
    encoder = AutoModelForCausalLM.from_pretrained(model_directory, local_files_only=True,
                                                torch_dtype=torch.bfloat16).get_decoder().to(device)
    encoder.requires_grad_(False).eval()
    if encoder.config.hidden_size != config.text_encoder.caption_channels:
        raise ValueError('Gemma caption width differs from configured DiT')
    def provide(batch_size, target_device):
        return encode_ppp_empty_text(tokenizer, encoder, batch_size=batch_size,
                                     model_max_length=config.text_encoder.model_max_length,
                                     device=target_device)
    return provide


def run_training(config, options, *, components=None, dataset=None, text_provider=None):
    """Actual PPP loop; injection is reserved for explicitly labeled CPU tests."""
    import copy
    import datetime
    import logging
    import time
    from dataclasses import asdict
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.data_loader import prepare_data_loader
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from diffusion.data.wids.wids import DistributedRangedSampler
    from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
    from diffusion.utils.lr_scheduler import build_lr_scheduler
    from radargen.losses.ppp_loss import PPPLoss
    from radargen.training.ppp_initialization import initialize_ppp_model
    from radargen.training.ppp_checkpoint import (save_ppp_checkpoint, load_ppp_checkpoint,
                                                capture_rng_state, restore_rng_state)
    config = copy.deepcopy(config)
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('ppp_training')
    accelerator = Accelerator(cpu=components is not None, mixed_precision=config.model.mixed_precision,
                              gradient_accumulation_steps=config.train.gradient_accumulation_steps,
                              log_with=None if config.report_to == 'none' else config.report_to,
                              project_dir=str(Path(config.work_dir)/'logs'),
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))])
    if options.smoke_check and accelerator.num_processes != 1:
        raise ValueError('The minimal checkpoint mutation smoke check requires one GPU/process')
    if components is None and accelerator.device.type != 'cuda':
        raise ValueError('Pretrained PPP training requires an allocated GPU; use --validate-only on login nodes')
    if options.max_updates is not None and options.max_updates <= 0:
        raise ValueError('--max-updates must be positive')
    if options.smoke_check and (options.max_updates is None or options.max_updates < 3):
        raise ValueError('--smoke-check requires --max-updates >= 3 (warmup starts at zero LR)')
    if config.train.online_metric:
        raise ValueError('PPP evaluation is a later work package; online_metric must be false')
    set_seed(config.train.seed + accelerator.process_index)
    generator = torch.Generator().manual_seed(config.train.seed + accelerator.process_index)
    resume = options.resume or config.resume_from or config.model.resume_from
    if isinstance(resume, dict):
        raise ValueError('PPP resume expects a local checkpoint path or latest, not baseline resume options')
    if resume is not None and not isinstance(resume, str):
        raise ValueError('PPP resume must be a local path or latest')
    if resume == 'latest':
        resume = str(Path(config.work_dir)/'checkpoints/latest.pth')
    fresh_config = copy.deepcopy(config)
    fresh_config.resume_from = fresh_config.model.resume_from = None
    if components is None:
        components = initialize_ppp_model(fresh_config, null_embed_path=options.null_embed_path,
                                          device=accelerator.device)
    model, encoder = components.model.to(accelerator.device), components.condition_encoder.to(accelerator.device)
    encoder.eval()
    if dataset is None:
        from radargen.training.radargen_ppp_dataset import build_radargen_ppp_dataset_from_config
        dataset = build_radargen_ppp_dataset_from_config(config)
    sampler = DistributedRangedSampler(dataset, num_replicas=accelerator.num_processes,
                                       rank=accelerator.process_index)
    if len(sampler) == 0:
        raise ValueError('Selected PPP inventory cannot supply a batch per rank')
    loader = DataLoader(dataset, batch_size=config.train.train_batch_size, sampler=sampler,
                        num_workers=config.train.num_workers, pin_memory=accelerator.device.type=='cuda',
                        drop_last=False, generator=generator)
    lr_scale_ratio = 1
    if config.train.auto_lr:
        lr_scale_ratio = auto_scale_lr(config.train.train_batch_size * accelerator.num_processes *
                                      config.train.gradient_accumulation_steps,
                                      config.train.optimizer, **config.train.auto_lr)
    optimizer = build_optimizer(model, copy.deepcopy(config.train.optimizer))
    assert_optimizer_membership(model, encoder, optimizer)
    # Preserve baseline warmup scaling and AcceleratedScheduler world-size stepping.
    config.train.lr_schedule_args = copy.deepcopy(config.train.lr_schedule_args)
    if config.train.lr_schedule_args.get('num_warmup_steps'):
        config.train.lr_schedule_args['num_warmup_steps'] *= accelerator.num_processes
    scheduler = build_lr_scheduler(config.train, optimizer, loader, lr_scale_ratio)
    contract = dict(world_size=accelerator.num_processes, dataset_size=len(dataset),
                    batch_size=config.train.train_batch_size, accumulation=config.train.gradient_accumulation_steps,
                    data=asdict(config.data), model=asdict(fresh_config.model), vae=asdict(config.vae),
                    text_encoder=asdict(config.text_encoder), optimizer=config.train.optimizer,
                    lr_schedule=config.train.lr_schedule, lr_args=config.train.lr_schedule_args,
                    loss=config.train.extra, seed=config.train.seed, gradient_clip=config.train.gradient_clip,
                    null_embed_path=str(Path(options.null_embed_path).resolve()),
                    text_model_dir=str(Path(options.text_model_dir).resolve()))
    progress = dict(epoch=0, batch_in_epoch=0, global_step=0)
    resume_rng = None
    if resume:
        _, _, _, resume_rng = load_ppp_checkpoint(resume, model, optimizer, scheduler,
                                                  contract=contract, rank=accelerator.process_index)
        progress = dict(resume_rng['ppp_training']['progress'])
        if (not all(isinstance(progress.get(k), int) and progress[k] >= 0
                    for k in ('epoch', 'batch_in_epoch', 'global_step'))
                or progress['batch_in_epoch'] > len(loader)):
            raise ValueError('Invalid PPP checkpoint training position')
    if text_provider is None:
        text_provider = load_local_text_provider(config, accelerator.device, options.text_model_dir)
    # Baseline preparation order; already rank-partitioned sampler must not be sharded again.
    model = accelerator.prepare(model)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    loader = prepare_data_loader(loader, device=accelerator.device, num_processes=1, process_index=0,
                                 put_on_device=True, rng_types=[])
    if accelerator.is_main_process and config.report_to != 'none':
        accelerator.init_trackers(config.tracker_project_name)
    ppp_loss = PPPLoss(out_img_size=(512,512),
                       normalize_by_gt_count=config.train.extra['normalize_ppp_by_gt_count']).to(accelerator.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    original = None
    if options.smoke_check:
        bare = accelerator.unwrap_model(model)
        original = (next(bare.dit.parameters()).detach().clone(),
                    bare.ppp_decoder.decoder.project_out.op_list[-1].conv.weight.detach().clone())
    started, last_saved = time.monotonic(), None
    updates_this_run = 0

    def save():
        nonlocal last_saved
        accelerator.wait_for_everyone()
        rng = capture_rng_state(generator)
        ranks = [None] * accelerator.num_processes
        if dist.is_initialized():
            dist.all_gather_object(ranks, rng)
        else:
            ranks = [rng]
        if accelerator.is_main_process:
            last_saved = save_ppp_checkpoint(Path(config.work_dir)/'checkpoints',
                accelerator.unwrap_model(model), optimizer, scheduler, progress=dict(progress),
                contract=contract, generator=generator, rank_rng_states=ranks)
            logger.info('PPP checkpoint saved: %s; progress=%s', last_saved, progress)
        accelerator.wait_for_everyone()

    stop = False
    for epoch in range(progress['epoch'], config.train.num_epochs):
        start_batch = progress['batch_in_epoch'] if epoch == progress['epoch'] else 0
        sampler.set_epoch(epoch)
        sampler.set_start(start_batch * config.train.train_batch_size)
        # Creating workers/iterator consumes only the loader generator, not training RNG.
        iterator = iter(loader)
        if resume_rng is not None:
            restore_rng_state(resume_rng, generator)
            resume_rng = None
        batches_this_epoch = 0
        for local_batch, batch in enumerate(iterator, start=start_batch):
            with accelerator.accumulate(model):
                loss, logs = training_microbatch(model, encoder, batch, text_provider, ppp_loss, config, accelerator)
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if not torch.isfinite(grad_norm):
                        raise ValueError('Nonfinite PPP gradient norm')
                optimizer.step()
                scheduler.step()
                # AcceleratedOptimizer suppresses zero_grad until the accumulation boundary.
                optimizer.zero_grad(set_to_none=True)
            batches_this_epoch += 1
            progress.update(epoch=epoch, batch_in_epoch=local_batch+1)
            if accelerator.sync_gradients:
                progress['global_step'] += 1
                updates_this_run += 1
                values = {key: accelerator.reduce(value.float(), reduction='mean').item() for key,value in logs.items()}
                values.update(grad_norm=accelerator.reduce(grad_norm.float(), reduction='mean').item(),
                              lr=scheduler.get_last_lr()[0])
                if progress['global_step'] % config.train.log_interval == 0 or updates_this_run == 1:
                    logger.info('PPP update %d: %s', progress['global_step'], values)
                    if config.report_to != 'none':
                        accelerator.log(values, step=progress['global_step'])
                if options.max_updates is not None and updates_this_run >= options.max_updates:
                    stop = True
                if time.monotonic()-started >= config.train.training_hours*3600:
                    stop = True
                stop = bool(accelerator.reduce(torch.tensor(int(stop), device=accelerator.device),
                                               reduction='sum').item())
                if progress['global_step'] % config.train.save_model_steps == 0 or stop:
                    save()
            del loss, logs
            if stop:
                break
        if stop:
            break
        if not batches_this_epoch and start_batch == 0:
            raise ValueError('PPP rank has no training batches')
        progress.update(epoch=epoch+1, batch_in_epoch=0)
        sampler.set_start(0)
        if (epoch+1) % config.train.save_model_epochs == 0 or epoch+1 == config.train.num_epochs:
            save()
    if options.smoke_check:
        bare = accelerator.unwrap_model(model)
        parameters = (next(bare.dit.parameters()),bare.ppp_decoder.decoder.project_out.op_list[-1].conv.weight)
        for before, after in zip(original, parameters):
            if torch.equal(before,after):
                raise AssertionError('Smoke check: intended DiT/decoder parameter did not update')
        assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
        assert not encoder.training and not encoder.encoder.training
        assert all(p.dtype==torch.float32 for p in bare.ppp_decoder.parameters())
        if not last_saved:
            save()
        saved_parameters = [p.detach().clone() for p in parameters]
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        optimizer_state_count = len(optimizer.state)
        saved_optimizer_parameters = [copy.deepcopy(optimizer.state[p]) for p in parameters]
        with torch.no_grad():
            for p in parameters:
                p.add_(1)
        optimizer.state.clear()
        load_ppp_checkpoint(last_saved, bare, optimizer, scheduler, contract=contract)
        for saved, restored in zip(saved_parameters, parameters):
            torch.testing.assert_close(saved, restored, rtol=0, atol=0)
        assert scheduler.state_dict() == saved_scheduler
        assert len(optimizer.state) == optimizer_state_count > 0
        for saved, parameter in zip(saved_optimizer_parameters, parameters):
            restored = optimizer.state[parameter]
            assert saved.keys() == restored.keys()
            for key, value in saved.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, restored[key], rtol=0, atol=0)
                else:
                    assert value == restored[key]
        logger.info('PPP smoke parameter updates and checkpoint restoration passed')
    accelerator.end_training()
    return dict(progress=progress, checkpoint=last_saved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--max-updates', type=int, help='Stop after this many optimizer updates in this invocation')
    parser.add_argument('--smoke-check', action='store_true', help='Verify updates and checkpoint mutation/restoration (one GPU)')
    parser.add_argument('--resume', help='Local PPP checkpoint or latest; fresh SANA loading is separate')
    parser.add_argument('--null-embed-path', default='output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth')
    parser.add_argument('--text-model-dir', default='/e/project1/nxtaim-1/huber7/pretrained/gemma-2-2b-it')
    options, config_args = parser.parse_known_args()
    try:
        config = pyrallis.parse(config_class=SanaConfig, args=config_args)
        validate_ppp_config(config)
    except (ValueError, TypeError) as error:
        print(f'PPP configuration error: {error}', file=sys.stderr)
        return 2
    if options.validate_only:
        print('PPP configuration valid (configuration-only; HDF5 contents are not checked).')
        return 0
    run_training(config, options)
    return 0


if __name__ == '__main__':
    sys.exit(main())
