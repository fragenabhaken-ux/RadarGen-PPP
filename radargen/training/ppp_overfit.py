"""Read-only eval-mode diagnostics for an explicitly selected tiny subset."""
import json
from pathlib import Path
import numpy as np


def summarize_fields(fields, point_mask, rcs_target, doppler_target, bounds):
    """Physical diagnostics on a single sample; inputs are detached NumPy arrays."""
    f = np.asarray(fields, dtype=np.float64)
    mask = np.asarray(point_mask, dtype=bool)
    if f.shape != (3, *mask.shape) or not mask.any() or not np.isfinite(f).all():
        raise ValueError('Invalid diagnostic fields or target mask')
    with np.errstate(over='raise', invalid='raise'):
        mass = np.exp(f[0]) / mask.size
        marks = f[1:] ** 3
    if not np.isfinite(mass).all() or not np.isfinite(marks).all():
        raise ValueError('Nonfinite diagnostic predictions')
    count = float(mass.sum())
    result = dict(expected_count=count, active_cell_count=int(mask.sum()),
        count_abs_error_active=abs(count-int(mask.sum())),
        log_intensity_min=float(f[0].min()), log_intensity_max=float(f[0].max()),
        cell_mass_min=float(mass.min()), cell_mass_max=float(mass.max()),
        mass_fraction_at_target_cells=float(mass[mask].sum()/count) if count > 0 else 0.)
    for i, (name, target) in enumerate(zip(('rcs', 'doppler'), (rcs_target, doppler_target))):
        low, high = bounds[name]
        values = marks[i]
        outside = (values < low) | (values > high)
        result.update({f'{name}_mae_at_targets': float(np.abs(values[mask]-target[mask]).mean()),
            f'{name}_raw_min': float(f[i+1].min()), f'{name}_raw_max': float(f[i+1].max()),
            f'{name}_physical_min': float(values.min()), f'{name}_physical_max': float(values.max()),
            f'{name}_outside_fraction_all_cells': float(outside.mean()),
            f'{name}_outside_fraction_target_cells': float(outside[mask].mean()),
            f'{name}_outside_fraction_intensity_weighted': float(mass[outside].sum()/count) if count > 0 else 0.})
    return result, mass, marks


def diagnose_overfit(model, encoder, dataset, text_provider, ppp_loss, config, accelerator, step):
    import h5py
    import torch
    from torch.utils.data import default_collate
    from accelerate.utils import send_to_device
    from scripts.train_ppp import compute_ppp_losses
    from radargen.training.ppp_checkpoint import capture_rng_state, restore_rng_state
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import logging
    log = logging.getLogger('ppp_overfit')
    # Diagnostics must not change the training RNG stream or model modes.
    generator = torch.Generator()
    rng = capture_rng_state(generator)
    modes = [(module, module.training) for module in model.modules()]
    directory = Path(config.work_dir)/'overfit_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    bounds = {key: (config.data.extra[key+'_min'], config.data.extra[key+'_max'])
              for key in ('rcs', 'doppler')}
    rows = []
    try:
        model.eval()
        if encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
            raise ValueError('Condition encoder must remain frozen and in eval mode')
        with torch.no_grad():
            for index in range(len(dataset)):
                sample = dataset[index]
                batch = send_to_device(default_collate([sample]), accelerator.device)
                with accelerator.autocast():
                    conditions = encoder([batch[k] for k in ('bev_color_map','bev_seg_map','bev_velocity_map')])
                    y, text_mask = text_provider(1, accelerator.device)
                    decoded = model(conditions, y, mask=text_mask, data_info=batch['data_info'])
                _, losses = compute_ppp_losses(decoded, batch, ppp_loss, config.train.extra)
                fields = decoded[0,:,0].float().cpu().numpy()
                mask = sample['point_mask'][0].numpy().astype(bool)
                rcs, doppler = (sample[k][0].numpy() for k in ('rcs_target','doppler_target'))
                stats, mass, marks = summarize_fields(fields, mask, rcs, doppler, bounds)
                # Dataset access above validates immutable source and metadata.
                source = dataset.dataset.frame_index[dataset.indices[index]][0]
                with h5py.File(source, 'r') as handle:
                    continuous_count = len(handle['ppp_targets/rcs'])
                identity = sample['data_info']
                row = dict(step=step, scene=identity['scene_token'], frame=int(identity['frame_idx']),
                    continuous_gt_count=continuous_count,
                    count_abs_error_continuous=abs(stats['expected_count']-continuous_count),
                    **stats, **{key: float(value.item()) for key,value in losses.items()})
                rows.append(row)
                np.savez_compressed(directory/f'step_{step:06d}_sample_{index}.npz',
                    fields=fields, point_mask=mask, rcs_target=rcs, doppler_target=doppler)
                fig, axes = plt.subplots(2,3,figsize=(12,7))
                panels = (np.log10(np.maximum(mass, 1e-30)), np.where(mask,rcs,np.nan), np.where(mask,doppler,np.nan),
                          mask, marks[0], marks[1])
                titles = ('log10 predicted cell mass','GT RCS at retained cells','GT Doppler at retained cells',
                          'GT occupied cells','Predicted RCS (physical)','Predicted Doppler (physical)')
                for ax, panel, title in zip(axes.flat, panels, titles):
                    im=ax.imshow(panel, origin='upper'); ax.set_title(title); fig.colorbar(im,ax=ax)
                fig.suptitle(f'Step {step}, frame {row["frame"]}: expected={stats["expected_count"]:.2f}, '
                             f'active={stats["active_cell_count"]}, continuous={continuous_count}')
                fig.tight_layout(); fig.savefig(directory/f'step_{step:06d}_sample_{index}.png',dpi=110); plt.close(fig)
        with (directory/'metrics.jsonl').open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        mean = {key: float(np.mean([row[key] for row in rows])) for key in
                ('loss_total','loss_ppp_raw','loss_rcs_raw','loss_doppler_raw','expected_count',
                 'active_cell_count','count_abs_error_active','rcs_mae_at_targets','doppler_mae_at_targets')}
        log.info('WP8 eval step %d: %s', step, json.dumps(mean, allow_nan=False))
    finally:
        for module, training in modes:
            module.training = training
        restore_rng_state(rng, generator)
