"""Read-only eval-mode diagnostics for an explicitly selected tiny subset."""
import json
from pathlib import Path
import numpy as np


def sample_ppp_thinning(cell_mass, rng):
    """Sample point locations from a piecewise-constant PPP intensity map."""
    lam = np.asarray(cell_mass, dtype=np.float64)
    if lam.ndim != 2 or not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError("cell_mass must be a finite, nonnegative 2D array")

    h, w = lam.shape
    lam_max = float(lam.max())
    if lam_max == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    # Each cell has unit area, so the proposal count is lam_max * H * W.
    n_proposals = int(rng.poisson(lam_max * h * w))
    if n_proposals == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    x = rng.random(n_proposals) * w
    y = rng.random(n_proposals) * h
    xi = np.minimum(x.astype(int), w - 1)
    yi = np.minimum(y.astype(int), h - 1)

    keep = rng.random(n_proposals) < (lam[yi, xi] / lam_max)
    return x[keep], y[keep], xi[keep], yi[keep]

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


def diagnose_overfit(model, encoder, dataset, text_provider, ppp_loss, config, accelerator, step, *, indices=None, lr=None):
    import h5py
    import torch
    from torch.utils.data import default_collate
    from accelerate.utils import send_to_device
    from scripts.train_ppp import compute_ppp_losses
    from radargen.training.ppp_checkpoint import capture_rng_state, restore_rng_state
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import logging
    log = logging.getLogger('ppp_overfit')
    # Diagnostics must not change the training RNG stream or model modes.
    generator = torch.Generator()
    # Distributed prediction snapshots must not initialize other ranks' CUDA devices.
    import random
    if indices is None:
        rng = capture_rng_state(generator)
    else:
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
               torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == 'cuda' else None)
    modes = [(module, module.training) for module in model.modules()]
    directory = Path(config.work_dir)/('overfit_diagnostics' if indices is None else 'prediction_images')
    directory.mkdir(parents=True, exist_ok=True)
    bounds = {key: (config.data.extra[key+'_min'], config.data.extra[key+'_max'])
              for key in ('rcs', 'doppler')}
    rows = []
    try:
        model.eval()
        if encoder is not None and (encoder.training or any(
                p.requires_grad or p.grad is not None for p in encoder.parameters())):
            raise ValueError('Condition encoder must remain frozen and in eval mode')
        with torch.no_grad():
            for index in (range(len(dataset)) if indices is None else indices):
                sample = dataset[index]
                if not sample['point_mask'].any().item():
                    identity = sample['data_info']
                    skipped = dict(step=step, lr=lr, dataset_index=index,
                        scene=identity['scene_token'], frame=int(identity['frame_idx']),
                        reason='empty target mask; visualization skipped')
                    log.warning('PPP visualization skipped: update=%s dataset_index=%s scene=%s frame=%s; empty target mask',
                                step, index, skipped['scene'], skipped['frame'])
                    with (directory/'skipped_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(skipped, allow_nan=False)+'\n')
                    continue
                batch = send_to_device(default_collate([sample]), accelerator.device)
                with accelerator.autocast():
                    conditions = [batch[k] for k in ('bev_color_map','bev_seg_map','bev_velocity_map')]
                    if config.model.model == 'DinoPPP':
                        decoded = model(*conditions)
                    else:
                        conditions = encoder(conditions)
                        y, text_mask = text_provider(1, accelerator.device)
                        decoded = model(conditions, y, mask=text_mask, data_info=batch['data_info'])
                _, losses = compute_ppp_losses(decoded, batch, ppp_loss, config.train.extra)
                fields = decoded[0] if decoded.ndim == 4 else decoded[0, :, 0]
                fields = fields.float().cpu().numpy()
                mask = sample['point_mask'][0].numpy().astype(bool)
                rcs, doppler = (sample[k][0].numpy() for k in ('rcs_target','doppler_target'))
                stats, mass, marks = summarize_fields(fields, mask, rcs, doppler, bounds)
                # Dataset access above validates immutable source and metadata.
                from torch.utils.data import Subset
                base, base_index = dataset, index
                while isinstance(base, Subset):
                    base_index = base.indices[base_index]
                    base = base.dataset
                source = base.frame_index[base_index][0]
                with h5py.File(source, 'r') as handle:
                    continuous_count = len(handle['ppp_targets/rcs'])
                    conditioning_range = float(handle['metadata/conditioning_coordinate_range'][()])
                    conditioning_images = [
                        handle[f'conditioning/{name}'][()]
                        for name in ('appearance', 'semantics', 'velocity')
                    ]
                identity = sample['data_info']
                row = dict(step=step, lr=lr, dataset_index=index, scene=identity['scene_token'], frame=int(identity['frame_idx']),
                    continuous_gt_count=continuous_count,
                    count_abs_error_continuous=abs(stats['expected_count']-continuous_count),
                    **stats, **{key: float(value.item()) for key,value in losses.items()})
                rows.append(row)
                if indices is None:
                    np.savez_compressed(directory/f'step_{step:06d}_sample_{index}.npz',
                        fields=fields, point_mask=mask, rcs_target=rcs, doppler_target=doppler)

                # Ignore empty GT cells; include all predicted heatmap values.
                mark_norms = []
                for target, prediction in zip((rcs, doppler), marks):
                    low = float(min(target[mask].min(), prediction.min()))
                    high = float(max(target[mask].max(), prediction.max()))
                    if low == high:
                        padding = max(abs(low) * 0.01, 1.0e-6)
                        low, high = low - padding, high + padding
                    mark_norms.append(Normalize(vmin=low, vmax=high))
                rcs_norm, doppler_norm = mark_norms

                gamma = 0.2
                h, w = mask.shape
                sample_rng = np.random.default_rng(
                    np.random.SeedSequence([int(step), int(index)])
                )
                sx, sy, sxi, syi = sample_ppp_thinning(mass, sample_rng)
                sampled_rcs = marks[0, syi, sxi]
                sampled_doppler = marks[1, syi, sxi]

                fig, all_axes = plt.subplots(4, 3, figsize=(14, 14))
                display_range = min(50.0, conditioning_range)
                for ax, image, title in zip(
                    all_axes[0], conditioning_images,
                    ('Conditioning appearance', 'Conditioning semantics', 'Conditioning velocity'),
                ):
                    image_h, image_w = image.shape[:2]
                    # Physical bounds describe the full map, centered on the vehicle.
                    crop_fraction = display_range / conditioning_range
                    left = int(np.floor(image_w * (1.0 - crop_fraction) / 2.0))
                    right = int(np.ceil(image_w * (1.0 + crop_fraction) / 2.0))
                    top = int(np.floor(image_h * (1.0 - crop_fraction) / 2.0))
                    bottom = int(np.ceil(image_h * (1.0 + crop_fraction) / 2.0))
                    ax.imshow(
                        image[top:bottom, left:right],
                        origin='upper',
                        extent=(
                            conditioning_range * (2.0 * left / image_w - 1.0),
                            conditioning_range * (2.0 * right / image_w - 1.0),
                            conditioning_range * (1.0 - 2.0 * bottom / image_h),
                            conditioning_range * (1.0 - 2.0 * top / image_h),
                        ),
                        aspect='equal',
                    )
                    ax.set_xlim(-display_range, display_range)
                    ax.set_ylim(-display_range, display_range)
                    ax.set_title(title)
                    ax.set_xlabel('m')
                    ax.set_ylabel('m')
                axes = all_axes[1:]

                # 11: GT occupied cells
                ax = axes[0, 0]
                im = ax.imshow(
                    mask.astype(np.uint8),
                    origin='upper',
                    extent=(0, w, h, 0),
                    cmap='gray',
                    vmin=0,
                    vmax=1,
                    aspect='auto',
                )
                ax.set_title('GT occupied cells')
                fig.colorbar(im, ax=ax)

                # 12 and 13: GT marks at occupied cells
                gt_mark_panels = (
                    (axes[0, 1], rcs, 'GT RCS', 'viridis', rcs_norm),
                    (axes[0, 2], doppler, 'GT Doppler', 'coolwarm', doppler_norm),
                )
                for ax, target, title, cmap, norm in gt_mark_panels:
                    if indices is not None:
                        yy, xx = np.nonzero(mask)
                        im = ax.scatter(
                            xx + 0.5,
                            yy + 0.5,
                            c=target[mask],
                            s=8,
                            cmap=cmap,
                            norm=norm,
                        )
                        ax.set_xlim(0, w)
                        ax.set_ylim(h, 0)
                        ax.set_aspect('auto')
                    else:
                        panel = np.where(mask, target, np.nan)
                        im = ax.imshow(
                            panel,
                            origin='upper',
                            extent=(0, w, h, 0),
                            cmap=cmap,
                            norm=norm,
                            aspect='auto',
                        )
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax)

                # 21, 22, 23: predictions
                prediction_panels = (
                    (axes[1, 0], mass, 'Pred λ mass', 'hot', None),
                    (axes[1, 1], marks[0], 'Pred RCS heatmap', 'viridis', rcs_norm),
                    (axes[1, 2], marks[1], 'Pred Doppler heatmap', 'coolwarm', doppler_norm),
                )
                for ax, panel, title, cmap, norm in prediction_panels:
                    im = ax.imshow(
                        panel,
                        origin='upper',
                        extent=(0, w, h, 0),
                        cmap=cmap,
                        norm=norm,
                        aspect='auto',
                    )
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax)

                # 31: gamma-adjusted intensity
                ax = axes[2, 0]
                im = ax.imshow(
                    mass ** gamma,
                    origin='upper',
                    extent=(0, w, h, 0),
                    cmap='hot',
                    aspect='auto',
                )
                gt_y, gt_x = np.nonzero(mask)
                ax.scatter(
                    gt_x + 0.5,
                    gt_y + 0.5,
                    s=8,
                    color='lightskyblue',
                    label='GT occupied cells',
                )
                ax.legend(loc='upper right')
                ax.set_title(f'Pred λ^γ (γ={gamma})')
                fig.colorbar(im, ax=ax)

                # PPP samples colored by their predicted marks.
                sampled_panels = (
                    (axes[2, 1], sampled_rcs, 'Sampled points, predicted RCS', 'viridis', rcs_norm),
                    (axes[2, 2], sampled_doppler, 'Sampled points, predicted Doppler', 'coolwarm', doppler_norm),
                )
                for ax, sampled_marks, title, cmap, norm in sampled_panels:
                    ax.set_facecolor('white')
                    if len(sx):
                        points = ax.scatter(
                            sx,
                            sy,
                            c=sampled_marks,
                            s=8,
                            cmap=cmap,
                            norm=norm,
                        )
                        fig.colorbar(points, ax=ax, label='Predicted mark')
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            'No points sampled',
                            transform=ax.transAxes,
                            ha='center',
                            va='center',
                            color='black',
                        )
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)
                    ax.set_aspect('auto')
                    ax.set_title(f'{title} (N={len(sx)})')

                lr_label = '' if lr is None else f', LR={lr:.6g}'
                fig.suptitle(
                    f'Step {step}{lr_label}, frame {row["frame"]}: '
                    f'expected={stats["expected_count"]:.2f}, '
                    f'active={stats["active_cell_count"]}, '
                    f'continuous={continuous_count}',
                    y=1.01,
                )
                fig.tight_layout()
                fig.savefig(
                    directory / f'step_{step:06d}_sample_{index}.png',
                    dpi=110,
                    bbox_inches='tight',
                )
                plt.close(fig)

        with (directory/'metrics.jsonl').open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        if not rows:
            return  # All selected samples were empty; finally still restores RNG/modes.
        mean = {key: float(np.mean([row[key] for row in rows])) for key in
                ('loss_total','loss_ppp_raw','loss_rcs_raw','loss_doppler_raw','expected_count',
                 'active_cell_count','count_abs_error_active','rcs_mae_at_targets','doppler_mae_at_targets')}
        log.info('WP8 eval step %d: %s', step, json.dumps(mean, allow_nan=False))
    finally:
        for module, training in modes:
            module.training = training
        if indices is None:
            restore_rng_state(rng, generator)
        else:
            torch.set_rng_state(rng[0])
            np.random.set_state(rng[1])
            random.setstate(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state(rng[3], accelerator.device)
