"""Validation of every completed HDF5 frame pair with metrics and matched plots."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


def json_values(value):
    """Convert NumPy scalars and undefined metrics to portable strict JSON."""
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_values(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sample_metrics(points, gt, normalization, mmd_loss, device):
    """Common point-cloud metrics; empty-cloud distances/MMD are undefined."""
    from evaluation.aggregation import PER_SAMPLE_METRICS
    from evaluation.metrics import compute_per_sample_metrics

    metrics = dict(gt_count=len(gt), generated_count=len(points),
                   count_abs_error=abs(len(points) - len(gt)))
    if len(points) and len(gt):
        metrics.update(compute_per_sample_metrics(points, gt, normalization, mmd_loss, device))
    else:
        metrics.update(dict.fromkeys(PER_SAMPLE_METRICS))
        score = float(len(points) == len(gt) == 0)
        for key in ('iou_th1', 'distance_attr_recall', 'distance_attr_precision', 'distance_attr_f1'):
            metrics[key] = score
    return json_values(metrics)


def plot_ppp_prediction(sample, fields, points, coordinate_range, title):
    """PPP targets at occupied cells, physical predicted fields, sampled marks."""
    from radargen.visualization import plot_radargen_prediction

    mask = sample['point_mask'][0].numpy().astype(bool)
    gt = tuple(sample[key][0].numpy() for key in ('point_mask', 'rcs_target', 'doppler_target'))
    pred = (fields['cell_mass_grid'], fields['rcs'], fields['doppler'])
    fig = plot_radargen_prediction(gt, pred, points, coordinate_range=coordinate_range, title=title)
    axes = [ax for ax in fig.axes if ax.get_title()]
    for col, name in enumerate(('GT occupied cells', 'GT RCS at occupied cells', 'GT Doppler at occupied cells')):
        axes[col].set_title(name)
        if col:
            axes[col].images[0].set_data(np.ma.masked_where(~mask, gt[col]))
    axes[3].set_title('Predicted PPP cell mass')
    # The first two density colorbars are not RadarGen probability density.
    colorbars = [ax for ax in fig.axes if ax not in axes]
    for ax in colorbars:
        if ax.get_ylabel() == 'PD':
            ax.set_ylabel('Occupied cells' if ax is colorbars[0] else 'Expected points per cell')
    return fig


class PPPPredictor:
    def __init__(self, model_config, config, device):
        from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
        self.seed = config['seed']
        self.pipeline = RadarGenPPPPipeline.from_config(
            model_config, checkpoint_path=config['checkpoint_path'],
            sana_checkpoint_path=config['sana_checkpoint_path'],
            null_embed_path=config['null_embed_path'], text_model_dir=config['text_model_dir'],
            device=device, seed=config['seed'])

    def seed_sample(self, index):
        self.pipeline.generator.manual_seed(int(np.random.SeedSequence([self.seed, index]).generate_state(1)[0]))

    def predict(self, sample, conditions):
        from radargen.ppp.sampling import ppp_cell_masses
        fields = self.pipeline.predict_fields(*(sample[k] for k in
            ('bev_color_map', 'bev_seg_map', 'bev_velocity_map')))
        points = self.pipeline.sample_fields(fields)
        output = dict(cell_mass_grid=ppp_cell_masses(fields['log_intensity']).numpy(),
                      rcs=fields['rcs_raw'].double().cpu().numpy() ** 3,
                      doppler=fields['doppler_raw'].double().cpu().numpy() ** 3)
        return points, output


class RadarGenPredictor:
    def __init__(self, model_config, config, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from radargen.inference.diffusion_pipeline import RadarGenDiffusionPipeline
        from diffusion.data.transforms import get_transform

        # Load local Gemma explicitly instead of relying on a remote model ID.
        tokenizer = AutoTokenizer.from_pretrained(config['text_model_dir'], local_files_only=True)
        tokenizer.padding_side = 'right'
        encoder = AutoModelForCausalLM.from_pretrained(config['text_model_dir'],
            local_files_only=True, torch_dtype=torch.bfloat16).get_decoder().to(device)
        encoder.requires_grad_(False).eval()
        self.pipeline = RadarGenDiffusionPipeline(config=config['model_config'], device=device,
                                                  tokenizer=tokenizer, text_encoder=encoder)
        from radargen.inference.model_download import find_model
        state = find_model(config["checkpoint_path"])
        state = state.get("state_dict", state)
        state.pop("pos_embed", None)
        missing, unexpected = self.pipeline.model.load_state_dict(state, strict=False)
        if set(missing) - {"pos_embed"} or unexpected:
            raise ValueError(f"RadarGen checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
        self.pipeline.model.eval().to(self.pipeline.weight_dtype)
        self.pipeline.eval()
        self.transform = get_transform('default_train_from_np', resolution=model_config.data.image_size)
        self.seed = config['seed']
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        self.size = model_config.data.image_size
        self.radius = model_config.data.extra['coordinate_range']
        self.steps = config['num_inference_steps']
        self.device = device

    def seed_sample(self, index):
        self.generator.manual_seed(int(np.random.SeedSequence([self.seed, index]).generate_state(1)[0]))

    def predict(self, sample, conditions):
        from radargen.inference.pcl_recovery import recover_pcl_attributes
        maps = [self.transform(image).to(self.device).unsqueeze(0) for image in conditions]
        result = self.pipeline(prompt='', height_single_image=self.size, width_single_image=self.size,
            height_num_images=1, width_num_images=1, guidance_scale=0.0, pag_guidance_scale=0.0,
            num_inference_steps=self.steps, generator=self.generator, bev_condition_maps=maps)[0]
        pred = (result.pd_map_np, result.rcs_map_np, result.doppler_map_np)
        xy, rcs, doppler = recover_pcl_attributes(*pred, image_size=self.size,
                                                coordinate_range=self.radius, device=self.device)
        return np.column_stack((xy, rcs, doppler)), pred


def write_reports(directory, rows, provenance):
    """Rewrite human-readable reports from the durable per-sample records."""
    keys = sorted({key for row in rows for key in row['metrics']})
    with (directory / 'metrics.txt').open('w') as stream:
        stream.write('\t'.join(['scene', 'frame', 'status', 'plot', *keys]) + '\n')
        for row in rows:
            values = [row['scene'], str(row['frame']), row['status'], row['plot']]
            values += ['NA' if row['metrics'].get(k) is None else str(row['metrics'][k]) for k in keys]
            stream.write('\t'.join(values) + '\n')
    summary = dict(model=provenance['model'], checkpoint=provenance['checkpoint_path'],
                   split='val', evaluated_samples=len(rows), expected_samples=provenance['sample_count'],
                   complete=len(rows) == provenance['sample_count'],
                   empty_gt_samples=sum(r['metrics']['gt_count'] == 0 for r in rows),
                   empty_generated_samples=sum(r['metrics']['generated_count'] == 0 for r in rows),
                   metrics={})
    for key in keys:
        values = [row['metrics'][key] for row in rows if row['metrics'].get(key) is not None]
        summary['metrics'][key] = dict(mean=float(np.mean(values)) if values else None,
            std=float(np.std(values)) if values else None, valid_samples=len(values))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (directory / 'summary.txt').open('w') as stream:
        for key, value in summary.items():
            if key != 'metrics':
                stream.write(f'{key}: {value}\n')
        stream.write('\nUndefined distances/MMD for empty clouds are NA and excluded from those means.\n')
        for key, stats in summary['metrics'].items():
            stream.write(f"{key}: mean={stats['mean']}, std={stats['std']}, valid_samples={stats['valid_samples']}\n")


def evaluate_samples(dataset, predictor, config, device, directory, provenance, *, resume=False):
    import h5py
    import torch
    from evaluation.metrics import MMDLoss, RBF
    from evaluation.ppp_metrics import compute_ppp_count_metrics, make_ppp_grid_xy
    from radargen.training.radargen_ppp_dataset import validate_ppp_metadata
    from radargen.visualization import plot_radargen_prediction

    mmd = MMDLoss(kernel=RBF(device=device), device=device)
    rows = []
    directory = Path(directory)
    records = directory / 'metrics.jsonl'
    if resume and records.exists():
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        if len(rows) > provenance['sample_count']:
            raise ValueError('More saved records than selected samples')
        for index, row in enumerate(rows):
            path, expected, _, _ = dataset.frame_index[index]
            if (row['index'] != index or row['scene'] != expected['scene_token']
                    or row['frame'] != int(expected['frame_index'])
                    or row['source'] != str(path) or not (directory / row['plot']).is_file()):
                raise ValueError('Saved records or plots do not match the selected inventory')
    write_reports(directory, rows, provenance)
    try:
        with records.open('a' if resume else 'x') as stream, torch.inference_mode():
            for index in range(len(rows), provenance['sample_count']):
                path, expected, size, mtime = dataset.frame_index[index]
                stat = path.stat()
                if path.is_symlink() or (stat.st_size, stat.st_mtime_ns) != (size, mtime):
                    raise ValueError(f'Indexed immutable sample changed: {path}')
                with h5py.File(path, 'r') as handle:
                    validate_ppp_metadata(handle, expected, resolution=dataset.resolution,
                        point_limit=dataset.point_limit, normalization=dataset.normalization)
                    target = handle['ppp_targets']
                    gt = np.column_stack([target[k][()] for k in ('xy', 'rcs', 'doppler')])
                    conditions = [handle[f'conditioning/{k}'][()] for k in ('appearance', 'semantics', 'velocity')]
                    gt_maps = tuple(handle[f'radargen_targets/{k}'][()] for k in ('point_density', 'rcs', 'doppler'))
                if gt.shape[1] != 4 or not np.isfinite(gt).all() or not (np.abs(gt[:, :2]) < dataset.point_limit).all():
                    raise ValueError(f'Invalid continuous targets: {path}')
                # PPP needs the exact training transform/rasterization; RadarGen uses raw conditioning.
                sample = dataset[index] if config['model'] == 'ppp' else None
                if hasattr(predictor, "seed_sample"):
                    predictor.seed_sample(index)
                points, fields = predictor.predict(sample, conditions)
                if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
                    raise ValueError(f'Invalid predicted point cloud: {path}')
                from radargen.core.normalization import NormalizationConfig
                metrics = sample_metrics(points, gt, NormalizationConfig(**dataset.normalization), mmd, device)
                scene, frame = expected['scene_token'], int(expected['frame_index'])
                plot = f'plots/{scene}/sample_{frame:06d}.png'
                title = f"{config['model']} validation: {scene}, frame {frame}; GT={len(gt)}, generated={len(points)}"
                if config['model'] == 'ppp':
                    mass = fields['cell_mass_grid']
                    analytic = compute_ppp_count_metrics(mass, gt, [], make_ppp_grid_xy(*mass.shape, dataset.point_limit))
                    metrics.update({k: v for k, v in analytic.items() if not isinstance(v, list)})
                    mask = sample['point_mask'][0].numpy().astype(bool)
                    for name in ('rcs', 'doppler'):
                        values = sample[name + '_target'][0].numpy()
                        metrics[name + '_mae_at_target_cells'] = float(np.abs(fields[name][mask] - values[mask]).mean()) if mask.any() else None
                    fig = plot_ppp_prediction(sample, fields, points, dataset.point_limit, title)
                else:
                    fig = plot_radargen_prediction(gt_maps, fields, points, coordinate_range=dataset.point_limit, title=title)
                destination = directory / plot
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fig.savefig(destination, dpi=config.get('plot_dpi', 110), bbox_inches='tight')
                finally:
                    fig.clear()
                status = 'ok' if len(gt) and len(points) else 'empty_cloud'
                row = json_values(dict(index=index, scene=scene, frame=frame, source=str(path),
                    model=config['model'], checkpoint=config['checkpoint_path'], seed=config['seed'],
                    status=status, plot=plot, metrics=metrics))
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                if len(rows) % 100 == 0:
                    write_reports(directory, rows, provenance)
                LOG.info('Validated %d/%d: %s frame %d', index + 1, provenance['sample_count'], scene, frame)
    finally:
        write_reports(directory, rows, provenance)
    return rows


def run_validation(config, output_dir=None, max_samples=None, *, resume=False):
    import pyrallis
    import torch
    from diffusion.utils.config import SanaConfig
    from radargen.training.radargen_ppp_dataset import validate_radargen_ppp_dataset_from_config

    if config['model'] not in ('ppp', 'radargen'):
        raise ValueError('model must be ppp or radargen')
    checkpoint = Path(config['checkpoint_path']).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint unavailable: {checkpoint}; pass --checkpoint with a local trained checkpoint')
    config = dict(config, checkpoint_path=str(checkpoint))
    with open(config['dataset_config']) as stream:
        data_config = pyrallis.load(SanaConfig, stream)
    data_config.data.extra['eval_split'] = 'val'
    data_config.data.extra['ppp_scene_tokens'] = None
    # Validate every scene's completion/signature/identity, independent of training manifests.
    dataset = validate_radargen_ppp_dataset_from_config(data_config)
    if not len(dataset):
        raise ValueError('No validation frame pairs available')
    with open(config['model_config']) as stream:
        model_config = pyrallis.load(SanaConfig, stream)
    if model_config.data.image_size != dataset.resolution:
        raise ValueError('Model and validation resolution differ')
    for key, value in dataset.normalization.items():
        if model_config.data.extra.get(key) != value:
            raise ValueError(f'Model and validation normalization differ: {key}')
    limit = max_samples if max_samples is not None else config.get('max_samples')
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError('max_samples must be a positive integer or null')
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    device = torch.device(config.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable for requested validation device')
    directory = Path(output_dir or config['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and any(directory.iterdir()):
        raise ValueError(f'Output directory must be empty to prevent mixing runs: {directory}')
    model_config.work_dir = str(directory)
    model_config.train.work_dir = str(directory)
    provenance = dict(config, split='val', sample_count=count, available_samples=len(dataset),
        config_sha256={key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                       for key in ('dataset_config', 'model_config')},
        checkpoint_size=checkpoint.stat().st_size, checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        inventory_sha256=hashlib.sha256(json.dumps([(str(p), e['scene_token'], e['frame_index'], s, m)
            for p, e, s, m in dataset.frame_index[:count]]).encode()).hexdigest())
    record = directory / 'validation_config.json'
    if resume:
        if not record.is_file() or json.loads(record.read_text()) != provenance:
            raise ValueError('Resume requires the same checkpoint, config, and immutable sample inventory')
    else:
        record.write_text(json.dumps(provenance, indent=2) + '\n')
    torch.manual_seed(config['seed'])
    predictor = (PPPPredictor if config['model'] == 'ppp' else RadarGenPredictor)(model_config, config, device)
    return evaluate_samples(dataset, predictor, config, device, directory, provenance, resume=resume)
