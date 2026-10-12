# Evaluation Pipeline

Dataset-agnostic, multi-model evaluation framework for radar point cloud generation.

## Quick Start

```bash
python evaluation/scripts/evaluate.py --config evaluation/configs/truckscenes_eval.yaml
```

## Configuration

All settings live in a YAML config file. See `evaluation/configs/truckscenes_eval.yaml`.

## Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| Chamfer Distance (xy) | per-sample, per-box | Spatial accuracy |
| Chamfer Distance (normalized) | per-sample, per-box | All attributes normalized to [0,1] |
| IoU | per-sample | At 1.0m threshold |
| MMD (xy, rcs, doppler) | per-sample | Distribution similarity |
| Recall / Precision / F1 | per-sample | Optimal one-to-one matching with distance + attribute thresholds |
| Density Similarity | per-box | min(N, M) / max(N, M) ratio between synthetic and GT points |
| Hit / Miss / FP rates | per-box | Box detection statistics |
| Invariant MMD per class | aggregated | Class-level distribution comparison in box-local coords |

## Adding a New Model

1. Implement `ModelWrapper` (see `evaluation/protocols.py`) and register it with `@register_model("my_model")` in a new file under `evaluation/models/`. See `evaluation/models/radargen_wrapper.py` as an example.
2. Import it in `evaluation/models/__init__.py`.
3. Add it to the config under `models`. See `evaluation/configs/truckscenes_eval.yaml` as an example.

## Full HDF5 validation with plots

Use `evaluation/scripts/validate.py` for every illuminated `val` frame pair
(including intermediate camera frames, not just annotated keyframes). It indexes
`preprocessing/full/hdf5_samples` through the existing metadata-validating loader,
checks completion markers and the processing signature, and verifies `metadata/split`
against the official adapter split. Training manifests are not used or changed.
Both models use identical continuous HDF5 ground truth for point-cloud metrics.

```bash
# Default: epoch_4_step_20000.pth from 2289081_PPP_train, with its training architecture.
sbatch jsc_jupiter/PPP_validation.sbatch

# Select a different compatible PPP checkpoint/config if needed.
sbatch jsc_jupiter/PPP_validation.sbatch --checkpoint /absolute/path/ppp.pth

# Supply a local RadarGen checkpoint (.pth or .safetensors).
sbatch jsc_jupiter/RadarGen_validation.sbatch --checkpoint /absolute/path/radargen.safetensors

# Optional bounded check before a full run (writes a separate Slurm output directory).
sbatch jsc_jupiter/PPP_validation.sbatch --max-samples 3
```

The configs are `evaluation/configs/PPP_validation.yaml` and
`evaluation/configs/RadarGen_validation.yaml`. `model_config` must match the chosen
checkpoint architecture (including PE/QK normalization). RadarGen uses
`configs/RadarGen_validation_inference.yaml`, local DC-AE/Gemma assets, 20 diffusion
steps, and the existing deconvolution recovery. PPP uses its trained decoder and
PPP sampling. Neither runs the camera foundation models: conditioning and GT maps
come from the completed HDF5 samples.

Outputs live in
`/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/<job_id>_<job_name>/`:

- `metrics.txt`: tab-separated scene/frame/status/plot and per-sample metrics.
- `metrics.jsonl`: durable per-sample records with checkpoint, source, and seed.
- `summary.txt` and `summary.json`: means, standard deviations, valid counts,
  empty-cloud counts, and an explicit completeness flag.
- `plots/<scene_token>/sample_<frame_index>.png`: one 3 × 3 comparison per pair.
- `validation_config.json`: checkpoint identity, inventory fingerprint, and settings.

The common metrics are spatial/normalized Chamfer, IoU, distance/attribute
precision/recall/F1, MMD for xy/RCS/Doppler, and point counts/count error. PPP also
reports analytical expected count/count error and mark MAE at occupied target cells.
These are frame-level metrics; this runner does not calculate per-box metrics on
interpolated, unannotated frames. RadarGen plots use the stored Gaussian PD and
Voronoi mark targets; PPP plots use occupied cells and their marks.

Empty clouds are retained without resampling. Distances and MMD are `NA`/JSON null
for empty clouds and excluded from those means; IoU/precision/recall/F1 are zero
when only one cloud is empty and one when both are empty. Each metric summary
states its valid sample count. Metadata/model/inference failures abort the job
rather than silently dropping frames; partial reports are marked incomplete.

Each sample has an independent seed derived from `(seed, dataset_index)`. After
a time limit, continue the previous directory with the same checkpoint/settings:

```bash
sbatch jsc_jupiter/PPP_validation.sbatch \
  --output-dir /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/OLD_JOB_ID_PPP_validation \
  --resume
```

Resume verifies the recorded checkpoint/config/inventory and saved plot identities;
completed samples are not generated again. A fresh run requires an empty output
directory. Slurm logs remain in `jsc_jupiter/logs/<job_id>_<job_name>.out/.err`.
