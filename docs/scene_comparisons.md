# Scene comparisons

Prediction and rendering are separate operations. The traffic-candidate-2 config
selects scene `044c648ac12345f1aedf33c9f91cdc5a`: 194 temporal frames at 10 Hz.
This exports qualitative comparisons, not annotated-keyframe benchmark metrics.

## Predict once

Choose model IDs from `evaluation/configs/scene_comparison_traffic2.yaml`:

```bash
sbatch jsc_jupiter/scene_comparison_predict.sbatch --models radargen ppp_dit_finetuned neck_finetuned
```

The launcher requests one hour on the standard four-GPU booster node allocation;
models run sequentially in one process using one GPU. No jobs are submitted by
creating these files. Additional models can be added as entries referencing existing
validation YAMLs, with optional `overrides` for predictor settings/checkpoints.
Paths in YAML are relative to the repository working directory.

Outputs under the configured root:

```text
<scene>/ground_truth.h5
<scene>/predictions/radargen.h5
<scene>/predictions/ppp_dit_finetuned.h5
<scene>/predictions/neck_finetuned.h5
```

Each compressed HDF5 contains all frames. `/frames/000000/points` is Nx4 float32
with columns `x,y,rcs,doppler` (meters, dBsm, m/s). Model frame groups also contain
`heatmap`: expected points per cell for PPP, or RadarGen's native Gaussianized
point-density map. Heatmaps use upper-origin raster coordinates over the recorded
coordinate range. Frame attributes retain source frame number, reference timestamp
and reference sensor token. Root attributes hold format version and provenance,
including model config text, predictor settings, checkpoint path/size/mtime, seed,
and dataset index identity. Both RCS and Doppler are saved regardless of render rows.

Existing complete exports with matching provenance are reused without loading the
model. An interrupted run uses `.partial.h5`; committed frame groups are skipped on
rerun and the file is renamed only after completion. Do not run simultaneous writers
for the same scene/model. A hard kill can damage an open HDF5 file; remove that
partial file and rerun if HDF5 cannot reopen it. Changed checkpoints/configurations
require a new model ID or output root; incompatible exports are never overwritten.
The immutable dataset index is reused and selected samples are checked on loading.

## Render without inference

Activate the repository environment, then run from the repository root:

```bash
python evaluation/scripts/scene_comparison.py render \
  --scene-dir /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/044c648ac12345f1aedf33c9f91cdc5a \
  --models radargen ppp_dit_finetuned \
  --marks doppler \
  --formats tar \
  --output-dir /tmp/traffic2_doppler
```

Select `--marks doppler rcs heatmap` for the full layout. Row order always remains
Doppler, RCS, heatmap. GT is the first column, followed by `--models` in the supplied
order; the GT heatmap cell is empty. Doppler and RCS color scales are fixed across
all columns and frames. Defaults: Doppler +/-30 m/s, RCS -20 to 66 dBsm; override
with `--doppler-limit` or `--rcs-limits`. Heatmap color limits are fixed across time
for each model, independently, since Gaussianized blobs and PPP intensities have
different units. `--gamma` controls only heatmap display, not stored values.

Output formats can be combined:

- `tar`: one `frames.tar` containing numbered PNGs (default; few filesystem names).
- `png`: loose numbered PNGs for external tools.
- `mp4`: pipe PNGs directly to ffmpeg, producing H.264 `comparison.mp4` without
  intermediate frame files. Requires ffmpeg with libx264 on PATH.

The default playback rate is 10 fps; `--fps` overrides it. Every rendered frame is
fixed size and includes the source frame ID and elapsed reference-sensor time.
`render_config.json` records the input export signatures and rendering options.
Use a separate output directory per layout; rendering overwrites output filenames.
To extract archived PNGs later: `tar -xf frames.tar`.
