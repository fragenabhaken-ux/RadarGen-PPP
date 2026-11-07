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

Scene rendering now always produces both the comparison layout and a separate
three-panel overlay for each PPP model: intensity, Doppler, and RCS, left to
right. Intensity uses white ground-truth points; mark panels colour ground-truth
points with the same physical scale as the predicted dense map. Dense mark maps
show the model's per-cell predictions, independent of sampled detections.

Run prediction before rendering historical exports to create the new
`predictions/<model>_dense.h5` files (intensity plus physical RCS and Doppler).
Existing point exports remain reusable. Overlay outputs are saved under
`<output-dir>/overlay_<model>/overlay.mp4` (or PNG/tar for those formats).
Colour limits stay fixed across frames. Intensity is expected points per cell,
displayed with the configured gamma. No other model appears in an overlay video.
The legacy RadarGen model retains its comparison column; dense overlays apply
to PPP models, including the final DiT and DINO checkpoints.

For both final four-node checkpoints on all three previously used scenes:

```bash
sbatch jsc_jupiter/scene_video_dit_dino_4nodes.sbatch
```

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

## Four-node DINO versus DINO RCS L2 on all three scenes

Render directly on the entry node from the saved exports; no inference or Slurm
allocation is needed. Activate the repository environment and load FFmpeg first:

```bash
source jsc_jupiter/activate_jupiter.sh
module load FFmpeg/.7.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for scene in bfe5157ddf0a47648a60d3e2737e8868 044c648ac12345f1aedf33c9f91cdc5a 9d1180d52b6a46328063042b8fb325de; do
  scene_dir="/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/$scene"
  python evaluation/scripts/scene_comparison.py render \
    --scene-dir "$scene_dir" \
    --models radargen dino_4nodes_2684031 dino_rcs_l2_2719598 \
    --marks doppler rcs heatmap --formats mp4 --fps 10 \
    --output-dir "$scene_dir/video_radargen_dino_2684031_dino_2719598"
done
```

The prediction configurations for traffic scenes 1–3 are
`evaluation/configs/scene_comparison_traffic{1,2,3}_dino_4nodes_rcs_l2.yaml`.
Columns: ground truth, RadarGen, four-node DINO (2684031, step 9971),
and DINO RCS L2 (2719598, step 9979). Rows: Doppler, RCS, heatmap.
All temporal frames are included, at 10 fps. Matching exports are reused.
The comparison video is saved under each scene directory as
`video_radargen_dino_2684031_dino_2719598/comparison.mp4`;
separate dense overlays are also generated for both DINO models.
