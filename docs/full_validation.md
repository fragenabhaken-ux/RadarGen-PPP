# Full validation and reusable predictions

`jsc_jupiter/FULL_validation.sbatch` evaluates RadarGen and the four-node/final DINO and DiT checkpoints on every annotated validation keyframe. It uses the existing validation metrics and class-wise MMD sampling policy. The current dataset index contains 1,906 annotated keyframes across 49 scenes. Unannotated temporal frames are excluded because foreground metrics require annotations.

Each checkpoint writes its own directory below `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/`:

- `predictions.h5`: losslessly compressed point clouds and dense predicted fields, without dtype conversion.
- `metrics.jsonl`, `foreground_metrics.jsonl`: per-frame and per-box metrics.
- `summary.json`, `summary.txt`, `metrics.txt`: aggregates and human-readable metrics.
- `validation_config.json`: checkpoint/configuration identities, immutable dataset inventory, and selected indices.

`save_plots: false` disables figure creation, PNGs, and TensorBoard images. `save_predictions: true` enables the archive; both options preserve earlier validation defaults when omitted. Ground truth and conditioning images remain in the original dataset, referenced by source paths and frame metadata rather than duplicated.

## Extract a prediction for plotting

```python
import h5py
import json

with h5py.File('/path/to/run/predictions.h5', 'r') as h:
    scenes = h['index/scene'].asstr()[()]
    frames = h['index/frame'][()]
    group_paths = h['index/group'].asstr()[()]
    # Choose a row from the compact index, or address a known scene/frame directly.
    frame = h[group_paths[0]]
    assert frame.attrs['complete']
    points = frame['points'][()]  # columns: x (m), y (m), RCS (dBsm), Doppler (m/s)
    rcs = frame['fields/rcs'][()]
    doppler = frame['fields/doppler'][()]
    density_name = 'cell_mass_grid' if 'cell_mass_grid' in frame['fields'] else 'point_density'
    density = frame[f'fields/{density_name}'][()]
    source_h5 = frame.attrs['source']
    metadata = json.loads(frame.attrs['metadata_json'])
    radius = h.attrs['coordinate_range']
    # imshow: origin='upper', extent=(-radius, radius, -radius, radius)
```

PPP cell mass is expected points per cell; RadarGen point density retains its predictor representation. These have different semantics and should be labeled separately in plots. PPP RCS and Doppler fields are saved after the predictor's physical mark transform, exactly as used for validation.

Archives include `provenance_json`, `normalization_json`, an index mapping evaluation order to source frames, and per-frame source metadata. Arrays use LZF compression, shuffling, and checksums, except empty arrays. Individual frames can be read without loading other scenes or dense maps. One process writes each archive; concurrent plotting should wait until the writer closes the file.

## Resume and comparison report

The array script automatically uses `--resume` for an existing run. Resume checks the original checkpoint/configuration/inventory and every committed prediction. Prediction data is flushed before its metric record is appended; an uncommitted frame is replaced on resume. Archive and summary completion are tracked separately. Resume supports orderly interruptions; storage corruption or an abrupt process kill may require file repair.

Each array task updates `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_metrics_comparison.md` under a file lock. It shows progress until all five runs finish. Then it verifies matching full-set frame identities and inventories and writes four tables: DINO runs against each other, DiT runs against each other, DINO runs alongside RadarGen, and DiT runs alongside RadarGen. Only first place is bold, based on unrounded values. Existing SS50 reports are preserved.

Manual regeneration:

```bash
python scripts/compare_full_validation.py \
  --root /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108 \
  --output /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_metrics_comparison.md
```
