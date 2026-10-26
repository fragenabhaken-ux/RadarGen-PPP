# Continuous PPP targets

Previous flow: HDF5 xy/rcs/doppler -> read_ppp_sample -> rounded-coordinate
selection -> final-cell selection -> binary/mark grids -> default collator ->
compute_ppp_losses. Both selection stages discard detections. Grid targets are
also consumed by overfit plots/diagnostics, validation and training utilities.

New default flow: read_ppp_sample(..., continuous_targets=True),
RadarGenPPPDataset(..., continuous_targets=True), or
build_radargen_ppp_dataset_from_config(..., continuous_targets=True) returns
`detections` float32 (N,4): metric flat-up x/y, RCS in dBsm, Doppler in m/s.
No coordinates are discretized and no detections are selected in this mode.
`point_limit` describes the BEV extent. Conditioning and data_info are unchanged.
The second reader return value only contains `stored` in continuous mode.

ContinuousPPPLoss accepts fields (B,3,H,W), detections (N,4), frame_indices
int64 (N,), and a shared positive scalar point_limit. It maps coordinates at
loss time, excludes outside points, retains both extent edges, flips raster y,
and gathers every detection including duplicates. The PPP measure retains the
configured reference area. Marks use the existing cubic link and signed L1.
Losses are returned unweighted as ppp/rcs/doppler. Normalization uses valid
in-extent detection counts per frame, then averages over all B frames. Empty
frames have zero observation/mark contribution but retain the PPP integral.

Continuous targets are the default for readers, datasets, and the config builder,
including manifest-loaded datasets. The legacy raster path requires an explicit
`continuous_targets=False`. The training loader now uses `collate_ppp_samples` and routes continuous
targets through `ContinuousPPPLoss`.

Batching: `collate_ppp_samples` stacks conditioning maps/metadata, concatenates
(N,4) detections, and adds int64 `frame_indices`, explicit `batch_size`, and a
shared scalar `point_limit`. Empty frames remain represented in BEV inputs and
batch_size even if no point refers to them. Mixed raster/continuous samples and
inconsistent extents are rejected. Legacy raster batches use default collation.

Training: `compute_ppp_losses` dispatches on continuous versus raster targets,
preserves the supplied PPPLoss reference area and normalization option, and
applies existing weights/log names. Both direct and singleton-field model
outputs are supported. Text conditioning uses the BEV batch size. Distributed
sampling happens before collation; the existing Accelerate loader uses
num_processes=1 because the sampler already shards frames. Do not enable splitting
or dispatching packed detections across ranks as ordinary sample tensors.

Diagnostics and plotting now consume the continuous lists. Ground-truth marks
are scattered at their original positions, with one color per detection. Exact
coincident points can visually overlap; the full list is retained in NPZ artifacts.
Count grids show multiplicities. Mean-mark display grids are used only to seed
existing plotting layouts; visible GT mark panels use every individual mark.
Per-detection MAE uses each target's own RCS/Doppler, never an aggregated target.
Diagnostics log detection_count, collision_count and count_abs_error_detections,
and include empty frames (zero mark contributions). Unique occupied-cell statistics
remain available separately. Artifacts store detections, point_limit, detection_counts
and predicted fields instead of lossy raster targets. The data smoke utility and
inference sample helper also use the packed collator.

Validation compares every GT detection against its pixel prediction for both
models. The new names are rcs_mae_at_detections and doppler_mae_at_detections;
legacy *_mae_at_target_cells names are compatibility aliases with the new
per-detection weighting. Empty GT yields undefined (null) evaluation mark MAE.
Validation provenance version is now 6, preventing resume of old mixed-semantics
reports. Training checkpoint contracts include continuous_detections_v1, so a
strict optimizer/progress resume from an old raster-target run is rejected;
model-weight loading for inference/fine-tuning remains unchanged.

Circular Doppler is not exposed by the new API; the current DINOv3 configuration
uses signed_l1. No model output or inference sampling representation changes.

Verification: tests/test_continuous_ppp_losses.py covers collisions with distinct
marks, mixed empty/nonempty frames, entirely empty targets, both BEV edges,
outside points, and old/new value and gradient equivalence without collisions.

Focused continuous-loss tests passed (3 tests). Focused packed batching and training-loss integration tests also passed (3 tests)
in `tests/test_ppp_collate.py`; a complete training run is deferred.

Consumer verification passed: 13 focused tests across continuous plotting/diagnostics,
existing overfit diagnostics, and validation. This includes PNG rendering, NPZ
list preservation, empty targets, collision MAE, both evaluation models, and
validation report/resume behavior. Full training and distributed GPU runs remain
deferred.


The main training configs now use `cosine_updates`, advanced once per successful
optimizer update (not per microbatch or rank). Both retain 200 warmup updates.
RadarGen (`PPP_train.yaml`) unifies the original/resumed six epochs and the
six fine-tuning epochs into one fresh optimizer run compressed to six epochs,
peaking at 2e-5 and ending at 1e-6. The fitted cosine uses the same normalized
progress over half the original duration; warmup remains 200 updates. Resume continues the original six-epoch timeline; it does
not add six more epochs. The historical target curve is 2e-5 for epochs 0–4,
1e-5 for 4–8, and 1e-6 for 8–12, compressed to boundaries 2, 4, and 6
epochs for the new run. There is no second optimizer reset/warmup.
DINOv3 (`PPP_train_dinov3.yaml`) retains only the initial 65 epochs, peaking at
1e-3 and ending at 1e-4; its separate fine-tuning run is excluded.

The schedules were initially fitted to the historical step trajectories. They
now deliberately decay earlier to spend more time at low LR: both decay after
the 200-update warmup, reach the floor at 75% of training, and hold it for the
final quarter (`decay_end_fraction: 0.75`). RadarGen reaches 1e-6 at epoch 4.5
and holds through epoch 6. DINOv3 reaches 1e-4 at epoch 48.75 and holds through
epoch 65, with its former peak hold removed. Peak LRs are unchanged. This is
no longer the least-squares fit to the historical step schedules.
Strict cosine resume also checks the total planned optimizer-update count.
