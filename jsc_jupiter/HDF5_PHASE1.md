# Phase 1 HDF5 smoke preprocessing

The runner processes only scene `018a60086f5e441fb09f476e55948b72`, with
`nsweeps=1`. It writes indices `0 ... len(samples)-2`, preserving the existing
camera-frequency index and all BEV/radar calculations. Training loading remains
unchanged. No raw images or t1 depth are written.

Output: `<output_root>/<scene_token>/sample_<frame_index:06d>.h5`.
Writes are synchronous. Each private `.partial` file is closed and validated
before atomic rename. Existing outputs and staging scene directories are refused.
After a failure, already completed samples remain; use a new output root for a
retry. Existing staging is never adopted or deleted. Cleanup unlinks permanent
dataset symlinks without following them. Uncatchable termination such as SIGKILL
can leave staging behind; it must be inspected before another run.

## Stored schema

- `conditioning/{appearance,semantics,velocity}`: `(R,R,3)` uint8 RGB maps.
- `conditioning/depth/<camera_name>`: native `(H,W)` floating t0 camera-Z depth,
  exact output dtype, with representation and meter-unit attributes.
- `radargen_targets/{point_density,rcs,doppler}`: `(R,R)` arrays with exact
  preprocessing dtypes (normally float32, float32, float64 respectively).
- `ppp_targets/xy`: `(N,2)` float32 metric RadarGen flat-up coordinates.
- `ppp_targets/{rcs,doppler}`: `(N,)` float32, dBsm and m/s. Doppler is the
  captured column 7; detections retain pixel collisions and bypass rasterization.
- `metadata/{scene_token,frame_index,sample_token}`: UTF-8, int64, UTF-8.
  The sample token is the reference sample-data record's annotated parent token;
  identity is `(scene_token,frame_index)`.
- `metadata/bounding_boxes/{center_global,size_wlh,rotation_wxyz}`:
  `(B,3)`, `(B,3)`, `(B,4)` float64. Geometry remains original/global.
- `metadata/bounding_boxes/{category_name,annotation_token,instance_token,visibility_token}`:
  `(B,)` UTF-8 strings.
- `metadata/bounding_boxes/attribute_tokens`: `(B,)` UTF-8 JSON arrays.
- `metadata/bounding_boxes/raw_annotations_json`: UTF-8 JSON of original records.
- The box group also stores source token/timestamp, previous sample token,
  previous timestamp, and previous original annotations as JSON. It never
  substitutes interpolated boxes for originals.

Other metadata datasets: schema version (`1.0`), split, dataset version,
reference sensor/token/timestamp/keyframe flag, camera/radar names, t0/t1 camera
and t0 radar sample-data tokens/timestamps/parent sample tokens, camera intrinsics,
camera-to-reference, reference-to-flat-up, global-to-flat-up, map resolution,
point limit, conditioning coordinate range, nsweeps, sigma, effective frustum
views, range and PPP filtering policies, normalization/configuration JSON, and
annotation policy. Numeric provenance retains source dtypes; timestamps are
int64 microseconds. Model weight hashes and a training manifest are deferred.

## Validation and launch

CPU-only tests (no model inference):

```bash
source jsc_jupiter/activate_jupiter.sh
python -m unittest jsc_jupiter.test_hdf5_samples -v
```

Default launch once the configured staging scene directory is absent:

```bash
sbatch jsc_jupiter/hdf5_smoke.sbatch
```

The previously staged scene exists at the default location. To preserve it,
use a fresh staging parent for this smoke run:

```bash
sbatch --export=ALL,HDF5_STAGING_ROOT=/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/staging/hdf5_phase1 \
    jsc_jupiter/hdf5_smoke.sbatch
```

The override stages `<fresh_parent>/<scene_token>/` and removes that owned scene
directory afterwards. The launcher uses cached models offline and does not enable
new compilation or precision modes. Submission is manual.

The real smoke run prints all array shapes/dtypes at frames 0, 50 and 100;
compares those frames' three raw radar targets against existing NPY outputs;
reports maximum absolute difference, exact equality and numerical equality
(`rtol=1e-6`, `atol=1e-8`); and rejects dtype/shape or numerical mismatches.
Every sample validates conditioning/depth presence and native shape, identity,
PPP lengths and exact captured-array serialization, and annotation readback.
These real-data comparisons are pending until the smoke job is explicitly run.
