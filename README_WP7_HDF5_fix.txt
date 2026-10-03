WP7 HDF5 ground-truth fix

Apply AFTER the original WP7 patch, from the repository root:
  git apply --check wp7_hdf5_gt.patch && git apply wp7_hdf5_gt.patch
  python tests/test_ppp_ground_truth.py
  python tests/test_ppp_metrics.py
  sbatch jsc_jupiter/wp7_evaluation_smoke.sbatch

Uses physical xy/RCS/Doppler from validated HDF5, retaining all detections.
Reuses strict dataset metadata, pair identity and immutable-file checks.
All models in a comparison use the same HDF5 GT when a PPP provider exists;
baseline-only evaluation retains raw loading. GT is never fed to prediction.

Evidence: creator.py captures targets after camera-frustum and strict range
filtering, before coordinate normalization and pixel deduplication. Existing
metadata validation enforces sweep count, camera selection and normalization.

Local verification: 1 HDF5 preservation/rejection test and 3 analytical tests
passed; Python syntax passed; incremental patch applies to original WP7 files.
Actual GPU rerun remains pending. Job 2166130 validated one sample then failed
on an unavailable raw sweep. The preprocessing runner stages scene archives
and removes staging afterward. No preprocessing change or raw extraction needed.
Empty sampled clouds still require an explicit metric policy; no resampling.
