# WP8: first tiny-subset overfit experiment

This is the first WP8 experiment, not completion of WP8 or a production run.
Use the actual `scripts/train_ppp.py` loop with four fixed validated HDF5 pairs
(indices 0, 32, 64, 96 in the configured single training scene). Start fresh
from local SANA/DC-AE assets, not from the six-update smoke checkpoint.
No downloads or embedding regeneration. No jobs have been submitted here.

Explicit diagnostic choices:
- 200 updates, batch size 1, four microbatches per update; each update sees all
  four selected samples using the existing sampler.
- Same CAME optimizer, unit loss weights, FP32 decoder, bf16 autocast and 0.1
  gradient clipping. Base LR 1e-4 with existing sqrt batch scaling gives
  effective LR 1.25e-5 for effective batch size four.
- Warmup 20 updates instead of 2000. Constant schedule thereafter.
- Condition dropout and model class dropout disabled only in this diagnostic
  configuration. Normal training retains its original defaults.
- Initial and every-25-update eval-mode diagnostics without condition dropout.
  Model modes and Python/NumPy/Torch RNG states are restored afterward.
- Save checkpoints every 100 updates and on the time/update stop. Up to roughly
  75 minutes of loop runtime, inside a 90-minute Slurm allocation. This is a
  safety budget, not a runtime estimate. Inspect the actual last update.

The subset indices are in `data.extra`, and dropout/diagnostic settings are in
`train.extra`, so they are covered by the existing resume contract. Do not resume
an old WP5 checkpoint into this changed training contract. No baseline training
files are changed. Only PPP training gains opt-in subset and diagnostics support.

## Apply and run

Apply `wp8.patch` from the repository root, then:

```bash
python tests/test_ppp_overfit.py
python scripts/train_ppp.py --config_path configs/RadarGen_PPP_WP8_overfit.yaml --validate-only
sbatch jsc_jupiter/wp8_overfit.sbatch
```

CPU NumPy diagnostics tests, Python syntax, YAML checks and shell syntax passed
locally. PyTorch/Accelerate integration and pretrained execution are NOT tested
in this local environment (PyTorch unavailable). Run the two commands before
submitting. Existing WP5/WP7 GPU results are prior evidence, not WP8 validation.

## What to inspect

The unique job output directory contains `overfit_diagnostics/metrics.jsonl`,
per-step/per-sample PNGs and NPZ arrays, `learning_curves.png`, and checkpoints.
Images use array row/column coordinates; physical coordinates retain the
existing WP6 convention. Color scales are automatic, so use numeric records
or NPZ data for exact across-step comparisons. Marks are never clipped.

Read losses at the same fixed eval samples, rather than comparing different
training microbatches. Expected count should move toward **active-cell count**,
the count in the deduplicated training objective. Continuous GT count is also
reported for comparison but is not the direct training target count. Check
that intensity concentrates at target cells and both physical mark MAEs
improve. A positive count change alone does not establish successful overfit.
Out-of-range mark fractions are reported over all cells, target cells and
weighted by predicted intensity. Training logs retain gradient norm and LR.

The summary reports changes; it does not invent an automatic convergence
threshold. If 200 updates fail to overfit, inspect curves before extending the
run or changing hyperparameters. Do not declare model quality from this test.
Next WP8 tasks: resolve outstanding structural checks, loss-scale run, weight
selection, LR range test, and short multi-GPU stability run before full training.

Send the `.out` and `.err` logs first. For visual review, upload the curves and
initial/final PNGs from `overfit_diagnostics/`.
