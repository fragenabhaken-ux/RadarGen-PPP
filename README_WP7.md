# WP7 patch

Apply from the RadarGen repository root, after uploading/extracting this package:

```bash
git apply --check wp7.patch
git apply wp7.patch
source jsc_jupiter/activate_jupiter.sh
python tests/test_ppp_metrics.py
sbatch jsc_jupiter/wp7_evaluation_smoke.sbatch
```

The patch changes three existing files (evaluation/evaluator.py,
evaluation/config.py, evaluation/models/__init__.py) and adds the PPP wrapper,
metrics, two configs, preflight, tests and Slurm script. Existing metrics.py,
aggregation.py, box_utils.py and baseline wrapper remain unchanged.

The GPU preflight processes three frames of the same smoke training scene. It
computes analytical metrics against Evaluator._get_gt_pcl(), counts empty outputs,
and exits with code 2 if any occur. This is an explicit policy stop, not a model
execution failure. It never resamples. Send both logs for review.

The full config uses the validation split but still references the six-update
checkpoint. Replace that checkpoint before any quality evaluation. The model
wrapper uses the evaluator split, not the inference YAML's training split.
It loads HDF5 scenes lazily, checks metadata and pair identity, and never uses
binary masks for analytical GT counts. Raw/legacy-map inference is unsupported
and errors explicitly. Baseline comparison is not configured because its local
trained checkpoint and legacy conditioning path have not been supplied.

After checking the preflight, the ordinary sampled-metric integration can be
run ON AN ALLOCATED GPU with:

```bash
python evaluation/scripts/evaluate.py --config evaluation/configs/truckscenes_eval_ppp_smoke.yaml
```

Do not run ordinary evaluation if the same selected-set preflight found empty
clouds; first decide an explicit policy. For the eventual full validation set,
run the preflight with --config evaluation/configs/truckscenes_eval_ppp.yaml
before running ordinary evaluation with that config. Keep the same seed, sample
order and checkpoint. Preflight and ordinary evaluation each make one prediction
per sample in their separate runs; metrics do not trigger a second forward.

Verified locally: three analytical CPU tests, compilation and shell syntax.
Pending: pretrained integration, sampled-metric runtime and baseline runtime.
WP7_status.tex is a complete status paragraph for the roadmap, not a replacement
for the whole document. No jobs, training or commits were performed here.
