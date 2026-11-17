# Full validation and all-scene exports — 2026-11-13

Replaces the cancelled entry-node SS50/PNG validation waiters. Both waiters and the waiting comparison process were terminated before inference began. The unused SS50 configs remain as provenance, not completed evaluations.

User authorized full validation of the two completed continuations on **develbooster**, maximum **2 hours/job**, with all-scene predictions packaged into the same jobs. Scope: **validation split**, **49 scenes**, **9,481 temporal frame pairs**, **1,906 annotated keyframes**. No training-scene exports or video rendering in this allocation.

## Job design

Two jobs, one per model; each one node / four GPUs / four independent workers / 72 CPUs per worker, booster partition with develbooster reservation, two-hour limit. Worker 0 performs full annotated-keyframe validation, saving **1,906 PNGs**, metrics and a lossless prediction archive. Workers 1–3 shard the 49 validation scenes (17/16/16), exporting point and dense-field HDF5 predictions for all temporal frames, compatible with the existing comparison/overlay renderer. Existing complete exports are reused only when provenance matches; partial scene exports resume.

Models: DiT **2867524 / 58,057**, compact DINO **2867528 / 20,595**. No training occurs. Both use their final saved checkpoints.

Runtime estimate: three scene GPUs each process approximately 3,160 frames plus per-scene startup; validation GPU processes 1,906 samples plus metrics/PNGs/archive writes. Prior short evaluations suggest this fits two hours, but this is not a measured full-job guarantee. HDF5 exports and full-validation records support continuation after an interrupted job; no automatic extra jobs are submitted. Full validation and scene export use their established respective seed indexing, so their sampled point clouds need not be identical for a shared frame.

Code: `scripts/run_ablation_full_validation_scenes.py`. Per-model JSON specs, full-validation configs, Slurm launchers and exact submission/output paths are recorded below when prepared/submitted.

Full-comparison references: final DINO **2773475 / 19,914**, best-F1 DINO **2773475 / 15,000**, and RadarGen, under `validation/FULL_validation_20261108/`. Comparisons must match sample identities/order/seeds, use the full 1,906-sample references, and remain separate per ablation.


## Submissions

| Model | Training checkpoint | Bundle SLURM job |
|---|---|---:|
| DiT without bottleneck | 2867524 / 58,057 | **2887635** |
| Compact DINO | 2867528 / 20,595 | **2887636** |

Both submitted successfully to **booster / develbooster**, one node, four GPUs, two hours each; two jobs total. Four orchestration tests passed, both configs/checkpoints verified, launcher syntax and diff checks passed. Scene locks guard shared GT export writers; compact-DINO processes scenes in reverse order to reduce contention.

### DiT outputs

- Config/spec/launcher: `evaluation/configs/FULL_ablation_2867524_step58057_png.yaml`, `evaluation/configs/FULL_ablation_2867524_bundle.json`, `jsc_jupiter/FULL_ablation_2867524_scenes.sbatch`.
- Metrics, full prediction archive and PNGs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/metrics/` (`summary.json`, `metrics.jsonl`, `predictions.h5`, `plots/<scene>/sample_<frame>.png`).
- Worker logs/statuses and archived submission: sibling `execution/`, files `worker_0.out`/`.err` through `worker_3.out`/`.err`, `worker_<rank>_status.json`.
- Separate comparison: `execution/comparison.md`, copied to `docs/ablations/FULL_dit_2867524_step58057_comparison.md` after validation completion and sample-identity checks.
- Scene exports: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<scene>/predictions/dit_2867524_step58057.h5` and `_dense.h5`.
- Scheduler logs: `jsc_jupiter/logs/2887635_FULL_ablation_2867524_scenes.out` and `.err`.

### Compact-DINO outputs

- Config/spec/launcher: `evaluation/configs/FULL_ablation_2867528_step20595_png.yaml`, `evaluation/configs/FULL_ablation_2867528_bundle.json`, `jsc_jupiter/FULL_ablation_2867528_scenes.sbatch`.
- Metrics, full prediction archive and PNGs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_neck_compact_2867528_step20595/metrics/`, same file layout.
- Worker logs/statuses and archived submission: sibling `execution/`, same worker file layout.
- Separate comparison: `execution/comparison.md`, copied to `docs/ablations/FULL_dino_compact_2867528_step20595_comparison.md`.
- Scene exports: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<scene>/predictions/dino_compact_2867528_step20595.h5` and `_dense.h5`.
- Scheduler logs: `jsc_jupiter/logs/2887636_FULL_ablation_2867528_scenes.out` and `.err`.

A successful complete bundle requires all four worker statuses `completed: true`, full validation complete at 1,906 samples/PNGs, and all 49 scene point/dense exports complete. Submitted is not completed.


## GPU-binding correction and random-DINO addition — 2026-11-13

Random-DINO second training allocation **2863217** completed successfully (11h50m08s), final checkpoint `epoch_91_step_29915.pth`. User requested the same full-validation/all-scene bundle.

During the preflight check, DiT bundle **2887635** had failed with OOM: all four tasks were exposed to all GPUs and defaulted to physical GPU 0 despite the srun binding flags. An overlapping read-only check on compact-DINO job **2887636** confirmed `CUDA_VISIBLE_DEVICES=0,1,2,3` and GPU memory concentrated on GPU 0.

Fix: `bind_worker_gpu` explicitly sets a single allocated physical GPU from `SLURM_STEP_GPUS` and `SLURM_LOCALID`, before any model initialization. Each worker records its visible GPU in status JSON. **8 orchestration/binding tests passed.**

Cluster requeue was attempted and returned `Requested operation is presently disabled` for both existing jobs. The active compact-DINO bundle **2887636** was then cancelled for replacement; DiT **2887635** was already failed. Worker logs/statuses were archived under each bundle's `execution/attempt_before_gpu_binding_fix/`. Approximately 916 DiT / 945 compact-DINO validation records were saved at inspection, plus completed scene exports. Replacement workers resume validation and reuse/resume matching scene exports; partial results are not discarded. No concurrent writers to the same validation output remain.

| Model | Current bundle job | Replaces |
|---|---:|---:|
| Random DINO, step 29,915 | **2889167** | New bundle |
| DiT, step 58,057 | **2889168** | 2887635 |
| Compact DINO, step 20,595 | **2889169** | 2887636 |

All are develbooster, one node / four GPUs / two hours. This is three current jobs, within the user's four-job limit; no additional automatic submissions.

Random-DINO config/spec/launcher: `evaluation/configs/FULL_ablation_2863217_step29915_png.yaml`, `evaluation/configs/FULL_ablation_2863217_bundle.json`, `jsc_jupiter/FULL_ablation_2863217_scenes.sbatch`.

Random-DINO output root: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_random_init_2863217_step29915/`: `metrics/` contains summary, records, all 1,906 PNGs and `predictions.h5`; `execution/` contains four worker logs/statuses and archived submission. Separate comparison: `execution/comparison.md`, copied to `docs/ablations/FULL_dino_random_2863217_step29915_comparison.md`. Scene predictions: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<scene>/predictions/dino_random_2863217_step29915.h5` and `_dense.h5`, for all 49 validation scenes / 9,481 frames.

Scheduler logs use `jsc_jupiter/logs/<BUNDLE_JOB>_FULL_ablation_<TRAINING_JOB>_scenes.out` and `.err`. DiT/compact output roots and scientific configs remain unchanged from the original submissions above. Confirm all four worker statuses and archive completeness before reporting success.


## Completion verified

Bundles **2889167**, **2889168** and **2889169** all completed with exit code 0 (1h08m41s, 30m36s, 33m28s respectively). All four worker statuses are complete and GPU bindings are distinct (0/1/2/3). Each full validation has 1,906 records/PNGs, a complete 1,906-frame prediction archive and a separate comparison report. All 49 scene point exports and all 49 dense exports per model have `complete=true`, totaling 9,481 frames each. Verified from saved artifacts.
