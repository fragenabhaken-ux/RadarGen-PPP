# Complete first/second-final ablation evaluation programme

Authorized 2026-11-14. User explicitly requested **both checkpoints for every ablation**, superseding the earlier conditional checkpoint-selection gate and pending first/second choices. “Intermediate” means the final first 12-hour allocation; “last” means the final second allocation. No validation-score checkpoint cherry-picking.

## Required matrix

| Architecture | First-final training run / step | Second-final training run / step |
|---|---|---|
| DiT without bottleneck | 2849053 / 28,537 | 2867524 / 58,057 |
| Compact DINO | 2849062 / 10,372 | 2867528 / 20,595 |
| Random DINO | 2847666 / 15,000 | 2863217 / 29,915 |
| ResNet-50 | 2856174 / 12,575 | 2882612 / final resolved only after training completes |

For all eight checkpoints:

- Fixed SS50, with matching scene/frame/seed identities; reuse completed runs. New SS50 evaluations save PNGs and predictions; previous completed metrics remain valid without repeating inference merely for file-layout changes.
- Full **1,906-keyframe** validation, including PNGs, complete metrics and lossless `predictions.h5` archives.
- The established SS50 count sweep: expected mean count targets **100–2000 by 100**, seeds **42/43/44**, fixed dense fields and identical GT; analytical PPP metrics, whole-area/foreground/per-class metrics, RadarGen reference curves and plots. CPU only. Native sweep points retain full-archive RNG provenance, not a claim of identical native samples to SS50 model inference.
- Frozen **VoxelNeXt epoch 400** on the full archived generated clouds: physical features, absolute Doppler, **z=1 in flat-up before coordinate conversion**, car/truck/trailer scoring. Preserve `preprocessing_verified=false` and existing data-policy caveats. No detector fine-tuning. Reuse completed compact-second detector result 2895299.
- Standard **ten scenes**, three-row comparison videos (Doppler, RCS, heatmap), columns **GT · selected checkpoint · DINO step 15,000 · RadarGen**. Eight checkpoints × ten scenes = **80 videos**. Existing 30 second-checkpoint videos reused. New temporal predictions are generated where needed for these ten scenes; earlier completed all-49-scene exports remain available and untouched.

## Four-job / two-hour constraint

Exactly **four GPU jobs**, one per architecture, one node / four GH200 GPUs each, booster partition with **develbooster reservation**, **02:00:00** limit. No automatic additional GPU submissions. Completed stages are skipped.

- GPU worker 0: first-final full validation, VoxelNeXt and any missing SS50.
- GPU worker 1: second-final equivalent; completed results reused.
- GPU workers 2 and 3: missing ten-scene prediction exports for first/second checkpoint respectively.

Detector inference uses the same explicitly isolated worker GPU after generator validation subprocesses have exited; it does not contend with scene workers. Explicit GPU selection uses the tested `bind_worker_gpu` fix because JSC exposed all GPUs despite srun binding in prior bundles.

The ResNet job has `afterok:2882612` and resolves the highest final saved step only after successful training completion; it must not evaluate a currently available intermediate checkpoint. Its first and second full validations run on separate GPUs, keeping two-checkpoint work parallel within the two-hour allocation.

Estimated critical path for a fresh checkpoint is full validation (previous approximately 1h), single-GPU detector evaluation and SS50, while scenes run on separate GPUs. This is an estimate, not a guaranteed two-hour completion. Full validation and scene exports resume; detector attempts are preserved and a new attempt is used after interruption. A timeout/failure is incomplete work, not success. No silent resource-limit extension.

## Entry-node CPU work

One coordinator waits for completed archives/exports and launches CPU work only when its inputs are ready. A single count-sweep process uses 24 one-thread CPU workers. Up to four video renderers each use two pinned CPU cores. `CUDA_VISIBLE_DEVICES` is empty for CPU stages; no inference uses the entry GPU. Existing complete count sweeps and verified MP4s are reused. The coordinator does not submit jobs.

## Configuration, execution and outputs

- Master manifest: `evaluation/configs/all_ablation_checkpoint_evaluations_20261114.json` (all eight entries, exact source and destination paths).
- GPU runner: `scripts/run_ablation_checkpoint_programme.py`.
- CPU coordinator: `scripts/finish_ablation_programme_cpu.py`.
- Per-architecture launchers: `jsc_jupiter/EVAL_all_checkpoints_{dit,compact,random,resnet}.sbatch`.
- Per-model GPU execution/logs/status: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/evaluation_programme_20261114/<model>/`.
- Full-validation archives/PNGs/metrics: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/<checkpoint-bundle>/metrics/`; older completed second bundles reused in place. Exact paths are in the manifest.
- Final resolved checkpoint metadata: each bundle's `resolved_checkpoint.json`. Per-checkpoint execution configs and report: sibling `execution/`.
- Detector outputs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/ablation_checkpoint_programme/<model>_<phase>_<training-job>/attempt_*/`, except reused compact-second 2895299.
- CPU logs/status/resources: `output/ablations/all_checkpoint_evaluations_20261114/cpu/`.
- Count-sweep tables/CSVs/plots: `output/paper_material/count_sweep_evaluation/ss50_<count-slug>_count_sweep/`; distinct first/second slugs in the manifest. Existing second DiT/compact/random directories unchanged.
- Prediction exports: scratch `scene_comparisons/<scene>/predictions/<resolved-model-id>{,_dense}.h5`.
- Comparison videos: scratch `scene_comparisons/<scene>/video_<resolved-model-id>_dino_2773475_step15000_radargen/comparison.mp4`.
- Live consolidated index: [all checkpoint evaluations](all_checkpoint_evaluations_index.md).

## Validation before submission

Thirteen focused tests passed: final-checkpoint resolution, reject noncontinued checkpoint, full/SS50 protocol separation, completed-result reuse, wrong-checkpoint rejection, detector-cache reuse, scene sharding, resume/PNG checks, failure status and explicit GPU binding. Python compilation, four Slurm syntax checks and diff checks passed. The detector adapter/scorer's seven existing tests passed earlier in this session.

GPU submission IDs and initial scheduler state are appended below. A complete programme requires all eight SS50/full/detector evaluations, all eight count sweeps and all 80 verified videos—not just successful job submission.

## Submission record

| Architecture | Evaluation job | Checkpoints | Dependency |
|---|---:|---|---|
| DiT without bottleneck | **2895546** | first and second final | none |
| Compact DINO | **2895547** | first and second final | none |
| Random DINO | **2895548** | first and second final | none |
| ResNet-50 | **2895549** | first and second final | afterok:2882612 (training) |

All four requested jobs submitted; no additional GPU jobs were created. First three were pending priority at submission; ResNet is waiting on its successful training dependency.

CPU entry coordinator launched via `output/ablations/all_checkpoint_evaluations_20261114/run_cpu_entry.sh`; log `cpu/launcher.log`, live `cpu/status.json` below that directory. It also refreshes the consolidated index and regenerates the four-model paired SS50 report at completion. ResNet is resolved dynamically, never from an in-progress checkpoint.

GPU scheduler logs: `jsc_jupiter/logs/<JOB>_EVAL_all_<architecture>.out` and `.err`. Each architecture execution directory archives the submission manifest and launcher under its job ID.
