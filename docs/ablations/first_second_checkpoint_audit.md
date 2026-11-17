# First-versus-second final-checkpoint audit and VoxelNeXt plan

Requested 2026-11-14. Compare the final first-allocation checkpoint with the final second-allocation checkpoint on the fixed SS50 set, flag earlier checkpoints that performed better, and only then select second-run checkpoints for downstream detection if no concerning regression is found.

## Checkpoints

| Model | First final | Second final |
|---|---|---|
| Random DINO | 2847666 / 15,000 | 2863217 / 29,915 |
| DiT without bottleneck | 2849053 / 28,537 | 2867524 / 58,057 |
| Compact DINO | 2849062 / 10,372 | 2867528 / 20,595 |
| ResNet-50 | 2856174 / 12,575 | 2882612 still training at audit start; no final checkpoint yet |

All four first-final SS50 runs already completed successfully, with common reference selection/seed 42. Reuse them rather than rerunning identical inference. Three completed second-final checkpoints are being evaluated serially on the entry GPU, which was free at launch. Metrics only; first-vs-second comparison uses the same 50 scene/frame/seed tuples, not the full-validation means or a subset of a differently seeded full run.

Configs: `evaluation/configs/PPP_validation_ss50_<SECOND_JOB>_step<STEP>_checkpoint_audit.yaml`. Launcher/log/exit statuses: `output/ablations/first_second_checkpoint_audit/run_entry.sh`, `launcher.log`, `<SECOND_JOB>.log`, `<SECOND_JOB>.exit_code`. Result directories end `_checkpoint_audit` below scratch `validation/`. ResNet's second-final comparison remains pending completion of training.

## Established VoxelNeXt procedure

Local implementation: `radar_detection/radar_eval/`, documentation `radar_detection/RADAR_EVALUATION.md`. The local detector checkout uses a separate `.venv_jsc`, with compiled spconv/CUDA operations; do not modify the generator environment.

- Freeze the supplied **epoch-400 VoxelNeXt** checkpoint from `radar_detection/private_assets/tomer/voxelnext_truckscenes_gt/`; strict graph/state loading. Recorded hash: `277c37e7cc9b05c96bde73a30c456e1f8c9a9b8dc8a694096cd5eda980cea0ed`.
- Evaluate existing complete **1,906-frame `predictions.h5` archives**, with `side=generated`. No generator resampling/inference and no detector fine-tuning. Keep physical [x,y,RCS,Doppler] values from the archived clouds.
- Match the latest prior corrected runs: **z=1 in the flat-up frame before conversion** (`DETECTION_HEIGHT=1`, `DETECTION_HEIGHT_FRAME=flat-up`), then transform through per-frame ego/calibration metadata into `RADAR_LEFT_SIDE`. Preserve RCS; use absolute Doppler. Do not silently revert to the older default `height-frame=detector`.
- Original five detector input features [x,y,z,RCS,abs(Doppler)], MeanVFE, voxel size **0.075×0.075×0.2 m**, max **10 points/voxel**, deterministic sample-token-based point shuffle (seed 42) before voxelization.
- Original three-class detection head and labels **car, truck, trailer**; scorer verifies sample coverage and transforms boxes/velocities to global coordinates.
- Official TruckScenes matching/AP/TP functions with the supplied three-class scoring config: class range 50 m, center-distance thresholds **0.5,1,2,4 m**, TP threshold 2 m, min recall/precision 0.1, max 500 boxes/sample, mean AP weight 5. Report mAP, class AP, NDS and TP errors; do not substitute the standard 12-class aggregate.
- Official illuminated validation split, each scene's last sample excluded: **1,906 samples across 49 scenes**. Use the same camera-frustum/range-filtered source data as the existing archives.
- Existing implementation splits into four contiguous shards (starts **0,477,954,1431**; max 477), merges strictly and scores the complete set. Explicit one-GPU-per-worker binding must also be applied here, given the bundle binding issue already observed on this cluster.

Latest corrected reference outputs: `detection/2838586_radargen_nameorder_z1_flatup/` and `detection/2838595_dino_2773475_final_z1_flatup/` under scratch. Older detector outputs include other height conventions; compare only matched protocols. The best-F1 DINO step-15k detector output needs its archived settings checked before inclusion.

Scientific caveat retained from previous evaluation: `preprocessing_verified=false`. Exact equivalence with the author's private PCD generation/height policy and original dataset version has not been established. This is a matched frozen-detector comparison of our archives, not an established reproduction of the author's published score.

**Checkpoint selection/detector submission is conditional on the SS50 audit. No downstream job has been submitted by this preparation.**


## Audit completed for the three finished second allocations

All three new SS50 evaluations exited 0, 50/50 samples. Reused first-final runs match identical scene/frame/seed tuples. Complete per-metric tables and exploratory paired bootstrap intervals: `SS50_first_vs_second_final_checkpoints.md`; machine-readable results: `output/ablations/first_second_checkpoint_audit/comparison.json`.

- **DiT:** F1 0.268136 → 0.261576; XY Chamfer 1.673151 → 1.715287; count MAE 200.94 → 219.84; RCS MAE 4.922243 → 5.036732. Earlier checkpoint is better on all four.
- **Compact DINO:** F1 0.271095 → 0.282727; XY Chamfer 1.674162 → 1.645975. Count MAE rises modestly 155.20 → 161.34 and RCS MAE 4.572303 → 4.599435; their paired intervals overlap zero. Proceed with second-final for the requested matched downstream evaluation.
- **Random DINO:** F1 0.266862 → 0.267042 (essentially unchanged); RCS MAE 4.863968 → 5.235161, an appreciable regression with a paired interval above zero. Checkpoint choice pending user clarification.
- **ResNet:** first-final F1 0.302766; second allocation still training at the last status check. No second-final checkpoint can be selected yet.

Paired frame-bootstrap intervals are exploratory (50 frames, seed 42, 10,000 resamples, no multiple-comparison correction); they are not a substitute for repeated training seeds or scene-cluster uncertainty. Separate checkpoint-choice questions were sent for DiT and Random DINO; do not silently choose their second checkpoints under the user's no-anomalies condition.

Prepared downstream launcher: `jsc_jupiter/VoxelNeXt_ablation_archive.sbatch`, develbooster, one node / four GPUs / 30 minutes, four explicit GPU-bound shards followed by merge and scoring, height 1 in flat-up coordinates. Uses the existing frozen detector environment. Checkpoint hash verified against the latest corrected reference runs. Seven existing adapter/scoring tests passed.


## Compact-DINO downstream submission

After the checkpoint review, submitted **2895299** for compact DINO **2867528 / 20,595**, using the completed full-validation archive `validation/FULL_ablation_bundles/dino_neck_compact_2867528_step20595/metrics/predictions.h5` under scratch.

Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/2895299_dino_compact_2867528_step20595_z1_flatup/`: four shard HDF5 files, merged `detections.h5`, scored `metrics.json`, archived launcher and `input_archive.txt`. Scheduler logs: `jsc_jupiter/logs/2895299_VoxelNeXt_ablation.out` and `.err`.

DiT and Random-DINO submissions remain pending explicit checkpoint choices after their observed regressions. ResNet's final-second audit remains pending completion of its training allocation.
