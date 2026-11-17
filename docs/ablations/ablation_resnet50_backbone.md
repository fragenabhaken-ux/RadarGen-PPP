# ablation_resnet50_backbone

Status: implemented; six ResNet tests and six ablation-tool tests passed; entry-node GPU probe passed; LR-range job **2846865** submitted on 2026-11-11.

## Scientific change

Replace DINO with a fully trainable **shared ResNet-50** encoder processing the three existing BEV conditioning maps. Preserve the DINO reference's **12-block, 1152-channel RoPE transformer neck** (nine heads) and **standard attention/PixelShuffle head**, predicting **1024×1024** PPP fields.

Backbone output contract: final 2048-channel ResNet features at **stride 16** (32×32 for 512×512 input), projected by a shared 1×1 convolution to **384 channels per modality**, then concatenated to 1152 channels. No classification head and no modality-specific LoRA adapters are needed in the fully trainable ResNet. Neck/head start fresh; no previous PPP checkpoint is transferred. Keep the same BEV input normalization and continuous-target losses as the reference.

## Approved pretrained variant and implementation

The user selected **standard ResNet-50 with matching ImageNet weights**, rather than the local V1c checkpoint. The implementation uses Torchvision's standard 7×7 input stem and `replace_stride_with_dilation=[false, false, true]`: the last stage keeps stride one with dilation instead of downsampling again. Weight tensor shapes remain compatible with the classification checkpoint.

- Source: Torchvision **ResNet50 IMAGENET1K_V2** (standard ResNet-50 / torchvision V1.5 stride placement, not ResNet-50-v1c).
- Official weight URL: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
- Official definition: [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- Local file: `output/pretrained_models/resnet50/resnet50-11ad3fa6.pth`.
- SHA-256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`, verified against the published `11ad3fa6` prefix.
- Hash record: `output/pretrained_models/resnet50/resnet50-11ad3fa6.sha256`.
- Strictly load the entire ImageNet state first, then remove the classifier and global pooling. No implicit weight download happens inside training. Full PPP checkpoint restoration does not require the original ImageNet asset.
- All backbone convolution and BatchNorm affine parameters train; BatchNorm running statistics update normally per worker, across the modality-stacked batch. No SyncBatchNorm conversion or frozen BN is introduced. The backbone is not checkpoint-recomputed, avoiding duplicate running-stat updates; existing neck/head activation checkpointing is unchanged.
- The BEV dataset's existing `[-1,1]` conditioning normalization is retained to match DINO. No ImageNet classification resize/crop or additional normalization is applied.
- Fully shared backbone and shared linear 1×1 projection across the three modalities; no LoRA. Backbone input batch is `3B`, split and channel-concatenated after projection into `(B,1152,32,32)`.

### Parameter budget

| Component | Parameters |
| --- | ---: |
| ResNet-50 encoder, classifier excluded | 23,508,032 |
| Shared 2048→384 projection | 786,432 |
| Unchanged 12×1152 RoPE neck | 191,122,176 |
| Unchanged standard 1024-output head | 23,615,267 |
| **Total / trainable** | **239,031,907** |

The final DINO reference has 239,863,331 parameters. Removing LoRA and adding the CNN projection are part of the backbone substitution; the feature grid, neck and head remain matched. ImageNet pretraining and DINO pretraining differ, so this comparison tests the chosen pretrained backbone systems, not architecture alone under identical pretraining.

## Workflow and outputs

Use the same [programme procedure](README.md): focused tests, a nine-update entry-node GPU probe at batches 1/4/8, conservative extrapolation below 90 GB decimal per GPU on four workers, then a five-update four-GPU confirmation and fresh **30-minute LR-range sweep** on `develbooster`.

- Model config: `configs/ablation_resnet50_backbone.yaml`.
- LR config: `configs/ablation_resnet50_backbone_lr_range.yaml`.
- Launcher: `jsc_jupiter/ablation_resnet50_backbone_lr_range.sbatch`.
- Model implementation: `radargen/models/resnet_ppp.py`.
- Entry artifacts: `output/ablations/ablation_resnet50_backbone/entry_probe_<timestamp>/`, plus sibling `.out`/`.err` files.
- LR artifacts: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_resnet50_backbone_lr_range/`, using the same filenames as the first ablation (`ddp_probe_summary.json`, `lr_calibration.json`, `lr_range/metrics.jsonl`, `lr_range/lr_loss.png`, `run_status.json`).

Only this requested ablation is in scope. `ablation_dino_frozen_lora` remains unimplemented; the user explicitly selected ResNet next.

## Implemented files and integration

- `radargen/models/resnet_ppp.py`: standard ResNet encoder, shared projection, reused neck/head. Classifier removed after strict ImageNet loading.
- `radargen/training/ppp_initialization.py`: local pretrained construction and full-checkpoint restoration without reloading ImageNet weights.
- `scripts/train_ppp.py`: explicit `ResNetPPP` config validation and direct-prediction/training dispatch.
- `radargen/inference/ppp_pipeline.py`: direct ResNet checkpoint inference without conditioning/text encoders.
- `scripts/probe_ablation_batch.py`, `scripts/run_ablation_lr_range.py`: generalized existing ablation tooling to support ResNet as well as DINO; recommendations must match the model name.
- `tests/test_resnet_ppp.py`: actual ResNet tests for stride-16 features and rectangular output, modality ordering, gradients, BatchNorm update count under checkpointing, strict pretrained loading, and full PPP inference restoration without the ImageNet file.

## Entry-node probe results

Exact directory: `output/ablations/ablation_resnet50_backbone/entry_probe_20261111T221106Z/`. Terminal logs: `output/ablations/ablation_resnet50_backbone/entry_probe_20261111T221106Z.out` and `.err`.

Files: `source_config.yaml`, `gpu_inventory.csv`, `updates.jsonl`, `batch_recommendation.json`. The JSON records byte counts and the complete memory fit. Three real forward/backward/AdamW updates were run at each of batches 1, 4 and 8, on eight samples spread across the 83,554-sample training manifest. Model seed 42; BF16; neck/head activation checkpointing; no prediction rendering. All nine updates completed with finite loss/gradient norm; a backbone weight changed.

| Batch/GPU | Peak allocated (GB decimal) | Peak reserved (GB decimal) | Slowest of last two updates (s) |
| --- | ---: | ---: | ---: |
| 1 | 6.67 | 7.99 | 0.505 |
| 4 | 17.57 | 18.45 | 0.857 |
| 8 | 32.13 | 32.68 | 1.377 |

Four-GPU recommendation: **20 samples/GPU**, effective batch **80**, accumulation one. Estimated per-GPU memory including conservative allowances: **88.40 GB**, below the **90 GB decimal** cap. The fit adds 10% uncertainty, **4.06 GB** for DDP/context, and **1.18 GB** already occupied at probe start. This is an extrapolation; the LR allocation first confirms the recommended batch with five real DDP optimizer updates and reduces it on a memory failure. All ranks must finish, rank-zero peak allocated/reserved memory is checked, and per-GPU driver telemetry is logged each second.

## Checks

- **6 ResNet tests passed**, including pretrained convolution values and removal of the classification head, exact unchanged neck/head parameter counts, finite gradients, one BN update per forward despite neck/head checkpointing, and checkpoint inference without external RGB weights.
- **6 existing random-DINO/tooling tests passed** after generalization.
- Both ResNet YAMLs passed configuration-only validation.
- Python compilation, Slurm shell syntax and `git diff --check` passed.

## LR-range execution contract

One node, four GH200 GPUs, `booster` partition with **`develbooster` reservation**; 45-minute allocation for a **30-minute training window** plus DDP confirmation/startup/plotting. A fresh pretrained ResNet and freshly initialized projection/neck/head are constructed after the disposable DDP confirmation. AdamW LR rises exponentially `1e-7 → 1e-2`, without warm-up. Update count is calibrated from DDP timestamps; a 1800-second training guard can stop before the upper endpoint, and numerical divergence may stop the sweep early. No production checkpoint, prediction rendering or TensorBoard output is enabled.

Output files follow the first ablation's layout: archived `submitted_config.yaml` and `entry_batch_recommendation.json`; `gpu_inventory.csv` and `gpu_memory.csv`; confirmation subdirectories `ddp_probe_b<N>/` with config/logs/memory JSONL; `ddp_probe_summary.json`; `lr_calibration.json`; resolved `sweep/config.yaml` and `sweep/gpu_memory.jsonl`; `lr_range.out`/`.err`; `lr_range/metrics.jsonl` and `lr_range/lr_loss.png`; `run_status.json` with exit codes, last logged LR and endpoint status.

## Submission record

- SLURM job **2846865**, `ablation_resnet50_backbone_lr_range`.
- Reservation: **develbooster**; partition: `booster`; one node / four GH200 GPUs; requested allocation 45 minutes.
- Submitted entry estimate: `/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ablations/ablation_resnet50_backbone/entry_probe_20261111T221106Z/batch_recommendation.json`.
- Exact cluster output directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2846865_ablation_resnet50_backbone_lr_range/`.
- Scheduler stdout: `jsc_jupiter/logs/2846865_ablation_resnet50_backbone_lr_range.out`.
- Scheduler stderr: `jsc_jupiter/logs/2846865_ablation_resnet50_backbone_lr_range.err`.
- Expected LR plot: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2846865_ablation_resnet50_backbone_lr_range/lr_range/lr_loss.png`.
- Submitted does not mean completed: inspect Slurm state and the run's confirmation/calibration/status JSON after execution. No production training was submitted.

Implementation stops here pending the user's choice of the next ablation. Frozen DINO, neck-depth and resolution variants have not been implemented as part of this step.

## Production configuration — prepared 2026-11-12, not submitted

User selected **constant LR 1e-4 after 200 warm-up optimizer updates**, following the first random-DINO four-node production protocol. This section configures only the first allocation; no production job has been submitted for ResNet and no second allocation is chained.

- Config: `configs/ablation_resnet50_backbone_train_4nodes.yaml`.
- Launcher: `jsc_jupiter/ablation_resnet50_backbone_train_4nodes.sbatch`.
- Four regular **booster** nodes / sixteen GH200 GPUs; **12-hour allocation**, **11.8-hour training guard**. `develbooster` is reserved for the short LR tests, not this production launcher.
- **Batch 20/GPU**, verified in the LR run; accumulation one; **320 effective batch**. Automatic LR scaling disabled.
- **200 warm-up updates → constant 1e-4**, no decay. Existing sixteen-rank scheduler handling preserves warm-up in optimizer updates.
- **15,000-update upper cap**, including warm-up. The time guard normally ends this model before that cap: measured one-node LR cadence is about **3.316 s/update** (3.313 s training plus 0.003 s data wait), suggesting an optimistic **12,800 updates in 11.8 hours** before additional multi-node/checkpoint/diagnostic overhead. The 15k cap is a safety ceiling, not a claim that it fits 12 hours.
- `num_epochs: 1000` avoids premature termination and allows a later explicitly approved continuation.
- Same standard ResNet-50 ImageNet-1K V2 initialization; fresh projection, twelve-block neck, head and AdamW state. **No trained PPP or LR-sweep weights**. All model parameters train; standard per-worker BatchNorm behavior is unchanged.
- Checkpoints every **1,000 updates**, all retained, plus final checkpoint at update cap or orderly time stop. Full model/optimizer/scheduler/progress and all-rank RNG state are preserved for a later continuation without repeated warm-up.
- TensorBoard loss/LR scalars and prediction diagnostics enabled; log interval 20 updates, prediction interval 1,000 updates and final stop, matching the random-DINO production convention.

### Output layout when explicitly submitted

`/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_resnet50_backbone_train_4nodes/` contains `submitted_config.yaml`, `submitted_launcher.sbatch`, `nodes.txt`, `launch_manifest.json`, `checkpoints/`, `tensorboard/`, `gpu_memory.jsonl` and native `train_log.log` where emitted.

Scheduler stdout/stderr: `jsc_jupiter/logs/<SLURM_ID>_ablation_resnet50_backbone_train_4nodes.out` and `.err`. The stderr contains per-update loss/LR records. Review this first allocation before choosing a checkpoint and retaining or changing LR for any second 12-hour allocation. No automatic second submission or validation job.

### Production submission record — 2026-11-12

User approved submission after reviewing the settings above. SLURM job **2848698**, `ablation_resnet50_backbone_train_4nodes`, submitted successfully; initial state **PENDING (Priority)** on `booster`, four nodes, twelve hours. This supersedes the prepared/not-submitted status above.

- Output directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2848698_ablation_resnet50_backbone_train_4nodes/` (created when the job starts).
- Scheduler stdout: `jsc_jupiter/logs/2848698_ablation_resnet50_backbone_train_4nodes.out`.
- Scheduler stderr: `jsc_jupiter/logs/2848698_ablation_resnet50_backbone_train_4nodes.err`.
- Only the first allocation submitted; review training before any continuation.

### Startup failure and correction — 2026-11-12

Job **2848698** failed after 2m45, before any optimizer update or checkpoint. The initial prediction diagnostic omitted `ResNetPPP` from its direct-model dispatch and attempted to call `encoder=None`, raising `TypeError: 'NoneType' object is not callable`. LR tests disabled previews and therefore did not exercise this path. This was not an OOM failure.

`radargen/training/ppp_overfit.py` now routes ResNet through `model(*conditions)`, like DINO. Regression coverage in `tests/test_continuous_ppp_consumers.py` exercises a ResNet-configured step-zero prediction with no encoder/text provider and TensorBoard output. Training settings and initialization remain unchanged; user authorized fixing and restarting the first allocation.

Validation: **9 tests passed** (`test_continuous_ppp_consumers.py` and `test_resnet_ppp.py`), including rendered startup preview/TensorBoard output; `git diff --check` passed. Replacement job **2856174** submitted using the same production config/launcher, fresh initialization (no prior updates existed).

Replacement output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2856174_ablation_resnet50_backbone_train_4nodes/`. Scheduler logs: `jsc_jupiter/logs/2856174_ablation_resnet50_backbone_train_4nodes.out` and `.err`. Checkpoint/TensorBoard/manifest layout is unchanged. No second allocation chained.


## First allocation completed; SS50 — 2026-11-13

Replacement production job **2856174** completed successfully, exit 0, elapsed **11h49m58s**, final checkpoint `checkpoints/epoch_47_step_12575.pth`. Last logged update 12,560: loss 10.6364, LR 1e-4.

Fixed SS50 validation launched on the entry node from final step **12,575**, same reference sample order and seed 42 as other ablations. Config: `evaluation/configs/PPP_validation_ss50_2856174_step12575.yaml`.

- Results: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/`.
- Launcher/log/exit status: `output/ablations/ablation_resnet50_backbone/ss50_step12575/run_entry.sh`, `validation.log`, `exit_code.txt`.
- No continuation submitted by this validation step; review results/loss first.


### SS50 completed successfully

Exit code **0**, summary `complete: true`, **50/50** samples. Identical scene/frame/seed tuples and order verified against random-DINO SS50.

| Metric | Mean |
|---|---:|
| chamfer_distance_all_norm | 0.037808 |
| chamfer_distance_xy | 1.606232 |
| count_abs_error | 160.040000 |
| distance_attr_f1 | 0.302766 |
| distance_attr_precision | 0.315710 |
| distance_attr_recall | 0.295508 |
| doppler_mae_at_detections | 1.763626 |
| doppler_mae_at_target_cells | 1.763626 |
| generated_count | 1026.140000 |
| gt_count | 1095.540000 |
| inference_time_seconds | 0.247426 |
| iou_th1 | 0.323237 |
| pcl_mmd_doppler | 0.167810 |
| pcl_mmd_rcs | 0.182655 |
| pcl_mmd_xy | 0.044164 |
| ppp_expected_total_count | 1031.866362 |
| ppp_total_count_abs_error | 155.433345 |
| rcs_mae_at_detections | 4.712151 |
| rcs_mae_at_target_cells | 4.712151 |
| target_collision_count | 12.560000 |

Full summary and foreground metrics: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/summary.json`.


## Second allocation prepared — 2026-11-13, not submitted

User requested preparation of exact-resume continuation from **2856174 step 12,575**, checkpoint `epoch_47_step_12575.pth`. Launcher: `jsc_jupiter/ablation_resnet50_backbone_resume_4nodes.sbatch`; unchanged config: `configs/ablation_resnet50_backbone_train_4nodes.yaml`, verified byte-identical to the original submitted config.

- Four regular booster nodes / 16 GPUs, 12-hour allocation, 11.8-hour training guard.
- Batch **20/GPU**, accumulation one, **320 global**; constant LR **1e-4**, no repeated warm-up.
- Full `--resume` restores model, AdamW optimizer, scheduler, progress and all-rank RNG state. Verified **338** populated optimizer parameter states, **16** rank RNG states, scheduler `last_epoch=201200`, last LR `[0.0001]`, progress epoch 47 / batch 261 / global step 12,575.
- Original config retains its historical 200-update warm-up for exact-resume compatibility; restored scheduler is already past warm-up. No scheduler replacement or weights-only transfer.
- Up to **15,000 additional updates**, global ceiling **27,575**; time guard likely ends earlier. First allocation achieved 12,575 updates. Checkpoints every **1,000 global updates**, all retained. No automatic further allocation.
- On submission, output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<JOB_ID>_ablation_resnet50_backbone_resume_4nodes/`, standard archived config/launcher/manifest, node list, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/<JOB_ID>_ablation_resnet50_backbone_resume_4nodes.out` and `.err`.
- Verification metadata: `output/ablations/ablation_resnet50_backbone/resume_preparation/checkpoint_verification.json`. Launcher shell syntax and diff checks passed. **Not submitted.**


### Second allocation submitted — 2026-11-13

User approved submission of the prepared continuation. Job **2882612**, `ablation_resnet50_backbone_resume_4nodes`, resumes **2856174 step 12,575** with full optimizer/scheduler/progress/RNG state, constant **1e-4**, no repeated warm-up. Settings unchanged: four nodes, 12-hour allocation / 11.8-hour training guard, batch 20/GPU (320 global), checkpoints every 1,000 updates, cap 15,000 additional updates.

Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2882612_ablation_resnet50_backbone_resume_4nodes/`; scheduler logs `jsc_jupiter/logs/2882612_ablation_resnet50_backbone_resume_4nodes.out` and `.err`. This supersedes the prepared/not-submitted status above. No automatic further allocation.
