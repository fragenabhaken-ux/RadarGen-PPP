# ablation_dino_neck_compact

Status: implemented; 30 focused/regression tests and the nine-update real GPU probe passed; LR-range job **2847479** submitted on 2026-11-11 to `develbooster`.

## Approved architecture

Thin the **inside of each neck block**, retaining twelve blocks and the full 1152-channel residual stream. This is separate from reducing neck depth or reducing the residual feature width.

| Setting | Standard final DINO neck | Compact neck |
| --- | ---: | ---: |
| Blocks | 12 | 12 |
| Residual/input/output channels | 1152 | 1152 |
| Attention heads | 9 | 4 |
| Channels/head | 128 | 128 |
| Total attention channels | 1152 | 512 |
| Parallel MLP hidden channels | 4608 | 128 |
| Combined QKV + MLP projection | 1152→8064 | **1152→1664** (3×512 + 128) |
| Combined attention + MLP output projection | 5760→1152 | **640→1152** (512 + 128) |

The user approved interpreting the final MLP-ratio suggestion as **128 hidden channels**, not ratio one relative to residual width 1152. Config keys are explicit: `neck_head_dim: 128`, `neck_mlp_hidden_dim: 128`, `neck_heads: 4`. Existing `mlp_ratio` remains the legacy residual-width-relative fallback and is ignored when the explicit hidden width is set.

Keep the existing LayerNorm, per-head RMS normalization, 2D RoPE, parallel attention/SELU-MLP branch structure, residual connection, final neck normalization, checkpointing, standard attention/PixelShuffle head and 1024 output. LoRA rank 16 and the fully trainable DINOv3 ViT-S/16 backbone are unchanged. Three 512 BEV modality maps still supply concatenated final-layer features at 1152×32×32.

## Approved initialization

Original local RGB-pretrained DINOv3 weights: `/e/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors`. **Fresh LoRA, compact neck and head; no trained PPP checkpoint**, no optimizer/scheduler/progress transfer. Fully trainable backbone and LoRA. The head keeps its existing design and normal initialization.

Historical standard-neck production runs transferred trained PPP backbone/adapters from `2391378`; those scores are context, not a controlled initialization match. A controlled neck-width comparison must use the same original DINO initialization on both sides.

## Implementation and checks planned

- Add optional per-head attention width and MLP hidden width to the parallel block/neck and pass them through DinoPPP.
- Preserve legacy defaults, parameter keys/shapes and constructor behavior for existing configs/checkpoints.
- Test exact compact projections, output/gradient shapes, checkpointing, invalid widths, state restoration, and unchanged legacy parameter counts.
- Run the short real-data entry-node test (batches 1/4/8; three updates each), BF16/AdamW, with backbone updates verified.
- Extrapolate to exclusive four-GPU Slurm workers below **90 GB decimal per GPU**, excluding unrelated shared-entry occupancy but retaining the model/context allowances and recording both.
- Submit five-update DDP confirmation followed by a fresh **30-minute exponential LR range**, `1e-7→1e-2`, no warmup, on **develbooster**. No production run.

## Files and outputs

- Model config: `configs/ablation_dino_neck_compact.yaml`.
- LR config: `configs/ablation_dino_neck_compact_lr_range.yaml`.
- Launcher: `jsc_jupiter/ablation_dino_neck_compact_lr_range.sbatch`.
- Model code: `radargen/models/dino_ppp.py`; config validation: `scripts/train_ppp.py`.
- Tests: `tests/test_dino_neck_compact.py`.
- Entry artifacts: `output/ablations/ablation_dino_neck_compact/entry_probe_<timestamp>/` plus sibling `.out`/`.err` logs; inventory, exact config, per-update JSONL and batch recommendation inside.
- Compute artifacts: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_dino_neck_compact_lr_range/`; archived config/entry estimate, GPU telemetry, DDP confirmation directories and JSON, calibrated sweep config, raw sweep logs, `lr_range/metrics.jsonl`, `lr_range/lr_loss.png`, `run_status.json`.
- Scheduler logs: `jsc_jupiter/logs/<SLURM_ID>_ablation_dino_neck_compact_lr_range.out` and `.err`.

Measurements, exact directories and job ID will be recorded after verification. Stop after this model and ask before implementing another ablation.

## Implemented interface and compatibility

`ParallelTransformerBlock` and `RoPETransformerNeck` now accept optional keyword-only `head_dim` and `mlp_hidden_dim`. `DinoPPP` exposes these as `neck_head_dim` and `neck_mlp_hidden_dim`, added at the end of its constructor so existing positional arguments retain their meaning. `scripts/train_ppp.py` validates positive integer widths and the per-head divisibility required by 2D RoPE.

When the new fields are absent, attention width remains equal to residual width and MLP width remains `int(residual_width * mlp_ratio)`. Legacy parameter names, shapes, initialization order and computations are retained. Compact checkpoints have the same key names but deliberately different projection shapes; they must be loaded with their own matching config.

## Parameter budget

| Component | Parameters |
| --- | ---: |
| DINOv3 + modality LoRA | 25,125,888 |
| Compact 12-block neck | 31,869,696 |
| Unchanged standard 1024-output head | 23,615,267 |
| **Total / trainable** | **80,610,851** |

Reference total: 239,863,331. The reduction is 159,252,480 parameters (about 66.4% of the predictor). This predicts parameter savings, not an equivalent throughput or activation-memory reduction; the high-resolution head is unchanged and remains computationally significant.

## Validation record

- **5 new compact-neck tests passed**: exact projections/counts; only expected projection shapes change; checkpointed/plain gradients agree; strict checkpoint restoration; legacy defaults and explicit legacy widths give exactly matching outputs; full tiny-DINO gradients/output shape; config validation.
- **10 existing DINO model tests passed**, **9 DINO attention tests passed**, and **6 ablation-tool tests passed**.
- Both compact YAMLs passed configuration-only validation. Python compilation, Slurm shell syntax and `git diff --check` passed.
- The real entry probe additionally verifies the actual pretrained timm DINO model and a backbone weight update; no production/PPP weights are transferred.

## Entry-node probe

Exact directory: `output/ablations/ablation_dino_neck_compact/entry_probe_20261111T225559Z/`. Terminal logs: `output/ablations/ablation_dino_neck_compact/entry_probe_20261111T225559Z.out` and `.err`.

Files: `source_config.yaml`, `gpu_inventory.csv`, `updates.jsonl`, `batch_recommendation.json`. Eight real samples span the training manifest; three forward/backward/AdamW updates at each of batches 1/4/8. All nine updates had finite losses and gradient norms; a backbone parameter changed. Actual total/trainable count: **80,610,851**.

| Batch/GPU | Peak allocated (GB decimal) | Peak reserved (GB decimal) | Slowest of last two updates (s) |
| --- | ---: | ---: | ---: |
| 1 | 4.01 | 4.57 | 1.217 |
| 4 | 12.88 | 13.38 | 1.926 |
| 8 | 24.71 | 25.00 | 3.018 |

Selected initial DDP batch: **24/GPU**, effective **96** (four GPUs, accumulation one). Conservative predicted per-GPU memory: **82.12 GB**, including 10% fit uncertainty and **2.79 GB** for DDP/context; limit 90 GB decimal. This is extrapolated, not a measured batch-24 DDP result.

The probe used `--dedicated-target`: **7.38 GB** occupied outside this process on the shared entry GPU is recorded but excluded from the exclusive Slurm target estimate. All measured model/optimizer allocations remain included. Entry timings are diagnostics on a shared GPU, not production throughput claims.

## LR-range contract

One `booster` node with **develbooster reservation**, four GH200 GPUs; 45-minute allocation allowing startup, five-update DDP confirmation and plotting around a **30-minute training guard**. The confirmation must finish on all ranks; rank-zero allocated/reserved peak must stay under 90 GB, with per-GPU driver telemetry also recorded every second. Memory failures retry a smaller batch; other failures stop for review.

After confirmation, discard its optimizer/model updates and initialize a new seed-42 model from the approved original RGB DINO checkpoint. Neck, LoRA and standard head start fresh. Exponential LR range **1e-7→1e-2**, no warmup; update count calibrated from actual DDP cadence. The wall-time guard or high-LR divergence can stop before the endpoint; actual last LR and exit status are reported rather than claiming a complete range. No production training, checkpoints, predictions or TensorBoard output.

Detailed cluster files: `submitted_config.yaml`, `entry_batch_recommendation.json`, `gpu_inventory.csv`, `gpu_memory.csv`, `ddp_probe_b<N>/config.yaml` and confirmation logs/memory JSONL, `ddp_probe_summary.json`, `lr_calibration.json`, `sweep/config.yaml`, `sweep/gpu_memory.jsonl`, `lr_range.out`, `lr_range.err`, `lr_range/metrics.jsonl`, `lr_range/lr_loss.png`, and `run_status.json`.

## Submission record

- SLURM job **2847479**, `ablation_dino_neck_compact_lr_range`.
- Reservation **develbooster**, partition `booster`, one node / four GH200 GPUs, 45-minute allocation.
- Recommendation supplied: `/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ablations/ablation_dino_neck_compact/entry_probe_20261111T225559Z/batch_recommendation.json`.
- Exact run directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847479_ablation_dino_neck_compact_lr_range/`.
- Scheduler stdout: `jsc_jupiter/logs/2847479_ablation_dino_neck_compact_lr_range.out`.
- Scheduler stderr: `jsc_jupiter/logs/2847479_ablation_dino_neck_compact_lr_range.err`.
- Expected LR plot: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847479_ablation_dino_neck_compact_lr_range/lr_range/lr_loss.png`.
- Submission is not completion: inspect Slurm state, confirmation/calibration JSON and `run_status.json` after execution.

No other model or production training was submitted in this step. Stop and ask the user before implementing another ablation.
# Production LR selection — 2026-11-12

After reviewing LR-range job **2847479**, the user selected **1e-5** for compact-DINO production training. This records the LR decision only; no production run is submitted by this change.

## Production submitted — 2026-11-12

### Final checkpoint and SS50

Job **2849062** completed with exit code 0 after 11h49m19s, final checkpoint `checkpoints/epoch_47_step_10372.pth`. User requested fixed SS50 evaluation on the entry node, serially after DiT. Config: `evaluation/configs/PPP_validation_ss50_2849062_step10372.yaml`; same reference 50 samples and seed 42 as prior comparisons, metrics only.

Results: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/`. Entry launcher: `output/ablations/ss50_dit_compact_final/run_entry.sh`; log: sibling `2849062.log`; exit status: `2849062.exit_code`. Completion must be confirmed from status and summary.

User authorized production job **2849062**, submitted after the separate DiT job. Config `configs/ablation_dino_neck_compact_train_4nodes.yaml`; launcher `jsc_jupiter/ablation_dino_neck_compact_train_4nodes.sbatch`. Configuration-only validation, shell syntax and diff checks passed.

- Four regular booster nodes / 16 GPUs, 12-hour allocation, 11.8-hour training guard. No develbooster reservation.
- Batch **24/GPU**, accumulation one, **384 global**. LR confirmation passed five DDP updates with rank-zero peak 73,190,604,800 bytes (below 90 GB).
- **1e-5**, warm-up **200 optimizer updates**, then constant; automatic LR scaling disabled. Existing AdamW and gradient clip 1.0 retained.
- **15,000-update upper cap**; one-node calibration 4.017 seconds/update implies approximately **10,600 updates** in 11.8 hours before additional multi-node/checkpoint/diagnostic overhead. Actual count is time-limited, not guaranteed.
- Original pretrained DINOv3 weights, fresh LoRA, compact twelve-block neck and standard 1024-output head; backbone trainable. No trained PPP or LR-sweep weights.
- Checkpoints every 1,000 updates, all retained; TensorBoard, memory logging and prediction diagnostics every 1,000 updates. No automatic second allocation; review first.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2849062_ablation_dino_neck_compact_train_4nodes/`, containing archived config/launcher, node list, launch manifest, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/2849062_ablation_dino_neck_compact_train_4nodes.out` and `.err`.



## SS50 completion verified

Entry-node validation exited **0**; summary reports `complete: true`, **50/50** samples and 50 metric records. Scene/frame/seed tuples match the random-DINO SS50 comparison in identical order.

| Metric | Mean |
|---|---:|
| chamfer_distance_all_norm | 0.042742 |
| chamfer_distance_xy | 1.674162 |
| count_abs_error | 155.200000 |
| distance_attr_f1 | 0.271095 |
| distance_attr_precision | 0.274452 |
| distance_attr_recall | 0.272678 |
| doppler_mae_at_detections | 1.782935 |
| doppler_mae_at_target_cells | 1.782935 |
| generated_count | 1087.300000 |
| gt_count | 1095.540000 |
| inference_time_seconds | 0.265570 |
| iou_th1 | 0.284736 |
| pcl_mmd_doppler | 0.162075 |
| pcl_mmd_rcs | 0.309700 |
| pcl_mmd_xy | 0.042495 |
| ppp_expected_total_count | 1086.691565 |
| ppp_total_count_abs_error | 153.775102 |
| rcs_mae_at_detections | 4.572303 |
| rcs_mae_at_target_cells | 4.572303 |
| target_collision_count | 12.560000 |

Full summary and foreground metrics: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json`.


## Second 12-hour allocation — exact resume, job 2867528

User authorized continuation after reviewing loss and SS50. Submitted separately in order: DiT then compact DINO. Launcher: `jsc_jupiter/ablation_dino_neck_compact_resume_4nodes.sbatch`; original production config: `configs/ablation_dino_neck_compact_train_4nodes.yaml`, verified byte-identical to source job's archived config.

- Source **2849062**, checkpoint `checkpoints/epoch_47_step_10372.pth`, global step **10,372**. Full `--resume` restores model, optimizer, scheduler, training position and all 16 rank RNG states.
- Saved optimizer verified: **628** populated parameter states; all groups at **1e-5**. Saved scheduler verified: `last_epoch=165952` and last LR **1e-5**, already beyond warm-up.
- **No repeated warm-up**, no scheduler replacement, no weights-only transfer. Historical 200-update warm-up remains in the unchanged config for exact-resume contract compatibility; restored scheduler continues past it.
- Four regular booster nodes / 16 GPUs, **12-hour allocation**, **11.8-hour training guard**, constant LR **1e-5**, batch **24/GPU**, **384 global**, accumulation one.
- Up to **15,000 additional updates**, hence at most global step **25,372**; time guard may stop earlier. Checkpoints every 1,000 global updates, all retained. No automatic further allocation.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2867528_ablation_dino_neck_compact_resume_4nodes/`; archived config/launcher/manifest, node list, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/2867528_ablation_dino_neck_compact_resume_4nodes.out` and `.err`.
- Verification: checkpoint metadata checks, unchanged-config comparison, launcher shell syntax and diff checks passed before submission.
