# ablation_dit_dcae_no_bottleneck

Status: implemented; 14 tests and the nine-update real GPU probe passed; LR-range job **2847195** submitted on 2026-11-11 to `develbooster`.

## Intended comparison

Replace the final DiT reference's joint attention/PixelShuffle head with the **original scalar DC-AE decoder adapted to accept full-width transformer features**. Retain the conditioning-only DiT, frozen DC-AE conditioning encoder, empty-prompt conditioning, continuous targets, 512 output, cubic mark links and the established optimizer/loss conventions. Decode all three modality streams separately through the same shared scalar decoder, as in the original DiT PPP model.

- Original output path: `(3B,1152,16,16) → DiT final prediction layer → (3B,32,16,16) → scalar DC-AE decoder → (3B,1,512,512)`.
- Proposed output path: `(3B,1152,16,16) → widened DC-AE input block → (3B,1024,16,16) → existing decoder stages → (3B,1,512,512)`.
- Bypass and remove the DiT final prediction layer so disconnected parameters do not remain in DDP.
- Preserve the pretrained DC-AE output adaptation from three RGB channels to one scalar field.
- Transformer features are not scaled VAE latents: do not apply the old inverse latent scaling factor to them. The frozen conditioning encoder keeps its original latent scaling unchanged.

## Approved input block

The user selected a **fresh 3×3 convolution, 1152→1024**, removing the native fixed channel-duplication shortcut. The old block combined a 32→1024 convolution with channel duplication; that shortcut cannot accept 1152 channels. No replacement shortcut or 1152→32 adapter is introduced. All later decoder stages, upsampling and scalar output remain unchanged.

This is channel reduction within learned spatial decoding, rather than the old forced 32-channel interface. The new convolution retains the native input block's kernel, stride-one and padding-one behavior.

## Approved initialization

The user clarified: **the same starting SANA + RGB DC-AE assets used to initialize run `2438928`**, not the trained checkpoint of that run and not `2516862`.

- SANA: `/e/project1/nxtaim-1/huber7/pretrained/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth`.
- RGB DC-AE: `/e/project1/nxtaim-1/huber7/pretrained/dc-ae-f32c32-sana-1.1/`.
- Empty-prompt validation asset: `output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth`.
- Frozen Gemma used live for empty-prompt conditioning: `/e/project1/nxtaim-1/huber7/pretrained/gemma-2-2b-it`.
- SANA weights load before the standard 32→99 conditioning-input adaptation. Missing Q/K normalization parameters use the existing validated constructor defaults; no unexpected checkpoint mismatch is accepted.
- RGB decoder weights load before averaging its final output convolution to one scalar channel. The new 1152→1024 input convolution is initialized fresh with the native layer's usual initialization.
- No trained PPP weights, optimizer state, scheduler state or progress are loaded. This differs from the historical standard-head run's `2516862` transfer, so architecture and initialization history must not be conflated in comparisons.
- Conditioning encoder stays frozen with its original scaling factor. Full-width output features bypass inverse VAE scaling because they are not VAE latents.

## Parameter budget

| Component | Trainable parameters |
| --- | ---: |
| DiT feature backbone, final prediction layer removed | 591,850,377 |
| Original scalar DC-AE decoder, widened input | 169,542,785 |
| **Prediction network total** | **761,393,162** |

The original scalar decoder had 159,220,865 parameters; widening adds 10,321,920. Frozen condition/text encoders are excluded from this table but remain in the actual GPU memory measurements. Native DC-AE activation checkpointing is retained.

## Implementation and checks

- Add a separate decoder type, leaving the existing `dc_ae` and standard joint head unchanged.
- Strictly verify the expected input block, channel counts and checkpoint-key compatibility; incompatible state must fail rather than silently load partially.
- Reuse the existing 512-output DC-AE stages and per-stream wrapper ordering `(B,3,1,H,W)`.
- Test full feature-channel access, gradients into DiT and the widened projection, compatible-weight preservation, strict restore, and unchanged existing variants.
- Extend the existing ablation entry probe and LR launcher for the latent DiT path: frozen condition encoding, live empty-prompt text conditioning, CAME optimizer, actual forward/backward updates and memory measurement. Do not silently substitute the direct DINO/ResNet AdamW path.
- Use batches 1/2/4 for the very short entry test (three optimizer updates each), then conservatively extrapolate to four GPUs below **90 GB decimal per GPU** and confirm the batch inside the compute allocation.
- Submit a **30-minute LR training window** to `develbooster`, with separate overhead for confirmation/startup/plotting; no production training.

## Planned files and outputs

- Model config: `configs/ablation_dit_dcae_no_bottleneck.yaml`.
- LR config: `configs/ablation_dit_dcae_no_bottleneck_lr_range.yaml`.
- Launcher: `jsc_jupiter/ablation_dit_dcae_no_bottleneck_lr_range.sbatch`.
- Entry artifacts: `output/ablations/ablation_dit_dcae_no_bottleneck/entry_probe_<timestamp>/` plus sibling `.out`/`.err` logs.
- Cluster artifacts: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_dit_dcae_no_bottleneck_lr_range/`.
- Scheduler logs: `jsc_jupiter/logs/<SLURM_ID>_ablation_dit_dcae_no_bottleneck_lr_range.out` and `.err`.

The resolved choices, implemented files, exact probe measurements, selected batch, tests and submission ID will be recorded here as work proceeds. Other ablations remain out of scope until requested.

## Implemented paths

- `radargen/training/ppp_decoder.py`: `adapt_ppp_decoder_input` and `FullWidthDCAEDecoder`, keeping the shared scalar interface and checking native input-block structure before changing it.
- `radargen/training/ppp_initialization.py`: separate `dc_ae_no_bottleneck` decoder mode; remove the final DiT prediction layer and return full-width features.
- `scripts/train_ppp.py`: recognize the new decoder type and permit its measured batch size in the four-GPU LR sweep, instead of the legacy hardcoded batch five.
- `scripts/probe_ablation_batch.py`: latent-path support with the actual frozen condition encoder, live Gemma provider, and CAMEWrapper optimizer.
- `scripts/run_ablation_lr_range.py`: recognize this latent DiT variant while retaining fresh-model DDP confirmation and sweep behavior.
- `tests/test_dcae_no_bottleneck.py`: full-width channel gradients, retained stage weights, feature scaling, native 512 shape/budget, tiny DiT forward/checkpointing/strict restore, and legacy-path checks.

## Verification record

- Five new decoder tests passed after correcting meta-device autocast handling and explicitly initializing affine RMSNorm tensors in the tiny no-pretrained CPU fixture. The real decoder uses the loaded pretrained affine weights; those weights are unchanged.
- Three existing PPP model tests passed, including original scalar decoder scaling/initialization behavior.
- Six ablation-tool tests passed (random-DINO guards, memory extrapolation, LR calibration).
- Both YAMLs passed config-only validation; Python compilation, Slurm shell syntax and `git diff --check` passed.
- LR start is explicitly `1e-7`, end `1e-2`, without warm-up. Production-template LR is provisional and is not submitted as production training.

For a controlled comparison of decoder architectures, the standard-head counterpart should also start from these fresh SANA assets; historical standard-head runs inherited trained PPP backbone weights and are context rather than a matched initialization control.

## Entry-node result and selected batch

Exact probe directory: `output/ablations/ablation_dit_dcae_no_bottleneck/entry_probe_20261111T222827Z/`.

- Terminal logs: `output/ablations/ablation_dit_dcae_no_bottleneck/entry_probe_20261111T222827Z.out` and `.err`.
- `source_config.yaml`: exact tested model configuration.
- `gpu_inventory.csv`: entry GPU identity/capacity and observed usage.
- `train_log.log`: native SANA/DC-AE/optimizer messages.
- `updates.jsonl`: per-update loss, timing and peak allocated/reserved bytes.
- `batch_recommendation_shared_entry.json`: original estimate retaining unrelated shared-entry occupancy.
- `dedicated_target_reestimate.json`: reproducible fit inputs and dedicated-target recalculation.
- `batch_recommendation.json`: selected dedicated-node recommendation supplied to Slurm.

Nine real updates completed at batches 1/2/4, with finite losses and gradient norms, and a verified DiT parameter change. Live frozen Gemma, frozen DC-AE conditioning encoding and CAMEWrapper are included in the measurement. No trained PPP checkpoint was loaded. Parameter counts measured from the real initialized predictor: **761,393,162 total and trainable**.

| Batch/GPU | Peak allocated (GB decimal) | Peak reserved (GB decimal) | Slowest of last two updates (s) |
| --- | ---: | ---: | ---: |
| 1 | 18.86 | 22.80 | 2.336 |
| 2 | 25.41 | 30.61 | 3.533 |
| 4 | 38.51 | 46.26 | 5.966 |

Selected initial DDP batch: **4 samples/GPU**, **16 effective batch**, accumulation one. Estimated per-GPU memory: **59.14 GB**, including 10% memory-fit uncertainty and **8.24 GB** DDP/context allowance, below 90 GB decimal.

At probe start, **55.11 GB** was occupied outside this process on the shared entry GPU. Retaining that unrelated occupancy would incorrectly constrain the dedicated four-GPU allocation to batch one. The submitted estimate excludes that external occupancy, while preserving all measured memory belonging to this predictor, its frozen encoders/text model and CAME state. The raw conservative result is retained separately. The probe now supports `--dedicated-target` for this distinction in future runs; no extra GPU updates were needed to recompute the estimate.

The fit selects batches in multiples of four where possible; batch four was actually measured, and the added DDP allowance remains an estimate. The compute job performs five successful DDP updates before accepting the batch, checks rank-zero allocated/reserved peaks, records all GPUs' driver memory once per second, and retries a smaller batch on memory failure. Numerical or other non-memory failures stop the job for review.

## LR output files

Inside the eventual job directory:

- `submitted_config.yaml`, `entry_batch_recommendation.json`: archived source config and selected estimate.
- `gpu_inventory.csv`, `gpu_memory.csv`: all allocated GPU identities and driver telemetry.
- `ddp_probe_b<N>/config.yaml`, `train.out`, `train.err`, `gpu_memory.jsonl`: five-update confirmation attempt(s).
- `ddp_probe_summary.json`: confirmation results and peak memory.
- `lr_calibration.json`: measured DDP cadence, planned updates, batch and 1800-second guard.
- `sweep/config.yaml`, `sweep/gpu_memory.jsonl`: resolved LR config and training memory trace.
- `lr_range.out`, `lr_range.err`: raw sweep outputs, with per-update actual LR/loss.
- `lr_range/metrics.jsonl`, `lr_range/lr_loss.png`: structured LR/loss values and smoothed plot.
- `run_status.json`: sweep/plot exit codes, logged updates, last LR, whether the upper endpoint was reached.

The 45-minute allocation includes confirmation, model/data/text startup and plotting; the sweep's training guard is 30 minutes. DDP confirmation updates are discarded; a new model and optimizer are initialized for the actual range. Throughput changes can cause the guard to stop before `1e-2`; high-LR divergence may also stop it earlier. Neither case is reported as a completed range.

## Submission record

- SLURM job **2847195**, `ablation_dit_dcae_no_bottleneck_lr_range`.
- Reservation: **develbooster**; partition: `booster`; one node / four GH200 GPUs; 45-minute allocation with a 30-minute sweep training guard.
- Recommendation argument: `/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ablations/ablation_dit_dcae_no_bottleneck/entry_probe_20261111T222827Z/batch_recommendation.json`.
- Exact cluster output directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847195_ablation_dit_dcae_no_bottleneck_lr_range/`.
- Scheduler stdout: `jsc_jupiter/logs/2847195_ablation_dit_dcae_no_bottleneck_lr_range.out`.
- Scheduler stderr: `jsc_jupiter/logs/2847195_ablation_dit_dcae_no_bottleneck_lr_range.err`.
- Expected LR plot: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847195_ablation_dit_dcae_no_bottleneck_lr_range/lr_range/lr_loss.png`.
- Submitted is not completed: inspect Slurm state, `ddp_probe_summary.json`, `lr_calibration.json` and `run_status.json` after execution.

No production training or additional ablation was submitted. Stop here and ask the user before implementing the next model.

## LR selection — 2026-11-12

After reviewing the zoomed LR-range plot from job **2847195** (1e-5 to 1e-3), the user selected **1e-4** for production training, superseding the earlier 2e-4 choice. This records the LR decision only; no production run has been submitted for this model.

## Production submitted — 2026-11-12

### Final checkpoint and SS50

Job **2849053** completed with exit code 0 after 11h51m25s, final checkpoint `checkpoints/epoch_21_step_28537.pth`. User requested fixed SS50 evaluation on the entry node, serially before compact-DINO. Config: `evaluation/configs/PPP_validation_ss50_2849053_step28537.yaml`; same reference 50 samples and seed 42 as prior comparisons, metrics only.

Results: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/`. Entry launcher: `output/ablations/ss50_dit_compact_final/run_entry.sh`; log: sibling `2849053.log`; exit status: `2849053.exit_code`. Completion must be confirmed from status and summary.

User authorized production job **2849053**. Config `configs/ablation_dit_dcae_no_bottleneck_train_4nodes.yaml`; launcher `jsc_jupiter/ablation_dit_dcae_no_bottleneck_train_4nodes.sbatch`. Configuration-only validation and shell syntax passed before submission.

- Four regular booster nodes / 16 GPUs, 12-hour allocation, 11.8-hour training guard. No develbooster reservation.
- Batch **4/GPU**, accumulation one, **64 global**. LR confirmation measured rank-zero peak 48,538,583,040 bytes; five DDP updates passed.
- Constant **1e-4 after 200 warm-up optimizer updates**, no automatic LR scaling; existing CAMEWrapper, gradient clip 0.1 and loss settings retained.
- **32,000-update upper cap**. One-node calibration 1.387 seconds/update suggests approximately 30,600 updates in 11.8 hours before extra multi-node/checkpoint/diagnostic overhead. Actual count is time-limited, not guaranteed.
- Fresh original SANA + RGB DC-AE pretrained assets; no trained PPP or LR-sweep weights. Decoder trainable, conditioning encoder frozen.
- Checkpoints every 1,000 updates, all retained; TensorBoard, memory logs and prediction diagnostics every 1,000 updates enabled. No automatic second allocation.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2849053_ablation_dit_dcae_no_bottleneck_train_4nodes/`, including archived config/launcher, node list, launch manifest, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/2849053_ablation_dit_dcae_no_bottleneck_train_4nodes.out` and `.err`.


## SS50 completion verified

Entry-node validation exited **0**; summary reports `complete: true`, **50/50** samples and 50 metric records. Scene/frame/seed tuples match the random-DINO SS50 comparison in identical order.

| Metric | Mean |
|---|---:|
| chamfer_distance_all_norm | 0.038992 |
| chamfer_distance_xy | 1.673151 |
| count_abs_error | 200.940000 |
| distance_attr_f1 | 0.268136 |
| distance_attr_precision | 0.289203 |
| distance_attr_recall | 0.256290 |
| doppler_mae_at_detections | 2.192674 |
| doppler_mae_at_target_cells | 2.192674 |
| generated_count | 986.120000 |
| gt_count | 1095.540000 |
| inference_time_seconds | 0.182602 |
| iou_th1 | 0.312679 |
| pcl_mmd_doppler | 0.284737 |
| pcl_mmd_rcs | 0.186247 |
| pcl_mmd_xy | 0.048421 |
| ppp_expected_total_count | 991.987485 |
| ppp_total_count_abs_error | 194.446012 |
| rcs_mae_at_detections | 4.922243 |
| rcs_mae_at_target_cells | 4.922243 |
| target_collision_count | 40.320000 |

Full summary and foreground metrics: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json`.


## Second 12-hour allocation — exact resume, job 2867524

User authorized continuation after reviewing loss and SS50. Submitted separately in order: DiT then compact DINO. Launcher: `jsc_jupiter/ablation_dit_dcae_no_bottleneck_resume_4nodes.sbatch`; original production config: `configs/ablation_dit_dcae_no_bottleneck_train_4nodes.yaml`, verified byte-identical to source job's archived config.

- Source **2849053**, checkpoint `checkpoints/epoch_21_step_28537.pth`, global step **28,537**. Full `--resume` restores model, optimizer, scheduler, training position and all 16 rank RNG states.
- Saved optimizer verified: **668** populated parameter states; all groups at **1e-4**. Saved scheduler verified: `last_epoch=456592` and last LR **1e-4**, already beyond warm-up.
- **No repeated warm-up**, no scheduler replacement, no weights-only transfer. Historical 200-update warm-up remains in the unchanged config for exact-resume contract compatibility; restored scheduler continues past it.
- Four regular booster nodes / 16 GPUs, **12-hour allocation**, **11.8-hour training guard**, constant LR **1e-4**, batch **4/GPU**, **64 global**, accumulation one.
- Up to **32,000 additional updates**, hence at most global step **60,537**; time guard may stop earlier. Checkpoints every 1,000 global updates, all retained. No automatic further allocation.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2867524_ablation_dit_dcae_no_bottleneck_resume_4nodes/`; archived config/launcher/manifest, node list, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/2867524_ablation_dit_dcae_no_bottleneck_resume_4nodes.out` and `.err`.
- Verification: checkpoint metadata checks, unchanged-config comparison, launcher shell syntax and diff checks passed before submission.
