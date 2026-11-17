# Architecture ablation programme

Paper-writing input: [Detailed architecture brief for all four implemented ablations](paper_architecture_brief.md), including motivations, exact architecture/initialization, trainability, parameter accounting and comparison caveats.

Status: **remaining ablations on hold at the user’s request (2026-11-11)**. Do not implement or submit additional models until explicitly resumed. Already submitted jobs remain unchanged. The user separately authorized the first four-node random-DINO production allocation on 2026-11-12; this does not resume the remaining model ablations.

When resumed, implementation proceeds **one model at a time**. After implementing, testing and submitting the LR-range experiment for one model, stop and ask before beginning the next. Clarify scientifically ambiguous choices with the user before implementing dependent changes. Production training requires separate user authorization; the first random-DINO 12-hour allocation has now been explicitly authorized.

## Models and order

Existing references: `ablation_dino_reference` (fully trainable DINO + modality LoRA, 12-block 1152-wide neck, standard attention/PixelShuffle head, 1024 output) and `ablation_dit_reference` (DiT with standard joint head, 512 output). References describe architecture; historical production checkpoints have different training histories and are not controlled fresh-training comparisons.

| Order | Name | Single change from reference | Status |
| --- | --- | --- | --- |
| 1 | `ablation_dino_random_init` | Random DINO backbone instead of pretrained weights; all backbone weights and LoRA train; neck/head start fresh | LR job `2846677` completed; first four-node production job `2847666` submitted |
| 2 | `ablation_dino_frozen_lora` | Freeze pretrained base DINO, train modality-specific LoRA | On hold |
| 3 | `ablation_resnet50_backbone` | Pretrained ResNet-50, stride-16 features projected to 384 channels/modality; retain 12-block 1152-wide neck and standard head | Implemented; tests and entry probe passed; develbooster LR job `2846865` submitted |
| 4 | `ablation_dino_neck6` | Six neck blocks instead of twelve | On hold |
| 5 | `ablation_dino_neck0` | No neck blocks; feed concatenated DINO features directly to standard head; head attention remains | On hold |
| 6 | `ablation_dino_output512` | 512 output instead of 1024 | On hold |
| 7 | `ablation_dit_dcae_no_bottleneck` | Original DC-AE output decoder with widened 1152-channel input projection, removing the preceding 32-channel projection | Implemented; 14 tests and real entry probe passed; develbooster LR job `2847195` submitted |
| 8 | `ablation_dit_output1024` | Standard head with 1024 output instead of 512 | On hold |
| 9 | `ablation_dino_neck_compact` | Twelve 1152-residual blocks; 4×128 attention channels, 128-hidden parallel MLP; unchanged head/output | Implemented; 30 tests and real entry probe passed; develbooster LR job `2847479` submitted |

Optional later, also on hold: `ablation_dino_neck4`. No full factorial combination is planned. Reference weights, frozen/unfrozen settings, resolution, data order, loss, batch size and training budget must be recorded for every comparison. External pretraining and previous PPP checkpoint transfers are distinct and must not be conflated.

## Per-model procedure

1. Implement an explicit config and any necessary model support. Record exactly what is randomly initialized, loaded or frozen. Use continuous detection targets and existing loss semantics.
2. Run focused CPU/configuration checks.
3. Run a **very short single-GPU entry-node probe** with real forward/backward/optimizer updates, BF16 and gradient checkpointing. Measure multiple microbatch sizes after optimizer-state initialization, without prediction rendering. Do not terminate unrelated GPU processes.
4. Estimate per-GPU memory at candidate batch sizes and extrapolate to **four replicated GPU workers**. DDP does not divide the model memory by four: effective batch = four times per-GPU microbatch (accumulation one). Include a conservative allowance for DDP communication buffers and uncertainty. For dedicated Slurm targets, exclude unrelated shared entry-node processes from the target estimate, but retain their observed usage in the record; the probe supports `--dedicated-target`. Target less than **90 GB decimal per GPU** (83.82 GiB), which is stricter than 90 GiB. State that extrapolation is an estimate.
5. Submit a one-node/four-GPU **LR-range test with a 30-minute training window**, allowing separate allocation overhead for startup and plotting. Use the established exponential range `1e-7` to `1e-2`, without warmup, fresh model/optimizer and no production checkpoints. Record update-budget calibration, achieved endpoint, early failure, memory and runtime. A short DDP confirmation should verify the extrapolated batch before the sweep and reduce it if necessary.
6. Document source files, configs, exact output locations, probe results, submission ID and job status. Prompt the user to proceed to the next model. Do not wait for a queued 30-minute job to finish before returning unless needed to resolve a failure.

## Output conventions

- Tracked implementation: `configs/ablation_*.yaml`, `scripts/*ablation*.py`, `jsc_jupiter/ablation_*.sbatch`.
- Model notes: one Markdown file per model in this directory.
- Entry-node probe artifacts: `output/ablations/<model>/entry_probe_<timestamp>/` (logs, machine-readable measurements, memory fit and batch recommendation).
- Cluster runs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_<model>_lr_range/` (submitted config, probe evidence, GPU inventory/monitoring, DDP confirmation, resolved sweep config, raw LR/loss log, metrics and plot).
- Scheduler stdout/stderr: `jsc_jupiter/logs/<SLURM_ID>_<model>_lr_range.out` and `.err`.
- LR range tests are disposable; their weights and optimizer states must not initialize production runs.

## First model

See [randomly initialized DINO](ablation_dino_random_init.md). This keeps the final DINO architecture (239,863,331 parameters) and removes both external DINO pretraining and prior PPP transfers. A future pretrained comparison should start fresh neck/head weights from the same seed and load only the specified DINO pretrained backbone, with otherwise matched training settings.

## Current requested model

The user selected [ResNet-50](ablation_resnet50_backbone.md) next, ahead of frozen DINO. Implementation order follows explicit user requests; the table order is the original plan.

## Current requested model: DiT DC-AE without bottleneck

The user selected [the adapted DC-AE DiT decoder](ablation_dit_dcae_no_bottleneck.md) next. The user approved removing its fixed input shortcut and using the same fresh SANA/RGB DC-AE assets as the initialization of `2438928` (no trained PPP checkpoint). Implementation, tests, probe results and LR submission are recorded in the model notes.

## Current requested model: compact DINO neck

The user approved [the compact neck](ablation_dino_neck_compact.md): twelve blocks, four 128-channel heads, 128-hidden MLP branch, 1152 residual width, fresh original DINO initialization and no PPP transfer.

## Authorized production: random DINO, first allocation

Second allocation authorized after review and submitted as **2863217**, exact resume from **2847666 step 15,000**: optimizer/scheduler/RNG restored, constant 1e-4, no repeated warm-up, four nodes / 12 hours, up to 15,000 additional updates. See model notes for verified checkpoint metadata and output paths.

[Production settings and output layout](ablation_dino_random_init.md): four nodes, batch 16/GPU, 15k-update cap, 200-update warmup then constant 1e-4, 11.8-hour guard and 1k checkpoints. Submit only the first 12-hour job; inspect before deciding on the second. All other ablations remain on hold.

First random-DINO production job: **2847666** (`ablation_dino_random_init_train_4nodes`), four regular-booster nodes, 12 hours. The second allocation remains unsubmitted pending review.

## Continuation protocol — user instruction 2026-11-12

After reviewing the completed jobs' training losses, the user will explicitly request the next **12-hour, four-node allocation**. Do not submit continuations automatically. Continue with an **exact/full checkpoint resume**, loading model weights, optimizer state, scheduler state, training progress and RNG state; do not use a weights-only initialization. Preserve the existing constant LR and scheduler progress, **without restarting warm-up**. Before submitting, verify the selected checkpoint and resume semantics; iteration limits must allow the intended additional training rather than inadvertently retaining an already reached absolute step cap. Any LR adjustment requires the user's instruction after review.

## ResNet production submitted

Current replacement job: **2856174**, submitted after fixing the startup prediction diagnostic; **2848698 failed before training**. Nine relevant tests passed before resubmission. Same training settings and output layout, with the new job ID; details in the model document.

The user selected LR 1e-4 for ResNet. [Its four-node configuration](ablation_resnet50_backbone.md) uses 200-update warm-up, batch 20/GPU, a 15k upper cap, 11.8-hour guard and 1k checkpoints. First production job **2848698** submitted on 2026-11-12; initially pending priority on regular booster. No second allocation is chained; remaining model ablations remain on hold.

## DiT without bottleneck production submitted

Job **2849053**, [settings and output paths](ablation_dit_dcae_no_bottleneck.md): four booster nodes, batch 4/GPU (64 global), LR 1e-4 after 200 warm-up updates, 32k update ceiling, 11.8-hour guard within 12-hour allocation, checkpoints every 1k. Fresh SANA/RGB DC-AE assets, no trained PPP transfer.

## Compact-DINO production submitted

Job **2849062**, [settings and output paths](ablation_dino_neck_compact.md): four booster nodes, batch 24/GPU (384 global), LR 1e-5 after 200 warm-up updates, 15k update ceiling, 11.8-hour guard within 12-hour allocation, checkpoints every 1k. Original pretrained DINO, fresh LoRA/compact neck/head. Both jobs submitted separately in that order on 2026-11-12; no automatic second allocations. Unimplemented ablations remain on hold.


## Second allocations: DiT and compact DINO

User approved exact-resume continuations after review: **2867524** (DiT from **2849053 step 28,537**, LR 1e-4, batch 4/GPU) and **2867528** (compact DINO from **2849062 step 10,372**, LR 1e-5, batch 24/GPU). Both four nodes / 12 hours, 11.8-hour guard, optimizer/scheduler/progress/RNG restored, no repeated warm-up, checkpoints every 1k. See individual model notes for launchers and output paths. No automatic further allocations.


## ResNet second allocation prepared (not submitted)

Exact resume from **2856174 step 12,575**, same LR 1e-4, optimizer/scheduler/RNG restored, no repeated warm-up, batch 20/GPU (320 global), four nodes / 12 hours, 11.8-hour guard and 1k checkpoints. Launcher `jsc_jupiter/ablation_resnet50_backbone_resume_4nodes.sbatch`; detailed verified state and output paths in the model notes. Prepared at user's request; no continuation submitted yet.


ResNet continuation subsequently approved and submitted as **2882612** on 2026-11-13; this supersedes its prepared/not-submitted status above. Exact resume from step 12,575, constant 1e-4, no repeated warm-up, four nodes / 12 hours. See model notes for paths.


## Full validation and all validation-scene predictions

After second allocations, user requested [bundled full validation with PNGs and all-scene exports](full_validation_scene_exports.md), replacing GPU-blocked entry SS50. Submitted **2887635** (DiT 58,057) and **2887636** (compact DINO 20,595), develbooster, one node / four GPUs / two hours per job. Each runs 1,906-keyframe full validation plus 9,481 temporal predictions across 49 validation scenes, with separate comparisons against final DINO, best-F1-checkpoint DINO and RadarGen. Check worker completion records before interpreting results.


Current full-validation bundle IDs after explicit GPU-binding fix: **2889167** (random DINO second run, step 29,915), **2889168** (DiT replacement/resume), **2889169** (compact-DINO replacement/resume). Old bundles 2887635 failed and 2887636 was stopped; saved outputs are retained and resumed. See full-validation/export notes for details.


[Standard ten-scene comparison videos for all three second-run ablations](standard10_second_run_videos.md): 30 videos, each GT + one ablation + DINO 15k + RadarGen, reusing full-scene exports on entry-node CPUs. See the video index/status for completion.


[First-versus-second final checkpoint audit and matched VoxelNeXt protocol](first_second_checkpoint_audit.md), requested 2026-11-14: completed SS50 comparisons for DiT, compact DINO and random DINO. DiT regresses on F1/geometry/count/RCS; Random DINO has flat F1 and worse RCS. User checkpoint clarification pending for these two. Compact DINO second-final improves F1 and was submitted to frozen-detector evaluation as **2895299**. ResNet second run still active at audit time.


## All eight fixed-budget checkpoints: complete evaluation programme

User authorized every evaluation for both final first- and second-run checkpoints, superseding the earlier checkpoint-choice holds. [Programme and exact output paths](all_checkpoint_evaluation_programme.md); [live consolidated index](all_checkpoint_evaluations_index.md). Four two-hour develbooster jobs: **2895546** DiT, **2895547** compact DINO, **2895548** random DINO, **2895549** ResNet (after successful training 2882612). GPU jobs cover missing SS50/full validation, VoxelNeXt and temporal predictions for ten standard video scenes; CPU entry coordinator handles count sweeps and 80 comparison videos, reusing completed work.


## Paper metric tables and official names

Model display names are **C3P-dar-SANA**, **C3P-dar-Dv3**, and **C3P-dar-RN50**. Random initialization and compact-neck variants retain the C3P-dar-Dv3 family name with explicit suffixes. Use `(ours)` when shown with RadarGen, consistent with paper-material instructions. Source/config class names and archived run IDs remain unchanged.

Completed source-backed LaTeX: [main tables (second-allocation finals)](paper_metric_tables.tex), [both first/second-final checkpoints](paper_metric_tables_all_checkpoints.tex). Generated by `scripts/build_paper_metric_tables.py`; exact source/checkpoint mapping in `paper_metric_table_sources.json`. Includes all complete 1,906-frame generation/foreground metrics, updated frozen-detector scores, class AP/TP errors, parameters and latency. Mean rankings use unrounded values, best bold and second-best underlined.
