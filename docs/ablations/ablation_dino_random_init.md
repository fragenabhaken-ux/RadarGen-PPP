# ablation_dino_random_init

Status: implemented; entry-node probe passed; LR-range job **2846677** submitted on 2026-11-11. No production training submitted.

## Scientific change

Use the final DINO architecture, but initialize the DINOv3 ViT-S/16 backbone randomly. No RGB pretrained file and no previous PPP checkpoint may be loaded. The full backbone, modality-specific LoRA, 12-block 1152-wide RoPE neck and standard 1024-output head train. The model retains the architecture's normal initialization (including zero-contribution LoRA initialization and ICNR/head initialization); random initialization does not mean replacing these schemes with arbitrary Gaussian weights.

Unchanged: three 512 BEV conditioning maps, continuous detection targets, cubic mark links and signed L1, equal PPP/RCS/Doppler weights, AdamW, BF16, activation checkpointing, epoch-wise shuffling, seed 42. Memory probes and LR tests disable dropout/prediction rendering and saving so memory and timing represent training updates.

## Files and outputs

Shared procedure and naming: [programme README](README.md).

| File | Purpose |
| --- | --- |
| `configs/ablation_dino_random_init.yaml` | Full architecture/production-settings template; LR and batch provisional, no production job submitted |
| `configs/ablation_dino_random_init_lr_range.yaml` | Disposable exponential LR sweep; 30-minute training guard |
| `radargen/training/ppp_initialization.py` | Explicit `backbone_initialization: random`; existing pretrained mode retains strict local-file checks |
| `scripts/train_ppp.py` | Validate random/frozen/weight-file combinations; reject weights-only PPP transfers into random ablation |
| `scripts/probe_ablation_batch.py` | Single-GPU real-data AdamW probe and conservative DDP extrapolation |
| `scripts/run_ablation_lr_range.py` | Five-update DDP confirmation; automatic smaller-batch retry for memory failures; fresh timed LR sweep |
| `jsc_jupiter/ablation_dino_random_init_lr_range.sbatch` | One node, four GPUs, 45-minute allocation including overhead |
| `tests/test_ablation_dino_random_init.py` | Guard initialization semantics, memory estimate and LR-window calibration |

The random flag is consumed by initialization, not passed into the model constructor. `checkpoint_path` is empty and `pretrained` is false. Exact resume of a future random-initialization checkpoint remains supported; weights-only initialization from previous PPP models is rejected.

## Validation and decisions

- Random initialization must be explicit; an accidentally missing pretrained file must still fail in pretrained mode.
- Probe actual optimizer updates so AdamW state is included.
- Four-GPU effective batch is four times per-GPU batch; memory remains per GPU.
- Memory target: below 90 GB decimal per GPU, with additional DDP and estimation margin.
- LR test: fresh initialization, exponential `1e-7 → 1e-2`, no warmup, 30-minute training guard. Submitted allocation includes overhead beyond the training window.

## Entry-node test result

Probe directory: `output/ablations/ablation_dino_random_init/entry_probe_20261111T215600Z/`.

- Raw terminal logs: `output/ablations/ablation_dino_random_init/entry_probe_20261111T215600Z.out` and `.err` (siblings of the directory).
- `source_config.yaml`: exact configuration used.
- `gpu_inventory.csv`: entry GPU model, UUID, total/used memory.
- `updates.jsonl`: per-update loss, elapsed time, peak allocated/reserved bytes.
- `batch_recommendation.json`: measurements, fit, DDP allowance, cap and proposed batch.
- Real samples from eight positions across the training manifest; batches 1, 4, 8; three full optimizer updates each. Dataset has 83,554 training samples.
- Model count: **239,863,331 total and trainable parameters**. A backbone parameter changed; all nine updates had finite loss and gradient norm.

| Batch/GPU | Peak allocated (GB decimal) | Peak reserved (GB decimal) | Slowest of last two updates (s) |
| --- | ---: | ---: | ---: |
| 1 | 5.93 | 7.42 | 0.676 |
| 4 | 14.80 | 15.80 | 1.057 |
| 8 | 26.63 | 27.28 | 2.197 |

Recommended initial four-GPU test: **16 samples/GPU**, **64 total**, accumulation one. Estimated per-GPU peak including allowances: **77.83 GB**; cap **90 GB decimal**. Estimate includes **18.26 GB** already occupied on the shared entry GPU, a **4.07 GB** DDP/context allowance, and 10% model-memory uncertainty. Occupied entry-node memory is conservatively retained even though a dedicated compute allocation may not have it. This is not a measured batch-16 DDP peak.

The LR job confirms the selected batch with five DDP optimizer updates and verifies rank-zero peak allocated/reserved memory below 90 GB; all four ranks must complete. It additionally records driver memory for every allocated GPU each second in `gpu_memory.csv`. Non-memory failures stop the job; memory failures retry a smaller batch.

## LR-range outputs and budget

Run directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_dino_random_init_lr_range/`.

- `submitted_config.yaml`, `entry_batch_recommendation.json`: archived source config and entry estimate.
- `gpu_inventory.csv`, `gpu_memory.csv`: device inventory and per-GPU driver telemetry.
- `ddp_probe_b<N>/config.yaml`, `train.out`, `train.err`, `gpu_memory.jsonl`: confirmation attempt and five-update trace.
- `ddp_probe_summary.json`: batch attempts and measured rank-zero peak.
- `lr_calibration.json`: measured DDP update cadence, resolved sweep length and effective batch.
- `sweep/config.yaml`: actual LR test settings; batch/update count resolved after confirmation.
- `sweep/gpu_memory.jsonl`: training-step memory trace.
- `lr_range.out`, `lr_range.err`: actual sweep stdout/stderr, including each optimizer LR and loss.
- `lr_range/metrics.jsonl`, `lr_range/lr_loss.png`: structured LR/loss data and smoothed plot.
- `run_status.json`: process/plot status, logged updates, last LR, whether the endpoint was reached.

Compute: **30 minutes training wall time on four GH200 GPUs**, at most approximately **2 GPU-hours for the sweep**, plus confirmation/startup/plotting. The allocation is 45 minutes to provide overhead; it is not a 45-minute sweep. The exponential update budget is estimated from the five confirmation updates. The 1800-second guard can end the sweep before `1e-2`; numerical divergence can also stop it early. Both cases remain visible in the raw log and status JSON. The sweep starts with a fresh seed-42 model and AdamW state; confirmation updates are discarded. No predictions, TensorBoard or checkpoints are written.

## Validation and submission record

- Six focused tests passed: random initialization bypasses RGB checkpoint loading; invalid random/pretrained/frozen combinations are rejected; pretrained missing-file checks remain strict; resume construction works; DDP memory extrapolation respects its per-GPU budget; LR update count follows the measured 30-minute cadence.
- Six existing DINO training tests passed, including exact checkpoint resume and smoke-check restoration using tiny CPU substitutes.
- Both ablation YAMLs passed config-only validation. Python syntax, Slurm shell syntax and `git diff --check` passed.
- Real single-GPU entry test: nine complete updates, no OOM/nonfinite failures; measured results above.
- Submitted job: **2846677**, `ablation_dino_random_init_lr_range`, one node / four GH200 GPUs, requested 45-minute allocation with 30-minute sweep window.
- Exact recommendation argument: `/e/project1/nxtaim-1/huber7/repos/RadarGen/output/ablations/ablation_dino_random_init/entry_probe_20261111T215600Z/batch_recommendation.json`.
- Exact run output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2846677_ablation_dino_random_init_lr_range/`.
- Scheduler stdout: `jsc_jupiter/logs/2846677_ablation_dino_random_init_lr_range.out`.
- Scheduler stderr: `jsc_jupiter/logs/2846677_ablation_dino_random_init_lr_range.err`.
- Expected LR plot: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2846677_ablation_dino_random_init_lr_range/lr_range/lr_loss.png`.
- Submission is not evidence of a completed sweep. Check `ddp_probe_summary.json`, `lr_calibration.json`, `run_status.json`, and Slurm status after execution. A nonfinite high-LR failure is retained rather than treated as a successful completed range.

Next model, only after user confirmation: **`ablation_dino_frozen_lora`**. Remaining ablations are not implemented or submitted.

## First production allocation — approved 2026-11-12

The user approved **only the first 12-hour allocation**, aiming eventually at two allocations with a review between them. The remaining model ablations remain on hold. Do not automatically submit the second allocation.

- Config: `configs/ablation_dino_random_init_train_4nodes.yaml`.
- Launcher: `jsc_jupiter/ablation_dino_random_init_train_4nodes.sbatch`.
- Four nodes, four GH200 GPUs each: **16 GPU workers**. Regular **booster** production partition, no short-test `develbooster` reservation.
- Batch **16/GPU**, accumulation one, **256 effective batch**. No automatic LR scaling.
- Fresh seed-42 random backbone, LoRA, standard 12×1152 neck and standard 1024 head. **No pretrained weights, no LR-sweep weights and no PPP checkpoint transfer.**
- **15,000 new optimizer updates maximum**, including warmup, passed explicitly via `--max-updates 15000`.
- **200 optimizer-update linear warmup → constant 1e-4**, no decay. The existing trainer multiplies the internal constant-scheduler warmup by world size because Accelerate advances it once per rank; a focused test verifies it reaches the plateau after 200 optimizer updates, not 3200.
- **11.8-hour training guard** inside the 12-hour Slurm allocation. Stop at update cap or guard, whichever comes first. The estimate (2.79 s/update on one node) is not guaranteed four-node throughput.
- `num_epochs: 1000` avoids an epoch ceiling ending the allocation early or complicating later continuation; the actual controls are the update cap and time guard.
- Checkpoints every **1,000 updates** and on orderly final stop. `keep_last_checkpoints: 0` retains every numbered checkpoint, with `latest.pth` pointing to the latest one.
- Checkpoints include model weights, AdamW state, scheduler state, epoch/batch/global progress, manifest contract, and RNG states for all sixteen ranks. A later allocation can continue without repeating warmup; any LR change will be decided explicitly after reviewing this run.
- TensorBoard enabled; loss/LR logs every 20 updates; prediction diagnostics every 1,000 updates and at stop, matching the previous production convention.

### Production output layout

Run directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<SLURM_ID>_ablation_dino_random_init_train_4nodes/`.

- `submitted_config.yaml`, `submitted_launcher.sbatch`, `nodes.txt`, `launch_manifest.json`: executed setup, node list, update cap and no-auto-continuation policy.
- `checkpoints/epoch_<epoch>_step_<step>.pth` and `checkpoints/latest.pth`: numbered/full-state checkpoints.
- `tensorboard/`: training scalars and prediction figures.
- `gpu_memory.jsonl`: rank-zero training memory trace.
- `train_log.log`: native model/checkpoint messages when emitted by the existing logger.
- Scheduler stdout/stderr: `jsc_jupiter/logs/<SLURM_ID>_ablation_dino_random_init_train_4nodes.out` and `.err`; the `.err` contains the `PPP update ...` loss/LR records.

Between allocations, review loss trends, prediction diagnostics and validation results before choosing a checkpoint and retaining or changing LR. No second job or validation job is chained automatically.

### Production submission record

- Job **2847666**, `ablation_dino_random_init_train_4nodes`, submitted 2026-11-12.
- **Four booster nodes / sixteen GH200 GPUs**, 12-hour allocation; no reservation and no chained second allocation.
- Exact output directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847666_ablation_dino_random_init_train_4nodes/`.
- Exact checkpoints: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847666_ablation_dino_random_init_train_4nodes/checkpoints/`.
- Exact TensorBoard directory: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2847666_ablation_dino_random_init_train_4nodes/tensorboard/`.
- Scheduler stdout: `jsc_jupiter/logs/2847666_ablation_dino_random_init_train_4nodes.out`.
- Scheduler stderr/loss log: `jsc_jupiter/logs/2847666_ablation_dino_random_init_train_4nodes.err`.
- Two production-specific tests passed: agreed fresh-random config and the actual Accelerate wrapper's sixteen-rank 200-update warmup. Configuration-only validation, shell syntax and `git diff --check` passed.
- Checkpoint updates are counted from zero for this fresh model, not from the LR sweep. The 15k cap applies to this invocation. A future explicitly approved resume can add another invocation's updates while retaining the global counter and saved optimizer/scheduler state.
- Submission is not completion; check Slurm status and update logs. Review this allocation before approving or configuring any second allocation.
# Final-checkpoint SS50 validation — 2026-11-12

## Second 12-hour allocation — exact resume

Following SS50/loss review, user authorized continuation job **2863217**, `ablation_dino_random_init_resume_4nodes`, from job **2847666**, checkpoint `epoch_45_step_15000.pth` (epoch 45, batch 285, global step 15,000).

- Launcher: `jsc_jupiter/ablation_dino_random_init_resume_4nodes.sbatch`; identical original production config `configs/ablation_dino_random_init_train_4nodes.yaml` (byte-matched against the source run's submitted config).
- Exact `--resume`: restore model, AdamW state (628 populated parameter states), scheduler, progress and all 16 rank RNG states. Verified saved optimizer LR **1e-4**, scheduler `last_epoch=240000` (16 scheduler ticks/update), `_last_lr=[1e-4]`.
- **No repeated warm-up.** Original config retains its historical 200-update warm-up to satisfy exact-resume compatibility; restored scheduler has already completed it. No scheduler replacement or weights-only initialization.
- Four regular booster nodes / 16 GPUs, batch **16/GPU**, **256 global**, constant **1e-4**, 12-hour allocation and 11.8-hour guard; checkpoints every 1,000 global updates, all retained.
- CLI `--max-updates 15000` counts **new updates in this invocation**, allowing global step **30,000** if time permits. No further automatic allocation.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2863217_ablation_dino_random_init_resume_4nodes/`, standard archived config/launcher/manifest, checkpoints, TensorBoard and memory logs.
- Scheduler logs: `jsc_jupiter/logs/2863217_ablation_dino_random_init_resume_4nodes.out` and `.err`.

Production job **2847666** completed successfully (11h49m07s), saving `checkpoints/epoch_45_step_15000.pth`. User requested SS50 on the entry node. Launched directly (no Slurm allocation), using `evaluation/configs/PPP_validation_ss50_2847666_step15000.yaml`, fixed reference sample order from validation 2455164, seed/selection seed 42, metrics only (no plots/prediction exports).

- Results: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/` (`metrics.jsonl`, `summary.json` and native validation provenance).
- Persistent launcher: `output/ablations/ablation_dino_random_init/ss50_step15000/run_entry.sh`.
- Log: `output/ablations/ablation_dino_random_init/ss50_step15000/validation.log`; exit code: sibling `exit_code.txt` when finished.
- Initial detached process did not survive startup; relaunched in a persistent entry-node execution session, Python PID **2777794**. Submission/startup is not evidence of successful completion; inspect exit status and summary.

**Completed successfully:** exit code 0; summary reports `complete: true`, 50/50 samples and 50 metric records. SS50 mean detection F1 **0.266862**, XY Chamfer **1.738812**, sampled-count absolute error **179.78**, RCS MAE at detections **4.863968**, Doppler MAE **1.952295**. Full metrics and provenance are in the results directory above. Entry-node inference timing is shared-hardware timing and should not be treated as a controlled speed comparison.



## Second allocation completed; full validation submitted — 2026-11-13

Training job **2863217** completed with exit 0, final global step **29,915** (`epoch_91_step_29915.pth`), elapsed 11h50m08s. Full-validation/all-validation-scene export bundle **2889167** submitted to develbooster (four GPUs / two hours), with PNGs, reusable predictions and separate comparison against final DINO, best-F1 DINO and RadarGen. See `full_validation_scene_exports.md` for corrected worker GPU binding and exact output paths.
