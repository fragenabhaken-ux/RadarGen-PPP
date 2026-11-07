# Full-width DiT PPP decoder experiments


## Two comparison experiments

Both variants retain the full-width DiT output, continuous targets, 512×512
BEV geometry, shared scalar decoder, losses, mark links and initialization
strategy described below. Each decoder is independently initialized from
scratch. The configs differ only in decoder type and output directory/name;
training batch size, seed, loss weights and schedule are identical.

| Variant | `model.extra.decoder_type` | Decoder parameters |
| --- | --- | ---: |
| PixelShuffle + bilinear + ResNet | `wide_pixelshuffle` | 159,119,201 |
| Nearest-neighbor + EfficientViT | `wide_nearest_efficientvit` | 159,156,673 |
| Original nearest-neighbor + convolution-only | `wide` | 158,758,177 |

The original implementation and its checkpoints remain available through
`wide`; its architecture has not been silently changed.

### PixelShuffle + bilinear + ResNet

A 1×1 convolution maps 1152→2048 channels at 16×16. The three stages are:

1. PixelShuffle(2), bilinear ×2 → `(512,64,64)`, then 33 ResNet blocks.
2. PixelShuffle(2), bilinear ×2 → `(128,256,256)`, then three ResNet blocks.
3. PixelShuffle(2) → `(32,512,512)`, then three ResNet blocks.

A final 1×1 convolution maps 32→1 per modality stream. Each ResNet block has
two pre-activation GroupNorm(1) → SiLU → 3×3 convolution operations, plus an
identity shortcut. All convolutions have biases and all 3×3 kernels use
padding one. Bilinear interpolation uses `align_corners=False` with exact
2× sizes. This adapts the supervisor's `2H−1` interpolation example to the
existing 512px targets: the stage resolutions are 64/256/512 rather than
63/251/1003. There is no interpolation after the final PixelShuffle.

The default stage depths are fitted to the loaded scalar DC-AE budget,
reserving three blocks at each of the last two stages and spending the
remaining budget at 64×64. That yields 33/3/3 blocks for the current assets.
This makes the first stage deep; matching parameter counts does not match
compute, memory or throughput. Blocks are checkpointed individually.

### Nearest-neighbor + EfficientViT

Six EfficientViT blocks refine the `(1152,16,16)` features. Five nearest-neighbor
2× upsampling stages with 3×3 channel projections use widths
768/384/192/96/48 at 32/64/128/256/512 pixels. The first two stages use an
EfficientViT block each; the last three retain the original convolutional
residual refinement. A final 3×3 convolution maps 48→1.

The eight attention blocks reuse DC-AE's `EViTS5_GLU` structure: multi-scale
ReLU LiteMLA with scale-five aggregation and head dimension 32, followed by
a gated local inverted bottleneck (`GLUMBConv`), with residual connections,
RMSNorm2d and SiLU. RMSNorm affine weights/biases are initialized to one/zero
explicitly, since the standalone DC-AE block constructor does not do this.
The six low-resolution blocks use hidden width 4472, fitted to the old
parameter budget in multiples of eight. The two later EfficientViT blocks
retain expansion ratio four.

These are two complete decoder designs. Their upsampling and block types
both differ, so a quality difference cannot be attributed to just one of
those choices. Both decoders receive the same transferred DiT backbone.

### Launch

```bash
sbatch jsc_jupiter/PPP_train_wide_pixelshuffle.sbatch
sbatch jsc_jupiter/PPP_train_wide_nearest_efficientvit.sbatch
```

Each launcher uses its matching `configs/PPP_train_wide_*.yaml`. Both default
to the evaluated continuous-target finetuned backbone from job 2516862,
epoch 6, step 12534. As with the original launcher, argument one overrides
the config and argument two overrides the backbone checkpoint. No job is
submitted by implementation or tests. GPU memory and throughput need a
Slurm measurement before treating the inherited batch size as established.

## Original convolution-only variant

`configs/PPP_train_wide_decoder.yaml` uses continuous detection targets and
`RadarGenPPP_600M_P1_D28` with `model.extra.decoder_type: wide`. The frozen
DC-AE condition encoder, DiT conditioning, modality ordering, losses and cubic
mark links follow the continuous-target baseline.

The DiT returns its final transformer features `(3B,1152,16,16)` directly.
Its old final normalization/modulation/projection head is removed. The new
shared decoder has six residual refinement blocks at 16×16 followed by five
nearest-neighbor 2× upsampling stages with 3×3 convolutions and residual
refinement. Stage widths are 768, 384, 192, 96 and 48, reaching 512×512.
A final scalar convolution produces each modality's raw field. The wrapper
returns `(B,3,1,512,512)`. There is no VAE latent unscaling or output activation.

The scalar-adapted DC-AE decoder has 159,220,865 parameters. The new decoder
has 158,758,177 (99.71% of that budget). Its low-resolution refinement hidden
width is 2016, fitted to the actual loaded decoder budget in multiples of eight.
All budgeted parameters participate in the forward pass. Activation
checkpointing follows `train.extra.decoder_grad_checkpointing`, falling back
to `train.grad_checkpointing`.

## Training and initialization

Submit from the repository root:

```bash
sbatch jsc_jupiter/PPP_train_wide_decoder.sbatch
```

The prepared launcher transfers the continuous-target finetuned DiT from
job 2516862, epoch 6, step 12534, referenced by the existing validation config.
This identifies the evaluated finetuned model; it does not establish a new
ranking of validation checkpoints. Override the config and source checkpoint:

```bash
sbatch jsc_jupiter/PPP_train_wide_decoder.sbatch configs/PPP_train_wide_decoder.yaml /path/to/legacy_ppp.pth
```

`--init-backbone-from` loads every DiT backbone parameter strictly, excluding
only the old `dit.final_layer.*` head. The new decoder starts from random
weights; optimizer, scheduler and progress start fresh. Both backbone and
decoder train. The six-epoch learning-rate schedule is inherited from the
continuous finetuning config as a starting point, and has not been tuned for
a fresh decoder. The launcher has not been submitted automatically.

New-variant checkpoints use the existing strict `--resume` / `--init-from`
paths and `RadarGenPPPPipeline.from_config` with the new config. Legacy
checkpoints cannot be resumed as full wide-decoder checkpoints. Omitting
`decoder_type` preserves the original DC-AE output path.

## Validation

`tests/test_wide_ppp_decoder.py` checks the full-resolution shape using meta
tensors, the parameter budget, real tiny-DiT forward/backward, activation
checkpointing and strict backbone transfer. GPU throughput, peak memory and
training quality require a Slurm run.

The original implementation passed 13 focused checks. Variant tests additionally
cover both full-resolution parameter budgets, the intended upsampling/block
types, real tiny-DiT gradients and modality ordering, BF16 forward/backward,
equal activation-checkpointed gradients, normalization initialization and
strict checkpoint round trips. Full-resolution shape checks use meta tensors;
CPU numerical checks use smaller channel widths and rasters.

Current verification: all 18 focused decoder/model/checkpointing/training tests
passed. Both experiment configs passed CLI validation; launcher syntax and
patch whitespace checks passed. The existing manual accumulation reference
was updated to use its configured 2×2 loss geometry, matching the workspace's
resolution-aware training loop rather than PPPLoss's 512×512 default.

## Four-frame overfit and GPU memory

```bash
sbatch jsc_jupiter/PPP_overfit_wide_pixelshuffle.sbatch
sbatch jsc_jupiter/PPP_overfit_wide_nearest_efficientvit.sbatch
```

These short tests use partition `booster` with reservation `develbooster`
(which is a reservation on this cluster, not a partition), one GPU and a
90-minute allocation. They derive separate subset manifests from the checked
full manifest, selecting scene `018a60086f5e441fb09f476e55948b72` and runtime
indices 0/32/64/96. Microbatch size stays one; four-step accumulation gives
an effective batch of four. There is no batch-size sweep. Both tests transfer
the same finetuned backbone, train a fresh decoder with CAME at LR 1e-5,
50-update warmup, constant scheduling thereafter, gradient clipping 0.1, no conditioning dropout,
BF16 and activation checkpointing. They stop at 200 updates or one hour of
training, leaving allocation time for startup and diagnostics.

Diagnostics every 25 updates, including step zero and early stop, include all four
samples; checkpoints are saved every 25 updates and at orderly stop, with
the last two retained. Job-specific outputs are under
`/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<jobid>_<jobname>`.

`train.extra.log_gpu_memory: true` writes `gpu_memory.jsonl` after model/text
startup and after every optimizer update. Each training peak spans all four
microbatches, backward and optimizer work. Counters reset before training,
so diagnostic rendering does not contaminate allocated-memory peaks. Logs
print the first three updates and then every ten. Fields include allocated,
reserved, peak allocated/reserved, step-start allocated, device free and
device total memory in GiB. Explicit step logging disables the older
microbatch-10–12 decoder probe, whose internal peak resets would interfere.

`gpu_usage.csv` samples driver-reported GPU usage every two seconds.
`gpu_memory_summary.json`/`.txt` summarize measured peaks and a provisional
batch-one extrapolation after optimizer warmup. The extrapolation uses
`step_start + batch * step_extra` and reserves 15% of device memory plus
observed non-PyTorch allocations. Step-extra also includes gradients and
workspaces that do not scale linearly with batch: this is only a starting
estimate. Verify larger full-data/DDP batches before relying on it.
`overfit_summary.txt` and `overfit_diagnostics/learning_curves.png` summarize
the loss and prediction metrics; convergence is assessed from those outputs.

Submitted on 2026-11-02 after 14 focused checks passed:

- PixelShuffle: job `2652272`, `PPP_overfit_wide_pixelshuffle`.
- Nearest-neighbor + EfficientViT: job `2652273`, `PPP_overfit_wide_nearest_efficientvit`.

Both submissions use reservation `develbooster`, microbatch one and
accumulation four. Slurm logs are in `jsc_jupiter/logs/<jobid>_<jobname>.out`
and `.err`; runtime results use the job-specific scratch directories above.

The first PixelShuffle attempt (2652272) failed during step-zero diagnostic
sampling because its default random output head produced pathological initial
intensities. The first EfficientViT attempt (2652273) reached training but was
cancelled for a matched restart. Both new decoder output heads now use
nonzero Gaussian weights with std 1e-3 and zero bias; parameter counts and
checkpoint shapes are unchanged. Diagnostic sampling skips point panels
explicitly above 100,000 expected points while retaining fields and losses;
concentrated but tractable intensities use exact per-cell Poisson sampling
instead of excessive global thinning proposals. Thirteen focused decoder,
memory-summary and diagnostic tests passed after these changes.

Replacement submissions: PixelShuffle `2652351`, nearest-neighbor +
EfficientViT `2652352`. These are the runs to inspect for the comparison.
The earlier EfficientViT attempt measured about 16.38 GiB peak allocated and
17.54 GiB reserved at microbatch one; final measurements come from the new runs.


### Numerical-instability retry

Jobs 2652351 (PixelShuffle) and 2652352 (EfficientViT) completed 20 and 29
optimizer updates respectively, then failed with a nonfinite PPP likelihood
or gradient norm. Neither failure was a CUDA OOM. Their original overfit
settings used CAME LR 1e-4, clipping 1.0 and 20-update warmup; loss and gradient
norms grew sharply as warmup approached the full LR.

The retry uses the continuous finetuning baseline's LR 1e-5 and clipping 0.1,
with 50-update warmup. Logging is every five updates; checkpoints and
four-frame diagnostics are every 25, retaining two checkpoints. Decoder
architecture, small output-head initialization, links, loss weights, samples,
seed, effective batch four, BF16 and checkpointing stay unchanged. The retry
starts fresh from the same legacy finetuned backbone, since the failed runs
had not yet saved checkpoints. Both configs pass CLI validation.

Retry jobs: PixelShuffle `2652426`, EfficientViT `2652427`, reservation
`develbooster`. The failed runs still provide batch-one memory measurements:

| Decoder | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| PixelShuffle | 17.75 GiB | 18.95 GiB |
| Nearest-neighbor + EfficientViT | 16.38 GiB | 17.54 GiB |

These include optimizer states and accumulated backward/optimizer work.
They establish memory use, not convergence or a verified maximum batch.


The lower-LR retries were monitored beyond both previous failure points and
through the end of warmup. At update 50, four-frame mean total loss fell from
41.6006 to 16.6669 for PixelShuffle and from 41.5956 to 16.9346 for EfficientViT.
EfficientViT reached update 75 with diagnostic mean total loss 15.364. All
observed values remained finite. These are intermediate stability checks,
not a completed-run result or an overfitting pass claim.

## Continue PixelShuffle to 1,000 updates

Both 200-update retries completed successfully: job 2652426 (PixelShuffle)
in 14m20s and job 2652427 (EfficientViT) in 10m38s. PixelShuffle's four-frame
mean total loss at update 200 was 10.502, with RCS MAE 2.854 dBsm and Doppler
MAE 2.119 m/s.

```bash
sbatch jsc_jupiter/PPP_overfit_wide_pixelshuffle_1k.sbatch
```

Submitted continuation: job `2653347`, `PPP_overfit_wide_pixelshuffle_1k`,
reservation `develbooster`, 90-minute allocation. It resumes the exact
`2652426_PPP_overfit_wide_pixelshuffle/checkpoints/epoch_199_step_200.pth`
checkpoint with optimizer, scheduler and RNG state, then applies 800 additional
updates (`--max-updates 800`) for 1,000 total. The config's epoch ceiling is
1100 and its training-loop time budget is 72 minutes, leaving allocation
margin for startup and final output. The saved model/data/loss/optimizer and
schedule contracts were compared against the new config and matched.

Microbatch one, accumulation four, the same four samples, constant LR 1e-5,
existing completed warmup, clipping 0.1, BF16, activation checkpointing,
25-update diagnostics/checkpoints, and memory logging remain in effect.
Each frame will have contributed to 1,000 optimizer updates at the target;
this means 4,000 single-frame training forwards/backwards across the complete
run, excluding diagnostic passes. The source 200-update checkpoint stays in
its original directory; continuation artifacts use their own job directory.

## Joint DiT streams with the shifted DINO head

`configs/PPP_overfit_dit_joint_dino_1k.yaml` selects
`decoder_type: wide_joint_dino_attention`. This reuses the existing
`AttentionPixelShuffleHead` class from `radargen/models/dino_ppp.py`, using
3×3 ICNR-initialized projections, widths 128/64/32, two local-convolution /
window-attention / MLP blocks per stage, four attention heads, 8×8 windows,
and alternating window shifts zero/four. No bilinear interpolation is used.

The DiT returns sample-major `(3B,1152,16,16)` full-width features. The decoder
reshapes them into `(B,3456,16,16)` without mixing different samples or
compressing individual streams. Its stages are:

| Operation | Projection channels | Output after PixelShuffle |
| --- | --- | --- |
| 3×3 projection + shuffle ×4 + attention | 3456→2048 | 128×64×64 |
| 3×3 projection + shuffle ×4 + attention | 128→1024 | 64×256×256 |
| 3×3 projection + shuffle ×2 + attention | 64→128 | 32×512×512 |
| 1×1 readout | 32→3 | 3×512×512 |

The training wrapper returns `(B,3,1,512,512)` in density/RCS/Doppler order.
The head has 66,082,595 parameters. Its readout starts with Gaussian weights
of std 1e-3 and zero bias to keep initial exponential PPP intensities well
scaled. Existing separate-stream variants and DINO checkpoints retain their
architectures. The head starts fresh; its first projection differs from
DINO's 1152-channel input, so this is architectural reuse, not a transfer of
trained DINO head weights. The DiT backbone transfers from the same finetuned
continuous-target job 2516862 used by the other fresh-decoder tests.

`train.extra.shuffle_each_epoch: true` enables the dedicated
`ResumableDistributedShuffleSampler`. A private Torch generator forms a global
permutation using `train.seed + epoch`, before rank sharding. Resume offsets
skip previously consumed rank-local samples from that same permutation.
Accelerate's prepared loader and sampler both receive the explicit epoch,
so resuming mid-epoch or starting a later epoch cannot reset the seed to zero.
The sampler preserves full distributed coverage by padding at most
world_size−1 entries when necessary; this single-GPU four-sample test needs
no padding. The shuffle flag is included in the saved training contract.
Old configs without the flag keep their original sequential sampler.

```bash
sbatch jsc_jupiter/PPP_overfit_dit_joint_dino_1k.sbatch
```

Submitted job: `2656420`, reservation `develbooster`, 1h50m allocation.
It starts a fresh 1,000-update test (not an extension of the bilinear model),
using the same four frames, microbatch one, accumulation four, CAME LR 1e-5,
50-update warmup, clipping 0.1, continuous targets and the existing links.
GPU memory and per-step compute/data-wait timing are logged. The training
loop has a 90-minute budget to leave allocation margin for startup and final
output. It saves checkpoints and all-frame diagnostics every 25 updates.

Validation covered the full `(2,3,512,512)` shape on meta tensors, exact
sample/modality concatenation for batch two, real tiny-DiT forward/backward,
BF16 and equal checkpointed gradients. DINO and old-decoder regressions passed.
Epoch-order tests covered seed+epoch, global rank shards, unchanged training
RNG and bitwise-identical interrupted versus uninterrupted training across an
epoch boundary; final checks also covered the prepared Accelerate loader.


### 1024 joint-head overfit before LR range

The joint head now supports `model.extra.output_scale: 2`: concatenated
3456-channel 16×16 DiT features pass through three ×4 PixelShuffle stages
(64→256→1024 spatial resolution; 128→64→32 channels). Each stage retains
two complete residual/window-attention refinement blocks, including the
1024×1024 stage. The head has 66,304,163 parameters. Continuous point targets
and count-normalized losses use the 1024×1024 output grid.

Job **2657335** uses `configs/PPP_overfit_dit_joint_dino_1024_1k.yaml`:
1000 optimizer updates on the same four frames, batch size 1 and accumulation
4, learning rate 1e-5 with 50 warmup updates, and seed + epoch shuffling.
The finetuned DiT backbone is transferred and the head initialized fresh.
GPU memory and timing logs and diagnostics every 25 updates are enabled.
The one-GPU develbooster allocation is limited to 1h50, with the training
time budget set to 1.5 hours.

The LR-range job **2657324** was canceled at the user's request after 1m13s.
The LR-range configuration remains prepared; it is deferred until the new
overfit results have been reviewed and is not chained automatically.


### Selected 512 decoder: LR range job 2658252

After both 1000-update overfits completed, the 512 head was selected for
efficiency (41m29 versus 1h00m22; adjusted mean total losses 4.8055 versus
4.8416). Job **2658252**, submitted with
`jsc_jupiter/PPP_LR_range_dit_joint_dino_512.sbatch`, uses the full continuous
training split and `configs/PPP_LR_range_dit_joint_dino_512.yaml`.
The four-GPU develbooster allocation first probes batch sizes 8, 4, then 2
per GPU for five complete DDP updates, selecting the first with 15% memory
headroom. A fresh backbone-transfer/head initialization then sweeps
1e-7 to 1e-2 over 600 updates, without accumulation or automatic LR scaling.
The backbone source is the original finetuned continuous-target checkpoint;
overfit weights and probe updates are not carried into the sweep.
Memory, timing, batch selection, and LR-loss plots are written to the job's
scratch work directory. Allocation limit is 1h50; sweep budget is 1.25h.


Job 2658252 was canceled at the user's request; memory probes were removed
from the 512 submission script. Replacement job **2658292** runs the sweep
directly on develbooster with batch **16 per GPU**, four GPUs, effective
batch **64**, accumulation 1. The estimate uses the completed 512 overfit:
10.741 GiB fixed allocation + 4.145 GiB per sample, yielding approximately
77.05 GiB allocated at batch 16 on 95 GiB GPUs. This is an extrapolation,
not a measured batch-16 peak. Memory monitoring remains enabled during the
actual LR sweep; there are no preliminary memory tests. The LR range is
unchanged (600 updates, 1e-7 to 1e-2).

User increased the memory target to approximately 90 GiB. Job 2658292 was canceled and replaced by **2658301** on develbooster, with batch **19/GPU**, effective batch **76**. The same linear estimate gives **89.49 GiB allocated/GPU**; actual DDP/workspace overhead is not measured at this batch. No preliminary memory tests are run; memory logging remains enabled.

At the user's request, batch was increased to **20/GPU** (effective **80**), estimated allocation **93.63 GiB/GPU**. Job 2658301 was canceled and replaced by **2658303**, using the same develbooster LR sweep without preliminary memory tests.

Final batch selection: **19/GPU**, effective batch **76**, estimated allocation **89.49 GiB/GPU**. At the user's request job 2658303 was canceled and replaced by **2658305** on develbooster. The direct 600-update LR sweep and memory logging remain enabled, without preliminary memory tests.


## Final head implementation

The consolidated [final PPP head implementation](final_ppp_head.md) describes
the selected shared head, the DiT bottleneck removal, DINO backbone unfreezing,
patch-artifact changes, output-resolution comparison and epoch-wise shuffling.
It records the final architecture choice alongside the historical experiments above.
