# DINOv3 attention PPP variant

`configs/PPP_train_dinov3_attention_1024.yaml` keeps 512x512 conditioning
and predicts three 1024x1024 PPP fields over the same physical BEV extent.
The shared DINOv3 ViT-S/16, RGB patch embedding, and modality-specific LoRA
adapters all train. Only the final DINO block (index 11) is selected. Its three
384-channel modality features concatenate directly into (B,1152,32,32), without
per-layer projections or fusion compression. All 12 RoPE neck blocks retain
1152 channels, with 9 attention heads of 128 channels. Q/K/V each have width
1152; the internal MLP expands to 4608 and returns to 1152 on the residual path.
The neck output normalization preserves width; the decoder receives 1152 channels.
`feature_fusion: last_concat` requires the last block and a neck width equal to
three backbone widths. The historical `multilayer` fusion remains the default
for existing configurations and checkpoints.

The new head uses 1x1 projections and ICNR-initialized pixel shuffle:
32 -> 128 (4x, 128 channels) -> 512 (4x, 64 channels) -> 1024 (2x, 32 channels).
Each stage has two residual attention blocks: local residual convolutions,
8x8 window self-attention and a residual MLP. Attention processes at most 4096
windows per call to avoid CUDA SDPA batch-grid limits at 1024 resolution
(batch 4 already contains 65536 windows). Local convolutions exchange
information across window boundaries. Activation checkpointing covers decoder
layers as well as the backbone and neck when enabled. `output_scale: 1` omits
the last stage and produces 512 fields, useful for resolution ablations.

Continuous detection lists remain the default. Nothing deduplicates, averages,
or rasterizes targets before the loss. Every detection, including coincident
ones with different marks, is gathered separately at the output grid resolution.
One cell still predicts one shared RCS/Doppler pair; multiple observations in
that cell each contribute to its mark loss. The PPP sampler retains Poisson
multiplicity per cell.

The loss reference area is the output cell count, matching sampling and count
diagnostics: cell mass = exp(log_intensity)/(H*W). Thus increasing resolution
alone does not multiply expected point count. The existing per-cell observation
likelihood convention is preserved: normalized NLL for identical constant fields
at 1024 differs from 512 by log(4) for nonempty frames. Do not directly interpret
raw NLL differences across resolutions as improvements; physical point-cloud
metrics and expected-count errors remain comparable.

Inference keeps separate input/output sizes. Evaluation already derives its
sampling, analytical counts and per-detection mark lookup from predicted field
shapes. No downsampling is needed for point-cloud comparisons. Conditioning
HDF5 files, their identity signatures and continuous radar coordinates remain
unchanged.

## Initialization and launch

The launcher transfers only the backbone (including modality adapters) from the
finetuned `2391378` checkpoint. The old 384-channel neck cannot initialize the new
1152-channel neck: the entire neck and decoder start fresh. Old fusion projections
are discarded. Transfer checks every backbone key and shape before loading, and restores
no optimizer, progress or RNG. Ordinary `--init-from` and `--resume` remain strict.
The launcher accepts an alternate config as argument 1 and source checkpoint as
argument 2. This source identifies the historical finetuned run, not a newly
verified best-scoring checkpoint.

```bash
python scripts/train_ppp.py --config_path configs/PPP_train_dinov3_attention_1024.yaml --validate-only
sbatch jsc_jupiter/PPP_train_dinov3_attention_1024.sbatch
```

The initial settings are conservative: batch 2/GPU, accumulation 8, four GPUs
(effective batch 64), AdamW 1e-5, 65 epochs, 200 warmup updates, cosine to 1e-6.
The single LR also applies to the fresh head; it is a starting experiment setting.
GPU memory, throughput and useful learning rates require a GPU smoke run before
production tuning. No training job is submitted by implementing this variant.

For validation, copy an existing DINO PPP validation config and point both
`model_config` and `dataset_config` to the new config, and `checkpoint_path` to
the new trained checkpoint. Keep the established frame selection and bounds.

## Four-sample overfit and memory measurement

`configs/PPP_overfit_dinov3_attention_1024.yaml` selects frames 0, 32, 64 and 96
of scene `018a60086f5e441fb09f476e55948b72`, with batch 4, no accumulation,
1000 updates, AdamW 1e-4, and 20 warmup updates. Continuous detection targets
and checkpointed full-backbone training remain enabled. Diagnostics run before
training and every 100 updates, with per-sample losses, counts, mark errors,
PNGs and detection-preserving NPZ files. Keep the same checkpointing and precision
settings when interpreting these memory measurements for production training.

```bash
sbatch jsc_jupiter/PPP_overfit_dinov3_attention_1024.sbatch
```

On this cluster develbooster is a reservation, not a partition. The launcher uses
`--partition=booster --reservation=develbooster`, with a 1h50m limit. It logs
PyTorch allocated/reserved memory and per-update peaks (including optimizer
steps), train/data-wait times, GPU model/capacity in `gpu_inventory.csv`, and
5-second driver memory samples in `gpu_memory.csv`. Slurm logs contain the
training measurements; standalone overfit artifacts go to `overfit_diagnostics/`.
After successful training, `overfit_summary.txt` and `learning_curves.png`
summarize the diagnostic trend. `--smoke-check` verifies parameter updates and
final checkpoint restoration. A successful run establishes batch 4 as feasible;
larger batch limits require measurement rather than linear extrapolation alone.

## Overlapping projections and shifted-window comparison

`configs/PPP_overfit_dinov3_attention_1024_shifted.yaml` changes only the decoder
architecture relative to the original four-sample experiment:

- `attention_projection_kernel_size: 3` uses padded 3x3 convolutions before
  each pixel shuffle; ICNR initialization and 4x/4x/2x factors remain.
- `attention_shifted_windows: true` alternates ordinary 8x8 windows with windows
  offset by four positions. Masked top/left padding implements the shifted
  partition without cyclic wrapping, so opposite physical BEV edges never
  become neighbors. Invalid padding keys are excluded, and output shape is
  restored after each block.

The 1152-channel neck, pretrained DINO/adapters, seed, four frames, batch 4,
BF16, checkpointing, optimizer schedule, continuous targets and 1000-update
budget match the baseline run `2652265`. Neck and decoder start fresh again;
this is not a continuation of the baseline's trained neck/head. Existing configs
retain 1x1 projections and unshifted windows for checkpoint compatibility.

```bash
sbatch jsc_jupiter/PPP_overfit_dinov3_attention_1024_shifted.sbatch
```

Artifacts and memory measurements use a separate job directory. Compare the
same samples/steps, especially 400 and 1000: per-detection mark errors, expected
counts, intensity mass at detection cells, and discontinuities at multiples of
32 output pixels. Less visible tiling alone is not evidence of better localization.
`configs/PPP_train_dinov3_attention_1024_shifted.yaml` provides the corresponding
full-dataset config for use with the existing training launcher's config argument.

## LR range test at estimated production batch size

The shifted decoder's batch-4 overfit run measured 13.78 GiB peak allocated,
2.80 GiB allocated at step start and 95 GiB GPU capacity. A rough conservative
model is `2.80 + (batch/4)*(13.78-2.80)` GiB, giving about 69 GiB at batch 24.
This estimate includes batch-independent workspaces in its scaled term and is
not an optimum-throughput measurement. Batch 32 would approach the device limit.

`jsc_jupiter/PPP_LR_range_dinov3_attention_1024_shifted.sbatch` first runs five
full DDP optimizer updates on four GPUs at batch 24/GPU, no accumulation, using
the full training manifest. It records measured peaks in `batch_probe_summary.json`.
A failed probe stops the launcher. A successful probe starts a separate process
with freshly initialized neck/head and the same finetuned DINO/adapters, so probe
updates never bias the LR sweep.

The sweep uses `configs/PPP_LR_range_dinov3_attention_1024_shifted.yaml`:
600 optimizer updates, effective batch 96, actual LR exponentially increasing
from 1e-7 to 1e-2, no warmup or automatic scaling, and the established continuous
loss/gradient clipping. Each update logs LR, all loss components, gradient norm
and rank-0 GPU memory; driver samples cover all GPUs. Three fixed full-dataset
examples are rendered every 200 updates. No model checkpoints are retained.
The job uses the develbooster reservation with a 1h50m limit.

```bash
sbatch jsc_jupiter/PPP_LR_range_dinov3_attention_1024_shifted.sbatch
```

The job directory contains `batch_probe.err`, `batch_probe_summary.json`,
`lr_range.err`, `sweep/gpu_memory.jsonl`, and the final `lr_range/lr_loss.png`.
The plot is also attempted if training fails numerically at high LR; the original
training failure status is preserved. A useful LR is selected from the measured
stable decreasing-loss region, not automatically from the minimum noisy loss.

Measured batch verification: job `2653542_PPP_LR_dino_shifted_b24` completed
all five batch-24/GPU DDP updates. Rank 0 peaked at 69.75 GiB allocated and
72.56 GiB reserved, with approximately 20.47 GiB driver-reported free memory.
This validates batch 24 as a working full-data distributed setting; throughput
optimality and the absolute maximum batch size were not measured.

## Constant-LR 12-hour production run

`configs/PPP_train_dinov3_attention_1024_shifted_constant_1e4.yaml` uses the
validated batch 24/GPU on four GPUs (effective 96), with no accumulation.
The full DINO backbone and modality-specific LoRA adapters train, initialized
from the same finetuned `2391378` checkpoint; neck and decoder start fresh.
Continuous detection lists drive ContinuousPPPLoss, retaining every detection
including collisions. Both mark links remain cubic.

The schedule warms up linearly for 100 optimizer updates, then holds AdamW LR
at 1e-4 without decay. Gradient clipping remains 1.0. Numbered checkpoints
are written every 1000 updates; epoch saves are disabled and
`keep_last_checkpoints: 0` retains all checkpoints. Predictions are logged
every 1000 updates to TensorBoard. The 65-epoch ceiling exceeds a single job's
budget so it does not end training early; the Slurm limit is 12 hours on the
normal booster partition. On a hard wall-time stop, resume from the latest
completed checkpoint. Periodic checkpoints contain optimizer/scheduler,
progress and per-rank RNG state.

```bash
sbatch jsc_jupiter/PPP_train_dinov3_attention_1024_shifted_constant_1e4.sbatch
```

The launcher snapshots the submitted config and GPU inventory in its job-specific
work directory. This production run uses neither the development reservation
nor weights produced by the LR-range test.

### Epoch-wise deterministic shuffle

The constant-LR production config sets `train.extra.shuffle_each_epoch: true`.
The old ranged sampler only stored the epoch number and returned consecutive
indices; it did not shuffle. The new resumable distributed sampler generates
one global `torch.randperm` using `train.seed + epoch` (zero-based), then shards
it across ranks. Standard distributed padding repeats at most world_size-1
entries so all ranks execute equal step counts. For 83,554 frames and four GPUs,
this yields 20,889 samples/rank, two padded entries globally, and 871 batches.

Both the sampler and Accelerate loader receive the explicit epoch, preventing
its iterator counter from overriding the epoch after resume. Within-epoch resume
reconstructs the same permutation and skips consumed rank-local samples.
The shuffle option is part of the strict checkpoint contract, so older ordered
runs cannot accidentally resume under changed ordering. Legacy configs retain
the ranged sampler unless they explicitly enable shuffling. The restarted DINO
run starts fresh from the same historical DINO/LoRA source, not from the stopped
production run; its artifacts are kept in a separate job directory.


## Final head implementation

The consolidated [final PPP head implementation](final_ppp_head.md) describes
the selected shared head, the DiT bottleneck removal, DINO backbone unfreezing,
patch-artifact changes, output-resolution comparison and epoch-wise shuffling.
It records the final architecture choice alongside the historical experiments above.
