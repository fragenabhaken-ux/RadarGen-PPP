# Direct DINOv3 BEV predictor

`radargen/models/dino_ppp.py` contains the complete prediction network as ordinary
PyTorch modules. It uses timm for the pretrained DINOv3 backbone and does not
depend on MMSegmentation. The template files remain reference implementations.

The three inputs are the dataset's normalized appearance, semantics and velocity
maps, each `(B, 3, H, W)`. They share the same BEV coordinates and pass separately
through one shared RGB DINO backbone, with a distinct LoRA adapter for each
conditioning modality. The network returns `(B, 3, H, W)` with channels:

1. Log-intensity, consumed directly by `PPPLoss`.
2. Raw RCS, cubed inside the existing masked RCS loss.
3. Raw Doppler, cubed inside the existing masked Doppler loss.

The predictor has no target autoencoder, latent-space processing, DiT, timestep,
text conditioning or modality indicators. Predicting the fields is deterministic
in evaluation mode; sampling a point cloud from those fields still uses the
existing PPP sampler and its random generator.

## Network

For a 512 × 512 input, the original RGB 16 × 16 patch projection creates a
32 × 32 feature grid for each condition. The inputs are stacked into a `3B` batch
for shared backbone processing; they are not concatenated into nine input
channels. Each modality uses its own LoRA parameters while sharing the original
DINO weights, including the frozen RGB patch projection.

At each selected DINO block (2, 5, 8 and 11, zero-based), the three modalities'
384-channel features are concatenated into `(B, 1152, 32, 32)` and projected to
128 channels. Concatenating these four layers gives 512 channels, which are fused
to 384. A 12-layer transformer neck with 2D rotary positions refines this grid.
The default `decoder: progressive` reconstructs the image through four successive
2× pixel-shuffle stages, initialized with ICNR. Their feature widths are 128, 64,
32 and 16 channels, at spatial sizes 64, 128, 256 and 512 respectively. The final
projection produces the three raw PPP fields.

The original RGB DINO checkpoint loads without changing its input convolution.
With `frozen_base: true`, the pretrained weights and RGB stem stay frozen; the
three modality-specific LoRA adapters, feature projections, fusion, neck and
decoder train. These separate adapters do not add modality-indicator channels
or tokens.

512 is divisible by 16, so the normal training path needs no padding or resize.
The module also accepts rectangular inputs: it pads incomplete patches on the
bottom and right, then crops predictions to the original size before returning.
The current HDF5 training dataset remains fixed at 512 × 512.

## Training

Use `configs/PPP_train_dinov3.yaml`. It retains the existing HDF5 manifest, loss
weights and condition dropout policy, and uses a standard PyTorch AdamW optimizer.
The only pretrained asset is the local DINOv3 checkpoint in
`model.extra.checkpoint_path`. VAE and text configuration sections are unnecessary.
The existing fresh-training launcher command remains the same.

```bash
python scripts/train_ppp.py --config_path configs/PPP_train_dinov3.yaml --validate-only

# Optional one-GPU smoke test: 200 updates, then 5 resumed updates.
sbatch jsc_jupiter/PPP_train_dinov3_smoke.sbatch

# Four-GPU LR range test: batch 128/GPU, accumulation 1, 2000 updates.
sbatch jsc_jupiter/PPP_LR_range_dinov3.sbatch

# Then run one full epoch on four GPUs with production training settings.
sbatch --job-name=PPP_dinov3_1epoch jsc_jupiter/PPP_train_dinov3.sbatch \
  configs/PPP_train_dinov3_1epoch.yaml

# Submit the supplied four-GPU launcher when ready:
sbatch jsc_jupiter/PPP_train_dinov3.sbatch
```

The smoke config keeps the complete model and full training manifest, uses batch
size 128 on one GPU with accumulation 1, and logs every update. Job 2361512
completed all 205 updates at this batch size, with peak allocated memory of
46.4 GiB and peak reserved memory of 50.4 GiB on its single GPU. Each batch
contains `3B` RGB backbone inputs. Its five-update warmup lets the short run reach
the configured learning rate. Condition dropout is disabled so all three inputs are exercised.
Predictions use the same three samples (indices
0, 100 and 1000) at updates 0, 50, 100, 150, 200 and 205. Each sample produces
the existing PPP plot with conditioning, ground truth, predicted fields and
sampled points. Empty target samples are logged and skipped. Checkpoints
are saved every 50 updates and at each phase's end. The launcher checks
parameter changes and checkpoint restoration, then starts a separate resumed
process. This checks execution and numerical stability; it is too short to judge
prediction quality.

Each job writes to
`/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<job-id>_<job-name>/`, including
`checkpoints/` and `prediction_images/`. The smoke job should finish with
`PPP smoke parameter updates and checkpoint restoration passed` twice, and its
last checkpoint should record global step 205. Configs and launcher syntax can
be checked without submitting a job; GPU memory and runtime are established by
the smoke run itself.

Each DINO training update also logs `gpu_allocated_gib`, `gpu_reserved_gib`,
`gpu_peak_allocated_gib` and `gpu_peak_reserved_gib` on rank 0. Peak counters cover
the complete optimizer update, including accumulated microbatches, backward and
the optimizer step. These are PyTorch allocator measurements; the GPU's total
memory use also includes CUDA context and allocations outside PyTorch.

LLview's job-overview GPU average includes all GPUs on the job's nodes in its
standard aggregation. One GPU at 100% and three idle GPUs therefore average
25%; this does not indicate that the active GPU itself was only 25% utilized.
The overview uses Active SM when available, otherwise GPU Utilization. See the
[overview documentation](https://apps.fz-juelich.de/jsc/llview/docu/jobreport/overview_graph/)
and [node-based aggregation](https://github.com/FZJ-JSC/llview/blob/main/configs/server/LLgenDB/conf_gpu/gpu.yaml#L220-L237).

### LR range test

`configs/PPP_LR_range_dinov3.yaml` uses the same model, batch size 128 per GPU and
accumulation 1 on four GPUs, giving an effective batch size of 512. It raises
AdamW's actual learning rate exponentially
from `1e-6` on update 1 to `1e-3` on update 2000, without warmup or automatic
batch-size scaling. Thirty epochs are allowed so the full manifest can supply all
2000 updates; the configured update limit stops the run. Loss weights remain
one, and conditioning dropout is disabled so all three inputs are present.

The launcher starts from the original RGB DINO checkpoint with fresh LoRA,
fusion, neck and decoder weights. This is a disposable sweep: it saves no
training checkpoints and has no resume phase. It records loss, the LR used,
gradient norm and GPU memory every update. Three fixed prediction examples
(indices 0, 100 and 1000) are rendered initially and every 200 updates.

Submit with `sbatch jsc_jupiter/PPP_LR_range_dinov3.sbatch`. The launcher requests
four GPUs for up to six hours and creates `lr_range/lr_loss.png` in the job's work
directory using the existing plotting script. If training returns with a
numerical error at the high-LR end, it also attempts to plot the partial log
while retaining the training failure as the job exit status. Finite rising
losses do not trigger automatic early stopping.

### Training checkpoints

Resume with `--resume /path/to/checkpoints/latest.pth`, or use `--init-from` to
load weights with fresh optimizer/scheduler state. These must be DINO PPP
checkpoints from this shared-RGB, per-modality-LoRA architecture with the same
decoder configuration. Earlier nine-channel DINO PPP checkpoints and the previous
DiT/autoencoder model have incompatible parameters; neither can initialize or
resume this architecture through those options. The original RGB DINO pretrained
checkpoint remains valid for fresh training.
Full checkpoint restoration does not reload the original RGB DINO checkpoint.

## Inference

The same training config defines the architecture. Load a trained checkpoint:

```python
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline

pipeline = RadarGenPPPPipeline.from_config(
    config, checkpoint_path="/path/to/dino_ppp_checkpoint.pth", device="cuda"
)
fields = pipeline.predict_fields(bev_color_map, bev_seg_map, bev_velocity_map)
# fields: log_intensity, rcs_raw, doppler_raw; each (H, W).
```

Inputs here are already transformed by the dataset. The evaluation and validation
PPP wrappers also accept this architecture without SANA, null-embedding or text
model paths. Existing latent PPP configs retain their original path.

## Focused CPU tests

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_dino_ppp*.py'
```

Tests exercise spatial reconstruction and gradients, RGB feature extraction,
loss links, condition dropout, deterministic prediction, training/checkpoint
resume, and inference without latent/text assets. Substitute backbones keep the
suite small; pretrained weights are not downloaded.
