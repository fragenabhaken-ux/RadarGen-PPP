# Image2Radar v2

One-shot raw-camera prediction of log-intensity, raw RCS, and signed Doppler.
The original `cam2radar` model is retained.

- Shared DINOv3-S with frozen base weights and shared rank-16 LoRA.
- Final DINO patch grid, projected through one per-token residual MLP.
- Learned 64×64 BEV queries; two pre-norm transformer blocks at 64×64 and
  two at 128×128, alternating regular and shifted local self-attention windows.
- Fresh gathered angular cross attention at 256×256, then residual convolution.
- PixelShuffle upsampling with widths 256 → 128 → 64 → 32; final 512×512
  linear three-channel output. No diffusion, timestep, or output sigmoid.

`Image2RadarV2.forward(camera_images, camera_intrinsics, reference_to_camera)`
uses the same four-camera ordering and calibration contract as `Camera2Radar`.
Images are uint8 RGB `(B,4,3,H,W)`. Intrinsics refer to original, undistorted
pixels. Bottom/right padding preserves their origin; partial patch rays use the
center of their valid footprint. The BEV spans ±`coordinate_range` metres,
with x increasing across columns and y decreasing across rows.

Cross attention computes camera-relative range for queries, elevation for
image keys, and a learned per-head relative azimuth bias. It sorts rays and
uses periodic angular search, gathering up to `max_keys` per camera and query,
then masks rays outside `angular_radius_degrees`. This cap approximates the
full strip: when more rays qualify, some are omitted. Tune the cap and radius
together; no dense BEV-by-image attention matrix is constructed. Queries
without matching rays get exactly zero cross-attention updates. This models
angular coverage, not physical visibility or occlusion.

`query_chunk_size` bounds temporary attention tensors. Checkpointing decoder
blocks and the fine cross-attention stage reduces saved training activations,
with recomputation cost. A dedicated GPU memory/throughput probe is required
before selecting a production batch size. The config starts at batch size 1.

## Parameter budget

The default model has **7,475,047 trainable parameters** and **29,061,991 total
parameters**, measured with the real DINOv3-S backbone. Shared cameras do not
multiply the parameter count. Full-resolution camera GPU memory and throughput
have not yet been measured.

## Training

Activate the repository environment, then use the existing trainer:

```bash
python scripts/train_ppp.py --config_path configs/Image2RadarV2_train.yaml
```

The config reuses existing manifest, checkpoint, storage, and continuous PPP
loss conventions. These assets must exist locally. Model construction checks
that model and target coordinate ranges agree. Training, overfit diagnostics,
checkpoint restoration, and validation recognize `Image2RadarV2` as a raw-camera
model. No cluster job is submitted automatically.

## Checks

```bash
python -m pytest tests/test_image2radar_v2.py -q
```

Tests use an injected tiny encoder, requiring no pretrained checkpoint. See
[design.tex](design.tex) for design rationale, references, and open experiments.
