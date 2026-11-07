# Final PPP head implementation

The final DiT and DINO PPP variants share the same trainable decoder design.
The two backbones supply different feature widths and spatial grids, but reuse
`AttentionPixelShuffleHead` from `radargen/models/dino_ppp.py`. This section
records the motivation, implementation and experimental basis for that choice;
the earlier decoder experiments remain documented in
[the DiT decoder notes](wide_ppp_decoder.md) and
[the DINO attention notes](dino_attention_ppp.md).

## DiT: remove the latent-channel bottleneck

The original PPP model retained the pretrained autoencoder decoding interface.
Each of the three DiT modality streams produced 1152-channel transformer
features at 16x16 resolution. The final DiT prediction layer reduced these
features to the autoencoder's 32 latent channels, after which the scalar VAE
(DC-AE) decoder reconstructed each output field. This inherited interface
forced all information for each field through a 1152-to-32-channel projection
before spatial decoding.

Removing this bottleneck was the primary motivation for replacing the decoder.
The new path returns the transformer features before the entire original final
prediction layer. It bypasses both that layer and the radar-output VAE decoder.
The frozen autoencoder encoder used for BEV conditioning is retained; removing
the output decoder does not remove conditioning encoding.

For each sample, the three (1152,16,16) streams concatenate along channels into
(3456,16,16). The new joint head predicts all three fields together, rather than
running a shared scalar decoder independently on each stream. Its first learned
3x3 projection maps 3456 channels to 2048 channels before PixelShuffle(4)
rearranges them into 128 channels at 64x64. Channel reduction is therefore part
of learned spatial decoding, without the old 32-channel latent interface.

## DINO: adapt the backbone and predict a finer grid

DINO was pretrained on natural images, whereas this model consumes the three
RGB BEV conditioning maps. We unfreeze the DINO backbone so that its features
can adapt to this input domain rather than treating the pretrained image
representation as fixed. The shared RGB patch embedding and modality-specific
LoRA adapters also train.

The final DINOv3 ViT-S/16 block provides 384 channels per conditioning stream
on a 32x32 grid. Concatenating the three streams gives 1152 channels. The
12-block RoPE neck preserves that width and supplies (B,1152,32,32) to the head.
There is no per-stream fusion compression before the neck in this variant.

For DINO we select 1024x1024 output fields from 512x512 conditioning maps over
the same physical BEV extent. Doubling each spatial dimension allows finer
intensity structure and separates some nearby detections that would otherwise
share one cell and one pair of predicted marks. Over a 100 m extent, the cell
width decreases from approximately 19.5 cm to 9.8 cm. Conditioning images and
continuous detection coordinates are unchanged; the higher resolution is
predicted by the learned decoder, not obtained by resizing its final output.

## Shared head: interleaved upsampling and refinement

Several heads were evaluated in small-sample overfitting experiments. These
included the larger DiT convolutional and EfficientViT decoders and the DINO
attention-head variants. The selected design gave the most useful overfit
results in these experiments and was adopted for both backbones. This is an
empirical architecture choice from memorization tests, not a claim of superior
held-out generalization.

An early attention-head implementation showed visible patch or tiling
artifacts. It used 1x1 pre-shuffle projections and unshifted attention windows.
The selected implementation interleaves learned upsampling with local residual
convolutions and window attention, uses overlapping 3x3 pre-shuffle projections,
and alternates ordinary and shifted windows. Together these changes resolved
the visible patch artifacts in the inspected overfit results. Because the
changes were tested together, their individual contributions are not isolated.

Each of the three stages applies a padded 3x3 convolution, PixelShuffle, and
two complete residual-attention blocks. PixelShuffle rearranges channel groups
into spatial positions; it performs no nearest-neighbor or bilinear
interpolation. The stage output widths are 128, 64 and 32 channels.
Pre-shuffle convolutions produce C_out times the square of the shuffle factor
channels and use ICNR initialization to reduce subpixel phase artifacts.

Each refinement block contains a local convolutional residual block, followed
by LayerNorm and residual multihead self-attention within 8x8 windows, then a
LayerNorm/linear/GELU/linear residual MLP with expansion factor two. Attention
has four heads. The first block uses shift zero, the second a four-cell offset.
Masked padding implements the offset without cyclic wrapping, so opposite BEV
edges cannot attend to each other. Local convolutions and shifted partitions
exchange information across the original window boundaries. The final
upsampling stage retains both full refinement blocks; it is not shortened.
A final 1x1 convolution maps 32 channels to the three output fields.

| Stage | Channels after shuffle | Selected DiT grid | Selected DINO grid |
| --- | ---: | ---: | ---: |
| Head input | 3456 / 1152 | 16x16 | 32x32 |
| 3x3 projection + PixelShuffle(4) + two blocks | 128 | 64x64 | 128x128 |
| 3x3 projection + PixelShuffle(4) + two blocks | 64 | 256x256 | 512x512 |
| 3x3 projection + PixelShuffle(2) + two blocks | 32 | 512x512 | 1024x1024 |
| 1x1 readout | 3 | 512x512 | 1024x1024 |

Thus both selected heads use factors 4/4/2. Their first projection differs in
input width, and their spatial costs differ because the backbone grids differ.
The DiT wrapper is `JointDinoAttentionPPPDecoder` with
`decoder_type: wide_joint_dino_attention` and `output_scale: 1`. The DINO
shifted variant selects `attention_projection_kernel_size: 3` and
`attention_shifted_windows: true`; its `output_scale: 2` retains the third
stage. These scale settings belong to different configuration interfaces and
should not be interpreted interchangeably.

## Why DiT remains at 512 rather than 1024

We also tested a DiT head with three factor-four stages: 16 -> 64 -> 256 ->
1024. It kept the same stage widths and all refinement blocks, including at
1024. The 512 head has 66,082,595 parameters; the 1024 head has 66,304,163.
The main added expense comes from spatial computation, not parameter count.

Both overfit runs completed 1000 optimizer updates on the same four frames,
with batch one and accumulation four. Run 2656420 (512) took 41m29s; run
2657335 (1024) took 1h00m22s. The 1024 run took approximately 45.5% longer.
At the same fixed time budget, that corresponds to approximately 31.3% fewer
updates or epochs, assuming similar relative throughput. This is an estimate
from the complete overfit jobs, including startup and diagnostics, not a
measured production-throughput ratio.

At update 1000, mean resolution-adjusted total losses were 4.8055 (512) and
4.8416 (1024). The 1024 head improved adjusted PPP likelihood and RCS error,
while the 512 head had lower Doppler and point-count error. There was therefore
no clear overall improvement sufficient to justify the extra cost, and the
512 DiT head was selected for subsequent LR tuning and training.

The resolution adjustment removes the per-cell likelihood convention's
log(4) offset from the 1024 PPP loss, and consequently from the summed total.
Cell mass is exp(log_intensity)/(H*W); equivalent spatial densities on a grid
with four times as many cells have a normalized per-cell NLL larger by log(4)
for nonempty frames. This adjustment is for comparing resolutions only; it
does not modify the training objective or the mark losses. Discretization
and cell collisions still differ between resolutions.

## Continuous targets and epoch-wise shuffling

Both selected variants use continuous detection-wise PPP, RCS and signed
Doppler losses. Every in-bounds detection contributes separately, including
multiple detections in one cell. The loss gathers predictions at each
continuous detection's output-cell location; targets are not deduplicated or
averaged into training maps. Each cell nevertheless predicts a shared mark
pair, so collisions can impose irreducible mark error. The continuous-target
choice is independent of the decoder architecture.

A noticeable periodic cycle in the training loss motivated changing the
sample order between epochs. Repeating the same ordered traversal can repeat
sample-dependent loss patterns. Both final variants therefore enable
`train.extra.shuffle_each_epoch: true`: the resumable distributed sampler
constructs a global permutation using `train.seed + epoch`, with zero-based
epoch indices, and shards it across ranks. The sampler and Accelerate loader
both receive the explicit epoch. Within-epoch resume reconstructs that
permutation and skips consumed rank-local samples. This changes ordering
reproducibly; it does not guarantee that all periodic loss structure disappears.

The selected DiT production configuration is
`configs/PPP_train_dit_joint_dino_512_constant_1e5.yaml`: constant LR 1e-5,
no warmup or decay, batch 19 per GPU on four GPUs (effective batch 76), and
checkpoints every 1000 optimizer updates with all checkpoints retained.
The Slurm allocation is 12 hours and the loop budget is 11.8 hours to leave
time for final saving. These are DiT run settings; the DINO optimizer and
schedule remain documented separately.
