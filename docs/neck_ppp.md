# CNN stem and PPP transformer neck

`NeckPPP` replaces DINO, per-modality LoRA and layer fusion with a trainable
nine-channel CNN stem. Four stride-two convolutions (32, 64, 128, 384 channels)
produce a 32 by 32 grid from the existing aligned 512 by 512 BEV conditions.
The 12-layer RoPE neck remains at stride 16. Projected CNN features at strides
8, 4, 2 and the full-resolution nine-channel input are concatenated into the
four progressive decoder stages and fused with trainable convolutions.
Set `use_skip_connections: false` for the original architecture.
There is no pretrained checkpoint dependency. Dataset normalization and PPP loss weights remain as in the DINO baseline.
Both NeckPPP configs use `mark_link: identity`: RCS and Doppler are predicted
directly in physical units in loss, diagnostic plots and sampling. One mark
value per cell remains unchanged. Existing checkpoints from the architecture
without skips require `use_skip_connections: false` and the original mark link.

`configs/PPP_train_neck.yaml` provides the full-data comparison configuration.
`configs/PPP_overfit_neck.yaml` selects four samples from one scene for an initial
2,000-update diagnostic using `jsc_jupiter/PPP_overfit_neck.sbatch`. This checks
whether the architecture can fit a small set; it does not establish generalization
or a fair quality comparison with a fully trained DINO model. Only the four final prediction PNGs are exported; intermediate diagnostics,
checkpoints and the learning-curve export are disabled. The launcher writes to its own job directory.

Run: `sbatch jsc_jupiter/PPP_overfit_neck.sbatch`.
