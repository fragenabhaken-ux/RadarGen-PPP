# DiTV2

DiTV2 predicts log intensity, RCS (dBsm) and signed Doppler (m/s) in one
forward pass from the three existing RGB BEV condition maps. It uses no
noise, diffusion timestep, text encoder or pretrained autoencoder.

Each condition has a trainable three-stage stride-two encoder. The three
64-channel feature maps are concatenated and projected to width 1152.
At 512 pixels, the latent raster is 64 x 64 (factor eight). Three learned
output modality embeddings create the density/RCS/Doppler streams. The
28 transformer blocks use joint ReLU linear attention across all streams,
learned constant residual gates, fixed spatial sine/cosine positions and
per-stream convolutional feed-forward layers. A shared trainable decoder
upsamples each predicted latent field by eight. All modules train jointly.

RCS and Doppler use identity links and physical-unit L1 losses. Intensity
uses the existing count-normalized PPP likelihood and continuous targets.
Condition dropout is zero. All three loss weights remain one. Existing
models retain cubic links by default; DiTV2 configs explicitly select
`train.extra.mark_link: identity`, including for inference and diagnostics.

## Run

Submit from the repository root; GPU training runs only inside Slurm:

```bash
sbatch jsc_jupiter/PPP_overfit_ditv2.sbatch
```

The launcher follows `jsc_jupiter/wp8_overfit.sbatch`: one Booster GPU,
72 CPUs, 30-minute allocation, 200 optimizer updates, batch size one and
four-microbatch accumulation. It repeats indices 0, 32, 64 and 96 from
training scene `018a60086f5e441fb09f476e55948b72`. The scene/subset manifest
is derived from the checked full manifest on the compute node; the source
manifest remains unchanged. DiTV2 starts from fresh weights, with AdamW
LR 1e-4, 20-update warmup, no weight decay and gradient clipping at one.
Diagnostics at step zero and every 25 updates contain losses, expected
count, mark MAEs and plots. Outputs go to the job-specific scratch directory
`/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/<jobid>_DiTV2_overfit`.
The launcher also runs the existing summary script after training.

This diagnostic measures memorization, not validation quality. Colliding
detections can have different marks and therefore impose an irreducible
error on the shared per-cell mark predictions.

The full training config is a starting configuration, not a tuned schedule.
The old DiT checkpoints are architecturally incompatible with DiTV2.

## Continue the four-frame test to 2,000 updates

`sbatch jsc_jupiter/PPP_overfit_ditv2_2k.sbatch` resumes job 2570275's
step-200 checkpoint for 1,800 additional updates, preserving optimizer,
scheduler and RNG state. The constant LR remains unchanged. The epoch
limit is raised to 2,000. Walltime is one hour based on the measured
first-run throughput.

`--final-pngs-only` suppresses periodic diagnostics, intermediate
checkpoints, NPZ dumps and TensorBoard. Four final PNGs are written under
`prediction_images/` in the new job's scratch directory, alongside one
final checkpoint and the configuration. Slurm logs retain scalar progress.

## LR range test

Submit `sbatch jsc_jupiter/PPP_LR_range_ditv2.sbatch` for fresh DiTV2
initialization on the full training split, batch 20 per GPU on four GPUs
(global batch 80), no accumulation, and 400 optimizer updates. AdamW LR
increases from 1e-8 to 1 with gamma `(1 / 1e-8) ** (1 / 399)`, without
warmup or automatic LR scaling. The existing exponential scheduler steps
once per applied optimizer update; the logged LR is captured before that
step and corresponds to the accompanying loss.

Every update logs total PPP training loss, component losses, gradient norm
and actual LR to the job's Slurm `.err` file. The launcher automatically
plots those records to `<job-work-dir>/lr_range/lr_loss.png`, with raw and
EWMA-smoothed curves and a logarithmic LR axis. It attempts plotting even
if high LR causes a numerical error. No checkpoints, prediction images or
TensorBoard files are written. GPU utilization/memory are recorded every
five seconds in `<job-work-dir>/gpu_usage.csv`. Walltime is one hour, with a 54-minute training budget to allow orderly plotting if the update limit is not reached.
