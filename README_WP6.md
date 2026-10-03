# WP6 implementation candidate

Copy the files into their matching paths in RadarGen. Existing training files
are reused, not replaced. Inspect `git status` first. If any destination already
exists, compare it before copying; these files were prepared from the uploaded
WP5 sources, not from a live JUPITER checkout.

Run from the repository root:

```bash
source jsc_jupiter/activate_jupiter.sh
python tests/test_ppp_inference.py
sbatch jsc_jupiter/wp6_inference_smoke.sbatch
```

Run the GPU job only after CPU tests pass. The job reads the WP5 step-6 checkpoint
and one sample from the explicitly configured training scene. It does not train,
modify checkpoints, or evaluate model quality.

## Behavior

The pipeline returns three raw fields from one DiT invocation. It uses the
existing frozen encoder, differentiable decoder wrapper, and live Gemma
empty-prompt provider in eval/no-grad mode. The SANA checkpoint used for model
construction is explicitly separate from the trained PPP wrapper checkpoint.
The existing strict PPP loader restores the latter without an optimizer.

Sampling uses independent Poisson counts with mass exp(f)/(H*W), metric pixel
centers with flipped rows, and cubic RCS/Doppler links. Multiplicity is preserved.
Sampling runs on CPU using a private seeded generator; identical seeds and sample
order reproduce sampling in the same environment. It does not guarantee identical
model predictions across hardware/software versions. No global RNG is reseeded.

HDF5 access reuses the validated training dataset and its transform. Consequently
this initial inference path also reads/rasterizes the stored targets and inherits
the dataset's explicit refusal of empty ground-truth frames. Targets are NOT fed
to the model. Supporting conditioning-only HDF5 files or empty-ground-truth frames
would require a separate loader; neither is silently introduced here. Synthetic
empty OUTPUT point clouds are supported.

No field caching is performed; sample identities are returned and printed.
No generated-count cap or mark clipping is imposed. Excessively large finite
intensities can require excessive memory; the current smoke checkpoint is for a
structural check only.

## Verification status

Python compilation and shell syntax checked in the preparation environment.
PyTorch is not installed there, so the included CPU tests have NOT been executed.
Actual checkpoint loading and GPU inference remain pending on JUPITER. The tests
cover normalization, inverse geometry, marks, multiplicity, zero output, seeded
sampling, invalid fields, and strict model-only loading of a substitute checkpoint.
They do not establish full pretrained runtime compatibility.

Append the accompanying status snippet to the roadmap without replacing the
roadmap with an older uploaded copy. Update pending checks only after running them.
