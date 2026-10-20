# TensorBoard on Jupiter

PPP training (including DINOv3) and shared PPP/RadarGen validation can write
TensorBoard events alongside the existing files. File output remains enabled.
Checkpoints and raw NPZ data retain their existing formats.

Add this root-level setting to a PPP training YAML:

```yaml
tensorboard: true
```

Alternatively pass `--tensorboard true` to `scripts/train_ppp.py`. The default
is false. This setting is independent of `report_to`; leave `report_to: none`
if you only want the new TensorBoard output and existing files.

For validation, add `tensorboard: true` to the evaluation YAML passed with
`--config` (for example `evaluation/configs/PPP_validation.yaml`), rather than
the model architecture YAML. Keep this setting unchanged when resuming validation.

Events are written to `<work_dir>/tensorboard` during training and
`<output_dir>/tensorboard` during validation. Training includes losses, learning
rate, gradient norm and available GPU memory metrics at `train.log_interval`.
Prediction/overfit figures and diagnostics follow the existing configured
intervals. Validation includes per-sample metrics and prediction figures, plus
aggregate summary metrics. Its horizontal axis is the selected sample index;
training uses optimizer update count. Validation resumes append new samples;
training resumes purge event steps after the restored checkpoint.

On the Jupiter host used for viewing:

```bash
# All RadarGen scratch runs, port 12356:
bash jsc_jupiter/tensorboard.sh
# A specific run and port:
bash jsc_jupiter/tensorboard.sh /e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/RUN 12356
```

The launcher uses the JSC TensorBoard module and binds to localhost. Forward
the port from your workstation to the host running TensorBoard using your
usual Jupiter SSH configuration, then open `http://localhost:12356`:

```bash
ssh -N -L 12356:localhost:12356 YOUR_JUPITER_HOST
```

TensorBoard dependencies are already listed in the repository requirements.
The legacy diffusion training script `scripts/train.py` continues to use its
existing `report_to` tracker configuration.
