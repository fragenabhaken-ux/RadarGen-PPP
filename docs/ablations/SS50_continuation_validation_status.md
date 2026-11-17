# Continuation SS50 with PNG output — 2026-11-13

User requested entry-node SS50 for the two completed second allocations and separate all-metric comparisons against final DINO, best-F1 DINO and RadarGen.

| Model | Completed training job | Final checkpoint | SS50 config |
|---|---:|---|---|
| DiT without bottleneck | 2867524 | epoch_44_step_58057.pth | evaluation/configs/PPP_validation_ss50_2867524_step58057_png.yaml |
| Compact DINO | 2867528 | epoch_94_step_20595.pth | evaluation/configs/PPP_validation_ss50_2867528_step20595_png.yaml |

Both training jobs completed with exit 0. Validations run serially on the entry node, with `save_plots: true`, TensorBoard disabled so PNGs are written directly, plot DPI 110, seed 42 and the fixed reference sample order.

At launch, other users occupied approximately 86 GiB of the entry GPU. The validation launcher waits for at least 50 GiB free before each evaluation rather than interrupting their processes or risking immediate OOM. **Waiting is not completed validation.**

- Launcher and status: `output/ablations/ss50_dit_compact_resume/run_entry.sh`, `launcher.log`, `<JOB>.log`, `<JOB>.exit_code`.
- DiT results/PNGs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2867524_step58057_png/plots/`.
- Compact-DINO results/PNGs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2867528_step20595_png/plots/`.
- Comparison worker: `output/ablations/ss50_dit_compact_resume/compare_results.py`, log `comparison.log`. It waits for exit codes, verifies 50 matching sample records and 50 PNGs, then writes separate reports.
- Expected reports: `docs/ablations/SS50_dit_dcae_no_bottleneck_2867524_vs_dino_radargen.md` and `docs/ablations/SS50_dino_neck_compact_2867528_vs_dino_radargen.md`.
- References: final DINO **2773475 / 19,914**, best-F1 DINO **2773475 / 15,000**, RadarGen validation **2404702**.

Verify exit codes and report files before reporting scientific results. No Slurm allocation submitted.


## Superseded by full-validation jobs

User cancelled entry-node SS50 in favor of bundled full validation and all-scene export jobs. Entry waiters and comparison process terminated before inference. See `full_validation_scene_exports.md`: jobs **2887635** and **2887636**, develbooster, two hours each, including PNGs. No SS50 result should be inferred from this abandoned setup.
