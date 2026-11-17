# DiT comparison with RadarGen and best-F1 DINO

Scene `9d1180d52b6a46328063042b8fb325de`, entry-node workflow. New DiT without-bottleneck checkpoint **2849053 / 28,537**. Reuse existing RadarGen and **2773475 / 15,000**, the best SS50-F1 DINO checkpoint (not final step 19,914), predictions.

- Config: `evaluation/configs/scene_comparison_traffic3_ablation_dit_best_dino.yaml`.
- Launcher: `output/ablations/scene_9d1180_dit_best_dino/run_entry.sh`; sibling `run.log` and `exit_code.txt`.
- New prediction ID: `dit_2849053_step28537`; point/dense HDF5 exports under the scene's `predictions/` directory.
- Comparison columns: GT, RadarGen, DiT without bottleneck, best-F1 DINO. Doppler, RCS and heatmap rows; all temporal frames, 10 fps.
- Output: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/9d1180d52b6a46328063042b8fb325de/video_radargen_dit_2849053_dino_2773475_step15000/comparison.mp4`.
- Comparison only (`--skip-overlays`); existing DINO overlays unchanged. No Slurm submission.

Verify exit status and MP4 metadata before reporting completion.

Completed successfully, exit code **0**. MP4 verified: **194 frames, 10 fps, 19.4 seconds**. Probe metadata saved beside entry logs as `video_verification.json`.
