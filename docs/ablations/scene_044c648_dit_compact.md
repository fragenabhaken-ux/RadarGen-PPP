# Scene videos: DiT without bottleneck and compact DINO

Scene: `044c648ac12345f1aedf33c9f91cdc5a`. User requested comparison and overlays on the entry node. Final checkpoints: DiT **2849053 / 28,537**, compact DINO **2849062 / 10,372**. All temporal frames, seed 42, 10 fps; Doppler, RCS and heatmap rows. Existing matching GT and RadarGen scene exports reused.

- Config: `evaluation/configs/scene_comparison_traffic2_ablation_dit_compact.yaml`.
- Entry launcher: `output/ablations/scene_044c648_dit_compact/run_entry.sh`.
- Log: `output/ablations/scene_044c648_dit_compact/run.log`; exit status: sibling `exit_code.txt` when finished. No Slurm job.
- Scene root: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/044c648ac12345f1aedf33c9f91cdc5a/`.
- New prediction IDs: `dit_2849053_step28537`, `dino_compact_2849062_step10372`, stored under `predictions/` with reusable point/dense HDF5 exports.
- Comparison: `video_radargen_dit_2849053_compact_dino_2849062/comparison.mp4` below the scene root. Columns: GT, RadarGen, DiT, compact DINO.
- DiT overlay: `overlays/dit_2849053_step28537/overlay.mp4`.
- Compact-DINO overlay: `overlays/dino_compact_2849062_step10372/overlay.mp4`.

Launch does not imply completion. Verify exit code, export completeness and MP4 metadata before reporting success.

Completed successfully: launcher exit code **0**. All three MP4s verified with ffprobe: **194 frames, 10 fps, 19.4 seconds**. Probe metadata saved in `output/ablations/scene_044c648_dit_compact/video_verification.json`.
