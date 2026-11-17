# Scene videos: random-init DINO

Scene `044c648ac12345f1aedf33c9f91cdc5a`, all 194 temporal frames, 10 fps. Checkpoint **2847666 / 15,000**, matching the SS50 evaluation (not the later continuation). User requested the same video workflow as the DiT/compact-DINO comparison: entry-node inference, comparison and overlay, Doppler/RCS/heatmap rows.

- Config: `evaluation/configs/scene_comparison_traffic2_ablation_dino_random.yaml`.
- Launcher: `output/ablations/scene_044c648_dino_random/run_entry.sh`.
- Log: sibling `run.log`; exit code: `exit_code.txt` when finished.
- Root: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/044c648ac12345f1aedf33c9f91cdc5a/`.
- Predictions: `predictions/dino_random_2847666_step15000.h5` and corresponding `_dense.h5`.
- Comparison (GT, RadarGen, random DINO): `video_radargen_dino_random_2847666_step15000/comparison.mp4`.
- Overlay: `overlays/dino_random_2847666_step15000/overlay.mp4`.

No Slurm submission. Verify successful exit and MP4 frame counts/duration before reporting completion.

Completed successfully, exit code **0**. Both videos verified with ffprobe: **194 frames, 10 fps, 19.4 seconds**. Probe metadata: `output/ablations/scene_044c648_dino_random/video_verification.json`.
