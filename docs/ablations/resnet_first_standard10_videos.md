# ResNet first-checkpoint comparison videos

Requested 2026-11-14: render the ten standard scenes using existing complete prediction exports, then copy all ten MP4s into the repository output directory. No GPU job is required or submitted.

Columns: **ground truth · C3P-dar-RN50 (2856174 / step 12,575) · final C3P-dar-Dv3 (2773475 / step 19,914) · RadarGen**. Rows: Doppler, RCS and heatmap. The reference is the final DINO checkpoint, not the 15k checkpoint. All videos use 10 fps and the original scene frame sequences.

Local copies: `output/resnet_first_checkpoint_comparisons_10/<scene>.mp4`. Its `manifest.json` records scene status, original path, local copy, file size and ffprobe results. Live state and per-scene logs: `output/ablations/resnet_first_standard10/`; coordinator log: `renderer.log`.

Original renders: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<scene>/video_resnet50_2856174_step12575_dino_2773475_final_radargen/comparison.mp4`.

Implementation: `scripts/render_resnet_first_standard10.py`. Entry-node launcher: `jsc_jupiter/VIDEO_resnet_first_standard10_entry.sh`; persistent execution session 40055. Four CPU render processes at most, each with one BLAS/OpenMP thread. Preflight checks export completeness, selected checkpoint, dataset/manifest identity and frame alignment. Completed videos are checked for frame count, fps and duration before atomic copying. Original predictions are unchanged. Verified matching renders can be reused on restart.

Run `bash jsc_jupiter/VIDEO_resnet_first_standard10_entry.sh` to resume. The ten scene tokens come from `jsc_jupiter/scene_extra10_step15000_tokens.txt`.

Completed 2026-11-14: all ten videos rendered successfully, passed frame-count/FPS/duration checks, and were copied into `output/resnet_first_checkpoint_comparisons_10/`. All copies match their recorded source file sizes; the manifest contains complete status and ffprobe results for every scene.
