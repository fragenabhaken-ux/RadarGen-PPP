# ResNet comparisons across all validation scenes

Requested 2026-11-14: comparison videos for every validation scene, using the final ResNet checkpoint, final DINO checkpoint and released RadarGen. Ground truth remains the first column, matching the existing comparison format.

- **C3P-dar-RN50:** run 2882612, step 25,030.
- **C3P-dar-Dv3:** run 2773475, final step 19,914 (not the 15,000-step checkpoint).
- **RadarGen:** released `RadarGen_600M_512px_TS.safetensors`, 20 inference steps.
- **Scope:** 49 validation scenes, 9,481 temporal frames. Ten fps; Doppler, RCS and heatmap rows.

Initial cached prediction coverage was 10/49 ResNet scenes, 3/49 final-DINO scenes and 13/49 RadarGen scenes. Complete compatible exports are reused. Checkpoint identity, complete flags, frame identities and dataset/manifest provenance are checked. Original prediction files are not relabelled or modified: optional render-only display labels provide the current architecture names.

## Execution and outputs

GPU prediction job **2904428**: `jsc_jupiter/VIDEO_resnet_all_scenes.sbatch`, booster partition, develbooster reservation, one node, four workers/four GPUs, two-hour limit. Scenes are divided among workers; missing models are exported sequentially within each scene, under the existing per-scene export lock. This job performs inference only, with no training or metric reruns.

CPU rendering runs on the entry node, up to four concurrent videos, via `jsc_jupiter/VIDEO_resnet_all_scenes_entry.sh`. It waits for each scene's three prediction files, renders and checks the MP4 with ffprobe. The initial detached-shell launch did not persist; it was replaced with a persistent execution session (38588).

Per-scene MP4:

```text
/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<scene>/video_resnet50_2882612_step25030_dino_2773475_final_radargen/comparison.mp4
```

The video index is `docs/ablations/resnet_all_scene_videos.md`. Status, exact configurations, cached coverage, GPU worker logs and CPU renderer log are in `output/ablations/resnet_all_scenes_final_dino/`. GPU scheduler logs are `jsc_jupiter/logs/2904428_VIDEO_resnet_all_scenes.out` and `.err`.

Implementation: `scripts/resnet_all_scene_videos.py`; existing comparison renderer extended with optional `--model-labels`. Syntax checks, shell checks and the two existing scene-rendering regression tests passed. Completed videos must match the actual scene frame count, 10 fps and the corresponding duration. Submission does not imply that exports or videos have completed.

To resume CPU rendering, rerun `bash jsc_jupiter/VIDEO_resnet_all_scenes_entry.sh`. Completed videos are verified and reused. GPU exports also resume through the established exporter, preserving complete results.
