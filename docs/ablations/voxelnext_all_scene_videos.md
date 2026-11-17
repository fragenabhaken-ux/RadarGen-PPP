# VoxelNeXt scene comparison videos

Completed 2026-11-14: all 49 validation scenes, covering all 1,906 saved annotated keyframes, rendered and verified on the entry node without new GPU inference.

Output: `output/voxelnext_scene_videos_resnet/<scene>/comparison.mp4`. The folder's `README.md` links every scene video; `status.json` records completion, frame counts, fps and duration. Each scene also includes `preview.png` and `manifest.json` with exact archive/detection paths, detector settings, sample identities and timestamps. Total frame count, individual frame counts, fps and duration were checked against the saved manifests.

Panels: final C3P-dar-RN50 (2882612, step 25,030) and released RadarGen. Points are coloured by RCS. Orange boxes are car/truck/trailer annotations; dashed green boxes are saved VoxelNeXt predictions at confidence >= 0.25. Ego vehicle is DeepPink with a 0.5-point black outline. Detection boxes are transformed from RADAR_LEFT_SIDE back into the common flat-up BEV frame.

These are annotated-keyframe sequences at approximately 2 fps, not detections for every 10 Hz temporal frame. The last sample excluded by the validation protocol is absent; there is no interpolation or tracking. The same frozen epoch-400 VoxelNeXt checkpoint was used for each model's own generated cloud, with height z=1 in flat-up coordinates. Equivalence to the original private preprocessing remains unverified.

Code: `scripts/render_voxelnext_scene_video.py` (single scene with optional `--scene` and `--output`); `scripts/render_voxelnext_all_scenes.py` (four CPU render processes, reusable verified outputs). The initial example scene was copied from its existing verified render. Coordinator log: `output/voxelnext_scene_videos_resnet/renderer.log`; per-scene logs: sibling `logs/`.
