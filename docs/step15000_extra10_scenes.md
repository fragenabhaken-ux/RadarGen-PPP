# Step 15000: additional scenes and full validation

Checkpoint: run 2773475, epoch_68_step_15000.pth. No PNG output.

The ten additional scenes were chosen with Python Random(42) from sorted validation scene tokens, excluding the three previously rendered scenes. All temporal frames are exported once per model and reused for comparison and overlay MP4s.

| Scene | Token | Temporal frames |
| --- | --- | --- |
| 1 | `d524527bfa2845c5bfe53952a5f50da5` | 194 |
| 2 | `2d7f4bc4b05a45bbb7a467d05fcb6e43` | 194 |
| 3 | `0f71ba062e3348e2ac15b2c4883a28ec` | 194 |
| 4 | `63fe44fc70824d4499228e8d293d0e91` | 194 |
| 5 | `51c56c7100164c8396b26672ac0c00c1` | 194 |
| 6 | `4e34f81d2670455aa5f9bcbb2b9cfbdf` | 194 |
| 7 | `2f60705136a348a0aad2283d38f33edd` | 194 |
| 8 | `2aaf3854cdee45f0a27d1da41b399509` | 194 |
| 9 | `b4225721431749d1b2ffb09ad40531d1` | 194 |
| 10 | `28ba0b16dc0b4f13ad32a47b2f8e5133` | 194 |

Scene outputs: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/<token>/video_radargen_dino_2773475_step15000/comparison.mp4` and `overlays/dino_2773475_step15000/overlay.mp4`.

Full validation covers every annotated validation keyframe; see [comparison](FULL_validation_step15000_vs_radargen.md).
