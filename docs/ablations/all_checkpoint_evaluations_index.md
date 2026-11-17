# All ablations: first- and second-final evaluation index

Both fixed training-budget checkpoints are included, regardless of which has better SS50. GPU jobs: one per architecture, develbooster, four GPUs, two-hour limit. ResNet waits for successful training completion. CPU count sweeps and video rendering run on the entry node; no additional GPU submissions.

| Checkpoint | Step | SS50 | Full validation / PNGs / predictions | VoxelNeXt | Count sweep | Videos |
|---|---:|---|---|---|---|---|
| DiT without bottleneck, first-run final | 28537 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2849053_step28537/metrics/summary.json) | [Metrics](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/ablation_checkpoint_programme/dit_first_2849053/attempt_000/metrics.json) | [Results](/e/project1/nxtaim-1/huber7/repos/RadarGen/output/paper_material/count_sweep_evaluation/ss50_dit_no_bottleneck_first_count_sweep/results.md) | complete |
| DiT without bottleneck, second-run final | 58057 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2867524_step58057_checkpoint_audit/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dit_dcae_no_bottleneck_2867524_step58057/metrics/summary.json) | [Metrics](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/ablation_checkpoint_programme/dit_second_2867524/attempt_000/metrics.json) | [Results](/e/project1/nxtaim-1/huber7/repos/RadarGen/output/paper_material/count_sweep_evaluation/ss50_dit_no_bottleneck_count_sweep/results.md) | complete |
| Compact DINO, first-run final | 10372 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_neck_compact_2849062_step10372/metrics/summary.json) | pending | pending | queued_or_running |
| Compact DINO, second-run final | 20595 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2867528_step20595_checkpoint_audit/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_neck_compact_2867528_step20595/metrics/summary.json) | [Metrics](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/2895299_dino_compact_2867528_step20595_z1_flatup/metrics.json) | [Results](/e/project1/nxtaim-1/huber7/repos/RadarGen/output/paper_material/count_sweep_evaluation/ss50_dino_compact_count_sweep/results.md) | complete |
| Random DINO, first-run final | 15000 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_random_init_2847666_step15000/metrics/summary.json) | pending | pending | queued_or_running |
| Random DINO, second-run final | 29915 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2863217_step29915_checkpoint_audit/summary.json) | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/dino_random_init_2863217_step29915/metrics/summary.json) | [Metrics](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/detection/ablation_checkpoint_programme/random_second_2863217/attempt_000/metrics.json) | [Results](/e/project1/nxtaim-1/huber7/repos/RadarGen/output/paper_material/count_sweep_evaluation/ss50_dino_random_count_sweep/results.md) | complete |
| ResNet-50, first-run final | 12575 | [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/summary.json) | pending | pending | pending | waiting |
| ResNet-50, second-run final | after training | pending | pending | pending | pending | waiting |

Full prediction archives are `metrics/predictions.h5`; PNGs are below `metrics/plots/`. A partial full-validation summary can exist before completion; inspect its complete flag. Count sweeps use targets 100–2000 in steps of 100, three seeds (42/43/44), and fixed maps. VoxelNeXt uses the frozen epoch-400 detector, z=1 flat-up, physical marks/absolute Doppler and the matched three-class scoring protocol.

## Standard-scene videos

Each video: GT · selected ablation checkpoint · DINO 15k · RadarGen; Doppler, RCS and heatmap. Ten scenes per checkpoint, 80 videos overall; prior verified videos reused.

### DiT without bottleneck, first-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [2f60705136a348a0aad2283d38f33edd](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2f60705136a348a0aad2283d38f33edd/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [2aaf3854cdee45f0a27d1da41b399509](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2aaf3854cdee45f0a27d1da41b399509/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [b4225721431749d1b2ffb09ad40531d1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/b4225721431749d1b2ffb09ad40531d1/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)
- [28ba0b16dc0b4f13ad32a47b2f8e5133](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/28ba0b16dc0b4f13ad32a47b2f8e5133/video_dit_2849053_step28537_dino_2773475_step15000_radargen/comparison.mp4)

### DiT without bottleneck, second-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [2f60705136a348a0aad2283d38f33edd](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2f60705136a348a0aad2283d38f33edd/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [2aaf3854cdee45f0a27d1da41b399509](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2aaf3854cdee45f0a27d1da41b399509/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [b4225721431749d1b2ffb09ad40531d1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/b4225721431749d1b2ffb09ad40531d1/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)
- [28ba0b16dc0b4f13ad32a47b2f8e5133](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/28ba0b16dc0b4f13ad32a47b2f8e5133/video_dit_2867524_step58057_dino_2773475_step15000_radargen/comparison.mp4)

### Compact DINO, first-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [2f60705136a348a0aad2283d38f33edd](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2f60705136a348a0aad2283d38f33edd/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [2aaf3854cdee45f0a27d1da41b399509](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2aaf3854cdee45f0a27d1da41b399509/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [b4225721431749d1b2ffb09ad40531d1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/b4225721431749d1b2ffb09ad40531d1/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)
- [28ba0b16dc0b4f13ad32a47b2f8e5133](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/28ba0b16dc0b4f13ad32a47b2f8e5133/video_dino_compact_2849062_step10372_dino_2773475_step15000_radargen/comparison.mp4)

### Compact DINO, second-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [2f60705136a348a0aad2283d38f33edd](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2f60705136a348a0aad2283d38f33edd/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [2aaf3854cdee45f0a27d1da41b399509](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2aaf3854cdee45f0a27d1da41b399509/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [b4225721431749d1b2ffb09ad40531d1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/b4225721431749d1b2ffb09ad40531d1/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)
- [28ba0b16dc0b4f13ad32a47b2f8e5133](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/28ba0b16dc0b4f13ad32a47b2f8e5133/video_dino_compact_2867528_step20595_dino_2773475_step15000_radargen/comparison.mp4)

### Random DINO, first-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dino_random_2847666_step15000_dino_2773475_step15000_radargen/comparison.mp4)
- 2f60705136a348a0aad2283d38f33edd: pending
- 2aaf3854cdee45f0a27d1da41b399509: pending
- b4225721431749d1b2ffb09ad40531d1: pending
- 28ba0b16dc0b4f13ad32a47b2f8e5133: pending

### Random DINO, second-run final

- [d524527bfa2845c5bfe53952a5f50da5](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/d524527bfa2845c5bfe53952a5f50da5/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [2d7f4bc4b05a45bbb7a467d05fcb6e43](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2d7f4bc4b05a45bbb7a467d05fcb6e43/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [0f71ba062e3348e2ac15b2c4883a28ec](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/0f71ba062e3348e2ac15b2c4883a28ec/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [63fe44fc70824d4499228e8d293d0e91](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/63fe44fc70824d4499228e8d293d0e91/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [51c56c7100164c8396b26672ac0c00c1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/51c56c7100164c8396b26672ac0c00c1/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [4e34f81d2670455aa5f9bcbb2b9cfbdf](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/4e34f81d2670455aa5f9bcbb2b9cfbdf/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [2f60705136a348a0aad2283d38f33edd](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2f60705136a348a0aad2283d38f33edd/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [2aaf3854cdee45f0a27d1da41b399509](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/2aaf3854cdee45f0a27d1da41b399509/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [b4225721431749d1b2ffb09ad40531d1](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/b4225721431749d1b2ffb09ad40531d1/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)
- [28ba0b16dc0b4f13ad32a47b2f8e5133](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/scene_comparisons/28ba0b16dc0b4f13ad32a47b2f8e5133/video_dino_random_2863217_step29915_dino_2773475_step15000_radargen/comparison.mp4)

### ResNet-50, first-run final

- d524527bfa2845c5bfe53952a5f50da5: pending
- 2d7f4bc4b05a45bbb7a467d05fcb6e43: pending
- 0f71ba062e3348e2ac15b2c4883a28ec: pending
- 63fe44fc70824d4499228e8d293d0e91: pending
- 51c56c7100164c8396b26672ac0c00c1: pending
- 4e34f81d2670455aa5f9bcbb2b9cfbdf: pending
- 2f60705136a348a0aad2283d38f33edd: pending
- 2aaf3854cdee45f0a27d1da41b399509: pending
- b4225721431749d1b2ffb09ad40531d1: pending
- 28ba0b16dc0b4f13ad32a47b2f8e5133: pending

### ResNet-50, second-run final

Waiting for checkpoint resolution after training.

