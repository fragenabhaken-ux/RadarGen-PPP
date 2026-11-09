# DINO four-node run: all-checkpoint SS50 comparison

Training run: `2684031`. Previous-training baseline: initial DINO run `2658284`, final step 10,000.

**Selection criterion:** highest whole-area distance-and-attribute detection F1 on the fixed SS50 benchmark. Exact ties share first place. Other metrics show trade-offs and do not change this selection rule.

New evaluations produce metrics only: no figures, PNGs, TensorBoard output, or prediction exports. Existing matched final-checkpoint and baseline evaluations are reused.

**Sweep incomplete: 3/11 evaluations complete.** Rankings below are provisional; the overall winner will be declared after every checkpoint finishes.

## All metrics

Mean ± population standard deviation; hit rate, class-wise MMD, and median runtime are scalars. Bold marks first place within each metric using unrounded values. Rows without arrows are not ranked; — unavailable or pending. Rankings remain provisional until completion.

| Scope | Metric | DINO initial final (2658284) | DINO four-node step 1000 | DINO four-node step 2000 | DINO four-node step 3000 | DINO four-node step 4000 | DINO four-node step 5000 | DINO four-node step 6000 | DINO four-node step 7000 | DINO four-node step 8000 | DINO four-node step 9000 | DINO four-node step 9971 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.625 ± 0.358** | — | — | — | — | — | — | — | — | 1.647 ± 0.382 | 1.641 ± 0.392 |
| Entire Area | CD Full ↓ | 0.039 ± 0.008 | — | — | — | — | — | — | — | — | **0.037 ± 0.007** | 0.037 ± 0.007 |
| Entire Area | IoU@1m ↑ | 0.312 ± 0.105 | — | — | — | — | — | — | — | — | 0.311 ± 0.114 | **0.316 ± 0.113** |
| Entire Area | DA Recall ↑ | 0.295 ± 0.105 | — | — | — | — | — | — | — | — | 0.289 ± 0.110 | **0.296 ± 0.111** |
| Entire Area | DA Precision ↑ | 0.307 ± 0.104 | — | — | — | — | — | — | — | — | 0.309 ± 0.116 | **0.312 ± 0.113** |
| Entire Area | DA F1 ↑ | 0.298 ± 0.104 | — | — | — | — | — | — | — | — | 0.296 ± 0.112 | **0.301 ± 0.111** |
| Entire Area | MMD Loc. ↓ | 0.049 ± 0.046 | — | — | — | — | — | — | — | — | **0.043 ± 0.042** | 0.049 ± 0.044 |
| Entire Area | MMD RCS ↓ | 0.207 ± 0.111 | — | — | — | — | — | — | — | — | **0.130 ± 0.087** | 0.138 ± 0.099 |
| Entire Area | MMD Doppler ↓ | **0.137 ± 0.272** | — | — | — | — | — | — | — | — | 0.144 ± 0.329 | 0.150 ± 0.333 |
| Foreground | CD Loc. ↓ | 1.002 ± 0.683 | — | — | — | — | — | — | — | — | 1.026 ± 0.678 | **0.962 ± 0.624** |
| Foreground | CD Full ↓ | 0.057 ± 0.037 | — | — | — | — | — | — | — | — | 0.062 ± 0.045 | **0.057 ± 0.042** |
| Foreground | Density Similarity ↑ | 0.422 ± 0.352 | — | — | — | — | — | — | — | — | 0.387 ± 0.342 | **0.429 ± 0.366** |
| Foreground | Hit Rate ↑ | **0.777** | — | — | — | — | — | — | — | — | 0.732 | 0.751 |
| Additional | GT point count | 1095.540 ± 320.639 | — | — | — | — | — | — | — | — | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1044.320 ± 310.947 | — | — | — | — | — | — | — | — | 1017.360 ± 296.956 | 1027.580 ± 296.780 |
| Additional | Point-count absolute error ↓ | **141.540 ± 156.163** | — | — | — | — | — | — | — | — | 154.700 ± 148.833 | 149.880 ± 156.189 |
| Additional | PPP expected total count | 1047.784 ± 309.060 | — | — | — | — | — | — | — | — | 1024.037 ± 299.645 | 1027.576 ± 291.684 |
| Additional | PPP total-count absolute error ↓ | **141.023 ± 156.404** | — | — | — | — | — | — | — | — | 149.657 ± 143.806 | 149.640 ± 150.669 |
| Additional | RCS MAE at target cells ↓ | **4.525 ± 1.607** | — | — | — | — | — | — | — | — | 4.766 ± 1.739 | 4.788 ± 1.755 |
| Additional | Doppler MAE at target cells ↓ | **1.668 ± 1.186** | — | — | — | — | — | — | — | — | 1.751 ± 1.247 | 1.757 ± 1.210 |
| Additional foreground | PPP expected count per box | 5.158 ± 7.863 | — | — | — | — | — | — | — | — | 5.014 ± 7.377 | 4.819 ± 6.993 |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.115 ± 7.493** | — | — | — | — | — | — | — | — | 5.331 ± 7.804 | 5.168 ± 7.892 |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.783 ± 0.271** | — | — | — | — | — | — | — | — | 0.766 ± 0.296 | 0.767 ± 0.292 |
| Foreground (car) | MMD Loc. ↓ | 0.163 | — | — | — | — | — | — | — | — | **0.046** | 0.110 |
| Foreground (car) | MMD RCS ↓ | 0.059 | — | — | — | — | — | — | — | — | 0.190 | **0.036** |
| Foreground (car) | MMD Doppler ↓ | 0.064 | — | — | — | — | — | — | — | — | **0.018** | 0.091 |
| Foreground (truck) | MMD Loc. ↓ | 0.152 | — | — | — | — | — | — | — | — | **0.071** | 0.081 |
| Foreground (truck) | MMD RCS ↓ | **0.248** | — | — | — | — | — | — | — | — | 0.436 | 0.252 |
| Foreground (truck) | MMD Doppler ↓ | **0.121** | — | — | — | — | — | — | — | — | 0.191 | 0.228 |
| Foreground (trailer) | MMD Loc. ↓ | 0.071 | — | — | — | — | — | — | — | — | 0.063 | **0.008** |
| Foreground (trailer) | MMD RCS ↓ | 0.864 | — | — | — | — | — | — | — | — | 0.591 | **0.469** |
| Foreground (trailer) | MMD Doppler ↓ | 0.654 | — | — | — | — | — | — | — | — | 0.825 | **0.395** |
| Runtime | Inference time (s/frame) ↓ | 0.260 ± 0.001 | — | — | — | — | — | — | — | — | 0.262 ± 0.001 | **0.260 ± 0.001** |
| Runtime | Median inference time (s/frame) ↓ | 0.260 | — | — | — | — | — | — | — | — | 0.262 | **0.260** |

## Sources

- DINO initial final (2658284): [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680332_PPP_validation_ss50_2658284/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680332_PPP_validation_ss50_2658284/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2658284_PPP_train_dino_shifted_1e4/checkpoints/epoch_11_step_10000.pth`.
- DINO four-node step 1000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_001000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_001000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_4_step_1000.pth`.
- DINO four-node step 2000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_002000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_002000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_9_step_2000.pth`.
- DINO four-node step 3000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_003000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_003000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_13_step_3000.pth`.
- DINO four-node step 4000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_004000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_004000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_18_step_4000.pth`.
- DINO four-node step 5000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_005000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_005000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_22_step_5000.pth`.
- DINO four-node step 6000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_006000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_006000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_27_step_6000.pth`.
- DINO four-node step 7000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_007000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_007000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_32_step_7000.pth`.
- DINO four-node step 8000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_008000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_008000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_36_step_8000.pth`.
- DINO four-node step 9000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701633_PPP_validation_ss50_dino_step9000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701633_PPP_validation_ss50_dino_step9000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_41_step_9000.pth`.
- DINO four-node step 9971: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_45_step_9971.pth`.
