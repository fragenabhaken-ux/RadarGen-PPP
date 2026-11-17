# DINO four-node run: all-checkpoint SS50 comparison

Training run: `2684031`. Previous-training baseline: initial DINO run `2658284`, final step 10,000.

**Selection criterion:** highest whole-area distance-and-attribute detection F1 on the fixed SS50 benchmark. Exact ties share first place. Other metrics show trade-offs and do not change this selection rule.

New evaluations produce metrics only: no figures, PNGs, TensorBoard output, or prediction exports. Existing matched final-checkpoint and baseline evaluations are reused.

**Best overall on SS50:** DINO four-node step 1000 (F1 = 0.303946).

**Best fine-tuned checkpoint:** DINO four-node step 1000 (F1 = 0.303946).

Best fine-tuned versus previous-training final: F1 difference +0.006386 (+0.639 percentage points).

All entries evaluated the same 50 frames in the same order, with the same inventory and seeds (verified). The best checkpoint is selected on SS50; it has not been selected using the full validation set.

## All metrics

Mean ± population standard deviation; hit rate, class-wise MMD, and median runtime are scalars. Bold marks first place within each metric using unrounded values. Rows without arrows are not ranked; — unavailable or pending. Rankings remain provisional until completion.

| Scope | Metric | DINO initial final (2658284) | DINO four-node step 1000 | DINO four-node step 2000 | DINO four-node step 3000 | DINO four-node step 4000 | DINO four-node step 5000 | DINO four-node step 6000 | DINO four-node step 7000 | DINO four-node step 8000 | DINO four-node step 9000 | DINO four-node step 9971 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.625 ± 0.358 | **1.593 ± 0.314** | 1.632 ± 0.367 | 1.629 ± 0.372 | 1.661 ± 0.392 | 1.644 ± 0.371 | 1.634 ± 0.368 | 1.630 ± 0.352 | 1.627 ± 0.384 | 1.647 ± 0.382 | 1.641 ± 0.392 |
| Entire Area | CD Full ↓ | 0.039 ± 0.008 | 0.039 ± 0.007 | 0.038 ± 0.007 | 0.038 ± 0.008 | 0.039 ± 0.008 | 0.038 ± 0.007 | 0.038 ± 0.007 | **0.037 ± 0.007** | 0.037 ± 0.007 | 0.037 ± 0.007 | 0.037 ± 0.007 |
| Entire Area | IoU@1m ↑ | 0.312 ± 0.105 | **0.322 ± 0.107** | 0.316 ± 0.115 | 0.315 ± 0.112 | 0.308 ± 0.113 | 0.314 ± 0.113 | 0.313 ± 0.111 | 0.314 ± 0.111 | 0.318 ± 0.115 | 0.311 ± 0.114 | 0.316 ± 0.113 |
| Entire Area | DA Recall ↑ | 0.295 ± 0.105 | **0.310 ± 0.102** | 0.292 ± 0.108 | 0.293 ± 0.110 | 0.285 ± 0.109 | 0.290 ± 0.110 | 0.293 ± 0.110 | 0.294 ± 0.106 | 0.298 ± 0.112 | 0.289 ± 0.110 | 0.296 ± 0.111 |
| Entire Area | DA Precision ↑ | 0.307 ± 0.104 | 0.303 ± 0.110 | 0.310 ± 0.112 | 0.310 ± 0.113 | 0.308 ± 0.115 | **0.313 ± 0.114** | 0.310 ± 0.110 | 0.307 ± 0.110 | 0.308 ± 0.111 | 0.309 ± 0.116 | 0.312 ± 0.113 |
| Entire Area | DA F1 ↑ | 0.298 ± 0.104 | **0.304 ± 0.104** | 0.298 ± 0.109 | 0.299 ± 0.111 | 0.294 ± 0.111 | 0.298 ± 0.111 | 0.298 ± 0.109 | 0.298 ± 0.107 | 0.300 ± 0.111 | 0.296 ± 0.112 | 0.301 ± 0.111 |
| Entire Area | MMD Loc. ↓ | 0.049 ± 0.046 | 0.050 ± 0.052 | 0.046 ± 0.055 | 0.045 ± 0.050 | 0.049 ± 0.058 | 0.048 ± 0.056 | 0.045 ± 0.048 | **0.042 ± 0.036** | 0.047 ± 0.052 | 0.043 ± 0.042 | 0.049 ± 0.044 |
| Entire Area | MMD RCS ↓ | 0.207 ± 0.111 | 0.244 ± 0.134 | 0.174 ± 0.096 | 0.172 ± 0.099 | 0.171 ± 0.097 | 0.151 ± 0.093 | 0.165 ± 0.111 | 0.143 ± 0.097 | 0.133 ± 0.088 | **0.130 ± 0.087** | 0.138 ± 0.099 |
| Entire Area | MMD Doppler ↓ | 0.137 ± 0.272 | **0.136 ± 0.301** | 0.137 ± 0.314 | 0.141 ± 0.323 | 0.158 ± 0.335 | 0.144 ± 0.337 | 0.147 ± 0.325 | 0.149 ± 0.325 | 0.144 ± 0.336 | 0.144 ± 0.329 | 0.150 ± 0.333 |
| Foreground | CD Loc. ↓ | 1.002 ± 0.683 | 1.054 ± 0.737 | 1.009 ± 0.595 | 1.034 ± 0.722 | 1.014 ± 0.661 | 1.029 ± 0.668 | 1.000 ± 0.621 | 0.993 ± 0.596 | 1.001 ± 0.626 | 1.026 ± 0.678 | **0.962 ± 0.624** |
| Foreground | CD Full ↓ | 0.057 ± 0.037 | 0.060 ± 0.043 | **0.055 ± 0.036** | 0.058 ± 0.040 | 0.061 ± 0.041 | 0.058 ± 0.041 | 0.061 ± 0.046 | 0.058 ± 0.046 | 0.060 ± 0.045 | 0.062 ± 0.045 | 0.057 ± 0.042 |
| Foreground | Density Similarity ↑ | 0.422 ± 0.352 | **0.446 ± 0.352** | 0.437 ± 0.359 | 0.433 ± 0.357 | 0.439 ± 0.365 | 0.437 ± 0.353 | 0.420 ± 0.357 | 0.441 ± 0.361 | 0.415 ± 0.348 | 0.387 ± 0.342 | 0.429 ± 0.366 |
| Foreground | Hit Rate ↑ | 0.777 | **0.814** | 0.777 | 0.788 | 0.762 | 0.784 | 0.766 | 0.792 | 0.792 | 0.732 | 0.751 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1044.320 ± 310.947 | 1115.600 ± 282.880 | 1021.000 ± 292.946 | 1025.740 ± 297.012 | 1010.420 ± 306.936 | 1006.780 ± 291.365 | 1027.280 ± 308.366 | 1039.560 ± 286.006 | 1049.660 ± 308.211 | 1017.360 ± 296.956 | 1027.580 ± 296.780 |
| Additional | Point-count absolute error ↓ | 141.540 ± 156.163 | **140.380 ± 152.406** | 158.140 ± 147.350 | 146.000 ± 141.210 | 159.440 ± 149.126 | 162.400 ± 147.881 | 158.660 ± 144.889 | 152.500 ± 156.784 | 150.040 ± 149.880 | 154.700 ± 148.833 | 149.880 ± 156.189 |
| Additional | PPP expected total count | 1047.784 ± 309.060 | 1121.794 ± 279.988 | 1027.968 ± 294.197 | 1028.383 ± 294.164 | 1016.534 ± 304.048 | 1012.274 ± 291.675 | 1032.389 ± 302.240 | 1039.501 ± 284.454 | 1054.923 ± 309.274 | 1024.037 ± 299.645 | 1027.576 ± 291.684 |
| Additional | PPP total-count absolute error ↓ | 141.023 ± 156.404 | **140.744 ± 151.971** | 148.952 ± 145.158 | 148.603 ± 140.502 | 156.384 ± 150.394 | 156.318 ± 142.040 | 152.720 ± 147.730 | 150.637 ± 150.620 | 142.583 ± 150.867 | 149.657 ± 143.806 | 149.640 ± 150.669 |
| Additional | RCS MAE at target cells ↓ | 4.525 ± 1.607 | **4.495 ± 1.565** | 4.598 ± 1.658 | 4.632 ± 1.690 | 4.624 ± 1.674 | 4.677 ± 1.693 | 4.681 ± 1.682 | 4.729 ± 1.717 | 4.757 ± 1.754 | 4.766 ± 1.739 | 4.788 ± 1.755 |
| Additional | Doppler MAE at target cells ↓ | **1.668 ± 1.186** | 1.680 ± 1.180 | 1.683 ± 1.194 | 1.690 ± 1.214 | 1.720 ± 1.222 | 1.711 ± 1.233 | 1.756 ± 1.232 | 1.734 ± 1.207 | 1.772 ± 1.251 | 1.751 ± 1.247 | 1.757 ± 1.210 |
| Additional foreground | PPP expected count per box | 5.158 ± 7.863 | 6.143 ± 9.568 | 5.202 ± 8.157 | 5.232 ± 8.267 | 5.386 ± 8.387 | 5.348 ± 8.675 | 5.247 ± 8.034 | 5.249 ± 7.969 | 5.257 ± 7.958 | 5.014 ± 7.377 | 4.819 ± 6.993 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.115 ± 7.493 | **4.783 ± 7.203** | 4.990 ± 7.675 | 4.969 ± 7.578 | 5.107 ± 7.773 | 5.020 ± 7.648 | 5.209 ± 7.964 | 5.160 ± 7.768 | 5.161 ± 7.777 | 5.331 ± 7.804 | 5.168 ± 7.892 |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.783 ± 0.271 | **0.799 ± 0.274** | 0.777 ± 0.281 | 0.774 ± 0.282 | 0.782 ± 0.273 | 0.766 ± 0.288 | 0.770 ± 0.291 | 0.778 ± 0.285 | 0.766 ± 0.295 | 0.766 ± 0.296 | 0.767 ± 0.292 |
| Foreground (car) | MMD Loc. ↓ | 0.163 | 0.101 | 0.146 | 0.028 | **0.026** | 0.079 | 0.056 | 0.122 | 0.142 | 0.046 | 0.110 |
| Foreground (car) | MMD RCS ↓ | 0.059 | 0.055 | 0.034 | 0.258 | 0.129 | 0.030 | 0.106 | **0.029** | 0.112 | 0.190 | 0.036 |
| Foreground (car) | MMD Doppler ↓ | 0.064 | 0.043 | 0.196 | 0.026 | 0.121 | 0.104 | 0.037 | 0.046 | **0.011** | 0.018 | 0.091 |
| Foreground (truck) | MMD Loc. ↓ | 0.152 | 0.204 | 0.101 | 0.222 | 0.081 | 0.107 | 0.088 | **0.025** | 0.076 | 0.071 | 0.081 |
| Foreground (truck) | MMD RCS ↓ | 0.248 | 0.433 | 0.624 | 0.420 | **0.208** | 0.261 | 0.739 | 0.250 | 0.432 | 0.436 | 0.252 |
| Foreground (truck) | MMD Doppler ↓ | 0.121 | 0.196 | 0.194 | 0.137 | **0.066** | 0.164 | 0.137 | 0.270 | 0.277 | 0.191 | 0.228 |
| Foreground (trailer) | MMD Loc. ↓ | 0.071 | 0.077 | 0.026 | 0.029 | 0.044 | 0.066 | 0.060 | 0.041 | 0.056 | 0.063 | **0.008** |
| Foreground (trailer) | MMD RCS ↓ | 0.864 | 0.666 | 0.865 | 0.817 | 0.629 | 0.952 | 1.143 | 0.808 | **0.463** | 0.591 | 0.469 |
| Foreground (trailer) | MMD Doppler ↓ | 0.654 | 0.695 | 0.513 | 0.506 | 0.576 | 0.465 | 0.522 | **0.372** | 1.034 | 0.825 | 0.395 |
| Runtime | Inference time (s/frame) ↓ | 0.260 ± 0.001 | 0.255 ± 0.001 | 0.260 ± 0.001 | 0.257 ± 0.001 | 0.259 ± 0.001 | 0.256 ± 0.001 | **0.254 ± 0.001** | 0.257 ± 0.001 | 0.258 ± 0.001 | 0.262 ± 0.001 | 0.260 ± 0.001 |
| Runtime | Median inference time (s/frame) ↓ | 0.260 | 0.255 | 0.260 | 0.257 | 0.259 | 0.256 | **0.254** | 0.257 | 0.258 | 0.262 | 0.260 |

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
