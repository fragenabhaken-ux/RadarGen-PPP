# DINO optimizer-resume: all-checkpoint SS50 comparison

Training run: `2773475`. Previous-training baseline: four-node DINO run `2684031`, final step 9,971.

**Selection criterion:** highest whole-area distance-and-attribute detection F1 on the fixed SS50 benchmark. Exact ties share first place. Other metrics show trade-offs and do not change this selection rule.

New evaluations produce metrics only: no figures, PNGs, TensorBoard output, or prediction exports. Existing matched final-checkpoint and baseline evaluations are reused.

**Best overall on SS50:** DINO resume step 15000 (F1 = 0.308612).

**Best fine-tuned checkpoint:** DINO resume step 15000 (F1 = 0.308612).

Best fine-tuned versus previous-training final: F1 difference +0.007486 (+0.749 percentage points).

All entries evaluated the same 50 frames in the same order, with the same inventory and seeds (verified). The best checkpoint is selected on SS50; it has not been selected using the full validation set.

## All metrics

Mean ± population standard deviation; hit rate, class-wise MMD, and median runtime are scalars. Bold marks first place within each metric using unrounded values. Rows without arrows are not ranked; — unavailable or pending. Rankings remain provisional until completion.

| Scope | Metric | DINO previous four-node final (2684031) | DINO resume step 10000 | DINO resume step 11000 | DINO resume step 12000 | DINO resume step 13000 | DINO resume step 14000 | DINO resume step 15000 | DINO resume step 16000 | DINO resume step 17000 | DINO resume step 18000 | DINO resume step 19000 | DINO resume step 19914 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.641 ± 0.392 | 1.643 ± 0.391 | 1.661 ± 0.387 | 1.667 ± 0.393 | 1.672 ± 0.419 | 1.673 ± 0.393 | **1.526 ± 0.321** | 1.678 ± 0.396 | 1.690 ± 0.385 | 1.702 ± 0.406 | 1.695 ± 0.403 | 1.697 ± 0.411 |
| Entire Area | CD Full ↓ | 0.037 ± 0.007 | 0.037 ± 0.007 | 0.037 ± 0.007 | 0.037 ± 0.007 | 0.037 ± 0.008 | 0.038 ± 0.007 | **0.037 ± 0.007** | 0.037 ± 0.007 | 0.037 ± 0.007 | 0.038 ± 0.007 | 0.037 ± 0.007 | 0.038 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.316 ± 0.113 | 0.313 ± 0.114 | 0.312 ± 0.115 | 0.310 ± 0.117 | 0.311 ± 0.117 | 0.307 ± 0.115 | **0.336 ± 0.107** | 0.306 ± 0.112 | 0.304 ± 0.111 | 0.301 ± 0.110 | 0.306 ± 0.113 | 0.306 ± 0.113 |
| Entire Area | DA Recall ↑ | 0.296 ± 0.111 | 0.292 ± 0.111 | 0.290 ± 0.113 | 0.286 ± 0.107 | 0.284 ± 0.113 | 0.277 ± 0.109 | **0.381 ± 0.128** | 0.279 ± 0.109 | 0.277 ± 0.107 | 0.277 ± 0.108 | 0.276 ± 0.110 | 0.278 ± 0.111 |
| Entire Area | DA Precision ↑ | 0.312 ± 0.113 | 0.307 ± 0.113 | 0.311 ± 0.116 | **0.313 ± 0.113** | 0.310 ± 0.114 | 0.310 ± 0.117 | 0.263 ± 0.090 | 0.307 ± 0.116 | 0.306 ± 0.116 | 0.304 ± 0.116 | 0.305 ± 0.120 | 0.307 ± 0.121 |
| Entire Area | DA F1 ↑ | 0.301 ± 0.111 | 0.297 ± 0.112 | 0.297 ± 0.114 | 0.296 ± 0.110 | 0.294 ± 0.113 | 0.290 ± 0.113 | **0.309 ± 0.104** | 0.290 ± 0.112 | 0.288 ± 0.111 | 0.287 ± 0.111 | 0.288 ± 0.114 | 0.290 ± 0.116 |
| Entire Area | MMD Loc. ↓ | 0.049 ± 0.044 | 0.046 ± 0.047 | 0.044 ± 0.043 | 0.044 ± 0.041 | 0.046 ± 0.040 | 0.044 ± 0.042 | **0.040 ± 0.035** | 0.043 ± 0.039 | 0.045 ± 0.041 | 0.046 ± 0.039 | 0.046 ± 0.043 | 0.045 ± 0.040 |
| Entire Area | MMD RCS ↓ | 0.138 ± 0.099 | 0.134 ± 0.091 | 0.126 ± 0.093 | 0.120 ± 0.094 | 0.109 ± 0.088 | 0.122 ± 0.102 | 0.263 ± 0.163 | 0.108 ± 0.089 | 0.105 ± 0.091 | **0.104 ± 0.092** | 0.108 ± 0.095 | 0.106 ± 0.095 |
| Entire Area | MMD Doppler ↓ | 0.150 ± 0.333 | 0.144 ± 0.326 | 0.145 ± 0.344 | 0.143 ± 0.333 | 0.145 ± 0.340 | 0.145 ± 0.331 | **0.134 ± 0.312** | 0.145 ± 0.330 | 0.143 ± 0.339 | 0.143 ± 0.329 | 0.144 ± 0.331 | 0.143 ± 0.331 |
| Foreground | CD Loc. ↓ | 0.962 ± 0.624 | 1.055 ± 0.649 | 1.038 ± 0.664 | 1.007 ± 0.623 | 0.980 ± 0.620 | 1.040 ± 0.704 | **0.925 ± 0.549** | 1.092 ± 0.754 | 1.030 ± 0.713 | 0.988 ± 0.640 | 1.013 ± 0.647 | 1.027 ± 0.649 |
| Foreground | CD Full ↓ | **0.057 ± 0.042** | 0.062 ± 0.050 | 0.061 ± 0.046 | 0.062 ± 0.049 | 0.061 ± 0.046 | 0.062 ± 0.047 | 0.060 ± 0.046 | 0.064 ± 0.048 | 0.064 ± 0.051 | 0.059 ± 0.043 | 0.060 ± 0.046 | 0.060 ± 0.039 |
| Foreground | Density Similarity ↑ | 0.429 ± 0.366 | 0.425 ± 0.343 | 0.386 ± 0.354 | 0.444 ± 0.359 | 0.388 ± 0.347 | 0.402 ± 0.344 | **0.445 ± 0.347** | 0.410 ± 0.350 | 0.412 ± 0.362 | 0.395 ± 0.353 | 0.416 ± 0.370 | 0.417 ± 0.359 |
| Foreground | Hit Rate ↑ | 0.751 | 0.799 | 0.732 | 0.770 | 0.740 | 0.766 | **0.814** | 0.758 | 0.740 | 0.717 | 0.714 | 0.736 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.580 ± 296.780 | 1034.140 ± 312.306 | 1007.780 ± 296.091 | 993.280 ± 294.818 | 991.480 ± 303.567 | 972.960 ± 284.059 | 1576.300 ± 420.567 | 986.540 ± 288.663 | 982.020 ± 288.496 | 985.240 ± 282.930 | 981.540 ± 284.519 | 983.700 ± 285.639 |
| Additional | Point-count absolute error ↓ | **149.880 ± 156.189** | 156.560 ± 153.687 | 160.800 ± 147.673 | 162.300 ± 150.207 | 166.580 ± 156.439 | 178.660 ± 163.089 | 485.680 ± 260.657 | 173.120 ± 151.947 | 174.760 ± 154.971 | 173.700 ± 154.490 | 175.080 ± 149.903 | 172.320 ± 147.856 |
| Additional | PPP expected total count | 1027.576 ± 291.684 | 1034.368 ± 306.823 | 1008.856 ± 295.322 | 999.519 ± 291.561 | 993.529 ± 295.456 | 982.411 ± 286.217 | 1571.546 ± 419.999 | 990.532 ± 288.004 | 987.846 ± 287.523 | 990.994 ± 283.705 | 987.674 ± 282.919 | 990.822 ± 283.798 |
| Additional | PPP total-count absolute error ↓ | 149.640 ± 150.669 | **149.597 ± 149.136** | 157.147 ± 148.162 | 159.786 ± 149.367 | 163.430 ± 154.589 | 168.931 ± 159.624 | 481.026 ± 260.719 | 165.681 ± 156.363 | 167.902 ± 152.069 | 166.841 ± 152.686 | 168.479 ± 153.036 | 167.971 ± 150.400 |
| Additional | RCS MAE at target cells ↓ | 4.788 ± 1.755 | **4.759 ± 1.735** | 4.809 ± 1.760 | 4.839 ± 1.780 | 4.884 ± 1.786 | 4.917 ± 1.818 | 4.846 ± 1.841 | 4.949 ± 1.832 | 4.974 ± 1.850 | 4.995 ± 1.859 | 5.013 ± 1.877 | 5.019 ± 1.877 |
| Additional | Doppler MAE at target cells ↓ | 1.757 ± 1.210 | 1.755 ± 1.257 | **1.746 ± 1.224** | 1.781 ± 1.275 | 1.801 ± 1.278 | 1.799 ± 1.297 | 1.815 ± 1.291 | 1.805 ± 1.313 | 1.820 ± 1.339 | 1.818 ± 1.329 | 1.831 ± 1.335 | 1.831 ± 1.342 |
| Additional foreground | PPP expected count per box | 4.819 ± 6.993 | 5.110 ± 7.469 | 4.832 ± 7.213 | 5.021 ± 7.697 | 4.932 ± 7.612 | 4.883 ± 7.576 | 6.938 ± 10.344 | 4.909 ± 7.501 | 4.822 ± 7.542 | 4.744 ± 7.460 | 4.791 ± 7.509 | 4.755 ± 7.502 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.168 ± 7.892 | 5.171 ± 7.700 | 5.348 ± 7.825 | 5.193 ± 7.673 | 5.247 ± 7.825 | 5.211 ± 7.726 | **5.015 ± 7.313** | 5.173 ± 7.833 | 5.209 ± 7.868 | 5.227 ± 7.845 | 5.222 ± 7.864 | 5.264 ± 7.883 |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.767 ± 0.292 | 0.776 ± 0.292 | 0.755 ± 0.303 | 0.755 ± 0.305 | 0.750 ± 0.309 | 0.751 ± 0.306 | **0.807 ± 0.279** | 0.741 ± 0.309 | 0.739 ± 0.313 | 0.734 ± 0.316 | 0.734 ± 0.316 | 0.732 ± 0.316 |
| Foreground (car) | MMD Loc. ↓ | 0.110 | 0.093 | **0.031** | 0.040 | 0.120 | 0.057 | 0.056 | 0.055 | 0.047 | 0.085 | 0.098 | 0.168 |
| Foreground (car) | MMD RCS ↓ | **0.036** | 0.051 | 0.165 | 0.113 | 0.063 | 0.175 | 0.273 | 0.142 | 0.150 | 0.133 | 0.135 | 0.072 |
| Foreground (car) | MMD Doppler ↓ | 0.091 | 0.032 | 0.027 | 0.037 | 0.095 | **0.016** | 0.068 | 0.042 | 0.047 | 0.217 | 0.029 | 0.152 |
| Foreground (truck) | MMD Loc. ↓ | 0.081 | 0.112 | 0.174 | 0.206 | 0.170 | 0.096 | **0.072** | 0.106 | 0.195 | 0.185 | 0.103 | 0.206 |
| Foreground (truck) | MMD RCS ↓ | 0.252 | 0.090 | 0.139 | 0.133 | 0.270 | 0.251 | 0.157 | 0.181 | 0.245 | **0.056** | 0.204 | 0.136 |
| Foreground (truck) | MMD Doppler ↓ | 0.228 | 0.145 | 0.130 | 0.082 | **0.049** | 0.099 | 0.200 | 0.130 | 0.206 | 0.131 | 0.198 | 0.282 |
| Foreground (trailer) | MMD Loc. ↓ | **0.008** | 0.082 | 0.047 | 0.018 | 0.135 | 0.035 | 0.042 | 0.114 | 0.068 | 0.089 | 0.020 | 0.018 |
| Foreground (trailer) | MMD RCS ↓ | 0.469 | 0.776 | 0.770 | 0.708 | 0.403 | 0.643 | 0.515 | 0.298 | 0.390 | **0.233** | 0.501 | 0.502 |
| Foreground (trailer) | MMD Doppler ↓ | 0.395 | 0.517 | 0.565 | 0.525 | 1.114 | 0.501 | **0.384** | 1.117 | 0.826 | 1.026 | 0.525 | 0.531 |
| Runtime | Inference time (s/frame) ↓ | 0.260 ± 0.001 | 0.258 ± 0.001 | 0.259 ± 0.001 | 0.260 ± 0.000 | 0.256 ± 0.001 | 0.256 ± 0.001 | 0.257 ± 0.001 | 0.257 ± 0.001 | 0.258 ± 0.001 | **0.255 ± 0.001** | 0.256 ± 0.001 | 0.258 ± 0.001 |
| Runtime | Median inference time (s/frame) ↓ | 0.260 | 0.258 | 0.259 | 0.260 | 0.256 | 0.256 | 0.257 | 0.257 | 0.258 | **0.255** | 0.255 | 0.259 |

## Sources

- DINO previous four-node final (2684031): [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_45_step_9971.pth`.
- DINO resume step 10000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_010000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_010000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_45_step_10000.pth`.
- DINO resume step 11000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_011000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_011000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_50_step_11000.pth`.
- DINO resume step 12000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_012000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_012000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_55_step_12000.pth`.
- DINO resume step 13000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_013000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_013000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_59_step_13000.pth`.
- DINO resume step 14000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_014000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_014000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_64_step_14000.pth`.
- DINO resume step 15000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_68_step_15000.pth`.
- DINO resume step 16000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_016000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_016000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_73_step_16000.pth`.
- DINO resume step 17000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_017000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_017000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_77_step_17000.pth`.
- DINO resume step 18000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_018000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_018000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_82_step_18000.pth`.
- DINO resume step 19000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_019000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_019000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_87_step_19000.pth`.
- DINO resume step 19914: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2773475_PPP_resume_dino_2684031_optimizer_4nodes/checkpoints/epoch_91_step_19914.pth`.
