# DINO RCS L2 final (2719598): SS50 comparison

All four runs evaluated the same 50 frames in the same order (verified).
Mean ± population standard deviation. ↓ lower is better; ↑ higher is better.
RCS L2 training losses are not compared with earlier L1 losses; these are physical validation metrics.

| Scope | Metric | DINO RCS L2 final (2719598) | 4-node DINO (2684031) | 4-node DiT (2684032) | RadarGen |
| --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.685 ± 0.403 | 1.641 ± 0.392 | 1.700 ± 0.428 | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | 0.038 ± 0.007 | 0.037 ± 0.007 | 0.042 ± 0.010 | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.305 ± 0.113 | 0.316 ± 0.113 | 0.294 ± 0.110 | 0.312 ± 0.105 |
| Entire Area | DA Recall ↑ | 0.280 ± 0.109 | 0.296 ± 0.111 | 0.239 ± 0.111 | 0.302 ± 0.122 |
| Entire Area | DA Precision ↑ | 0.301 ± 0.116 | 0.312 ± 0.113 | 0.279 ± 0.129 | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | 0.287 ± 0.112 | 0.301 ± 0.111 | 0.255 ± 0.119 | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | 0.046 ± 0.047 | 0.049 ± 0.044 | 0.067 ± 0.063 | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | 0.136 ± 0.088 | 0.138 ± 0.099 | 0.268 ± 0.232 | 0.097 ± 0.162 |
| Entire Area | MMD Doppler ↓ | 0.145 ± 0.331 | 0.150 ± 0.333 | 0.274 ± 0.584 | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.027 ± 0.615 | 0.962 ± 0.624 | 1.090 ± 0.768 | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | 0.063 ± 0.044 | 0.057 ± 0.042 | 0.061 ± 0.040 | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.393 ± 0.339 | 0.429 ± 0.366 | 0.366 ± 0.353 | 0.450 ± 0.347 |
| Foreground | Hit Rate ↑ | 0.762 | 0.751 | 0.706 | 0.781 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1014.320 ± 295.757 | 1027.580 ± 296.780 | 949.120 ± 292.826 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | 162.500 ± 149.579 | 149.880 ± 156.189 | 226.340 ± 184.591 | 623.040 ± 441.057 |
| Additional | PPP expected total count | 1016.721 ± 296.091 | 1027.576 ± 291.684 | 947.683 ± 289.272 | — |
| Additional | PPP total-count absolute error ↓ | 154.532 ± 152.374 | 149.640 ± 150.669 | 226.412 ± 184.251 | — |
| Additional | RCS MAE at target cells ↓ | 4.981 ± 1.885 | 4.788 ± 1.755 | 4.704 ± 1.762 | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | 1.811 ± 1.266 | 1.757 ± 1.210 | 2.028 ± 1.380 | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.909 ± 7.409 | 4.819 ± 6.993 | 4.794 ± 8.744 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.269 ± 7.909 | 5.168 ± 7.892 | 6.110 ± 8.745 | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.762 ± 0.296 | 0.767 ± 0.292 | 0.717 ± 0.291 | — |
| Foreground (car) | MMD Loc. ↓ | 0.133 | 0.110 | 0.203 | 0.072 |
| Foreground (car) | MMD RCS ↓ | 0.057 | 0.036 | 0.081 | 0.012 |
| Foreground (car) | MMD Doppler ↓ | 0.116 | 0.091 | 0.127 | 0.046 |
| Foreground (truck) | MMD Loc. ↓ | 0.296 | 0.081 | 0.044 | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.113 | 0.252 | 0.376 | 0.020 |
| Foreground (truck) | MMD Doppler ↓ | 0.084 | 0.228 | 0.247 | 0.108 |
| Foreground (trailer) | MMD Loc. ↓ | 0.037 | 0.008 | 0.074 | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | 0.328 | 0.469 | 0.799 | 0.049 |
| Foreground (trailer) | MMD Doppler ↓ | 0.678 | 0.395 | 0.586 | 0.343 |
| Runtime | Inference time (s/frame) ↓ | 0.251 ± 0.001 | 0.260 ± 0.001 | 0.170 ± 0.002 | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | 0.251 | 0.260 | 0.169 | 1.178 |

Result directories:

- DINO RCS L2 final (2719598): `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2735088_PPP_validation_ss50_2719598`
- 4-node DINO (2684031): `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031`
- 4-node DiT (2684032): `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701616_PPP_validation_ss50_2684032`
- RadarGen: `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation`
