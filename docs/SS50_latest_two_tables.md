# Latest SS50 comparisons

All runs complete; identical 50 frames and evaluation order verified. Mean ± population standard deviation; ↓ lower is better; ↑ higher is better.

## DINO 2773475 versus 4-node DINO and RadarGen

| Scope | Metric | DINO optimizer resume final (2773475) | 4-node DINO (2684031) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.697 ± 0.411 | 1.641 ± 0.392 | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | 0.038 ± 0.008 | 0.037 ± 0.007 | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.306 ± 0.113 | 0.316 ± 0.113 | 0.312 ± 0.105 |
| Entire Area | DA Recall ↑ | 0.278 ± 0.111 | 0.296 ± 0.111 | 0.302 ± 0.122 |
| Entire Area | DA Precision ↑ | 0.307 ± 0.121 | 0.312 ± 0.113 | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | 0.290 ± 0.116 | 0.301 ± 0.111 | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | 0.045 ± 0.040 | 0.049 ± 0.044 | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | 0.106 ± 0.095 | 0.138 ± 0.099 | 0.097 ± 0.162 |
| Entire Area | MMD Doppler ↓ | 0.143 ± 0.331 | 0.150 ± 0.333 | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.027 ± 0.649 | 0.962 ± 0.624 | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | 0.060 ± 0.039 | 0.057 ± 0.042 | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.417 ± 0.359 | 0.429 ± 0.366 | 0.450 ± 0.347 |
| Foreground | Hit Rate ↑ | 0.736 | 0.751 | 0.781 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 983.700 ± 285.639 | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | 172.320 ± 147.856 | 149.880 ± 156.189 | 623.040 ± 441.057 |
| Additional | PPP expected total count | 990.822 ± 283.798 | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | 167.971 ± 150.400 | 149.640 ± 150.669 | — |
| Additional | RCS MAE at target cells ↓ | 5.019 ± 1.877 | 4.788 ± 1.755 | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | 1.831 ± 1.342 | 1.757 ± 1.210 | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.755 ± 7.502 | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.264 ± 7.883 | 5.168 ± 7.892 | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.732 ± 0.316 | 0.767 ± 0.292 | — |
| Foreground (car) | MMD Loc. ↓ | 0.168 | 0.110 | 0.072 |
| Foreground (car) | MMD RCS ↓ | 0.072 | 0.036 | 0.012 |
| Foreground (car) | MMD Doppler ↓ | 0.152 | 0.091 | 0.046 |
| Foreground (truck) | MMD Loc. ↓ | 0.206 | 0.081 | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.136 | 0.252 | 0.020 |
| Foreground (truck) | MMD Doppler ↓ | 0.282 | 0.228 | 0.108 |
| Foreground (trailer) | MMD Loc. ↓ | 0.018 | 0.008 | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | 0.502 | 0.469 | 0.049 |
| Foreground (trailer) | MMD Doppler ↓ | 0.531 | 0.395 | 0.343 |
| Runtime | Inference time (s/frame) ↓ | 0.258 ± 0.001 | 0.260 ± 0.001 | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | 0.259 | 0.260 | 1.178 |

## DiT 2773477 versus 4-node DiT and RadarGen

| Scope | Metric | DIT optimizer resume final (2773477) | 4-node DiT (2684032) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.704 ± 0.460 | 1.700 ± 0.428 | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | 0.042 ± 0.009 | 0.042 ± 0.010 | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.298 ± 0.113 | 0.294 ± 0.110 | 0.312 ± 0.105 |
| Entire Area | DA Recall ↑ | 0.247 ± 0.116 | 0.239 ± 0.111 | 0.302 ± 0.122 |
| Entire Area | DA Precision ↑ | 0.278 ± 0.127 | 0.279 ± 0.129 | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | 0.258 ± 0.121 | 0.255 ± 0.119 | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | 0.065 ± 0.063 | 0.067 ± 0.063 | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | 0.248 ± 0.217 | 0.268 ± 0.232 | 0.097 ± 0.162 |
| Entire Area | MMD Doppler ↓ | 0.269 ± 0.580 | 0.274 ± 0.584 | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.144 ± 0.774 | 1.090 ± 0.768 | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | 0.063 ± 0.047 | 0.061 ± 0.040 | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.354 ± 0.359 | 0.366 ± 0.353 | 0.450 ± 0.347 |
| Foreground | Hit Rate ↑ | 0.669 | 0.706 | 0.781 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 980.380 ± 305.957 | 949.120 ± 292.826 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | 210.760 ± 182.916 | 226.340 ± 184.591 | 623.040 ± 441.057 |
| Additional | PPP expected total count | 976.778 ± 300.772 | 947.683 ± 289.272 | — |
| Additional | PPP total-count absolute error ↓ | 211.093 ± 179.453 | 226.412 ± 184.251 | — |
| Additional | RCS MAE at target cells ↓ | 4.806 ± 1.826 | 4.704 ± 1.762 | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | 2.027 ± 1.386 | 2.028 ± 1.380 | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.821 ± 8.810 | 4.794 ± 8.744 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | 6.091 ± 8.703 | 6.110 ± 8.745 | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.717 ± 0.295 | 0.717 ± 0.291 | — |
| Foreground (car) | MMD Loc. ↓ | 0.159 | 0.203 | 0.072 |
| Foreground (car) | MMD RCS ↓ | 0.027 | 0.081 | 0.012 |
| Foreground (car) | MMD Doppler ↓ | 0.082 | 0.127 | 0.046 |
| Foreground (truck) | MMD Loc. ↓ | 0.196 | 0.044 | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.097 | 0.376 | 0.020 |
| Foreground (truck) | MMD Doppler ↓ | 0.100 | 0.247 | 0.108 |
| Foreground (trailer) | MMD Loc. ↓ | 0.085 | 0.074 | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | 0.327 | 0.799 | 0.049 |
| Foreground (trailer) | MMD Doppler ↓ | 1.104 | 0.586 | 0.343 |
| Runtime | Inference time (s/frame) ↓ | 0.178 ± 0.004 | 0.170 ± 0.002 | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | 0.179 | 0.169 | 1.178 |
