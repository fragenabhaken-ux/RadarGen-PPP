# Selected SS50 model comparison

## Local SS50 results: focused model comparisons

All models evaluated the same 50 validation keyframes in the same order (verified). Mean ± population standard deviation, rounded to three decimals. Except in the explicitly labeled pairwise DINO table, bold marks the best value and underline marks the second-best distinct value within each table, using unrounded means/scalars; exact ties share the same rank. ↓ lower is better; ↑ higher is better; — unavailable. Rows without arrows are not ranked.

Selection rule: “best” means highest whole-area detection F1 among the initial, four-node continuation, and RCS L2 continuation checkpoints for each new model family. This favors balanced detection performance; no model wins every metric. The best previous DiT remains binary-target fine-tune run 2356875, not the continuous-target fine-tune run 2516862.

Selected new DiT: New DiT joint DINO (2658827); selected new DINO: 4-node DINOv3 final (2684031); best new model: 4-node DINOv3 final (2684031). These selections include both RCS L2 continuation checkpoints; the intermediate DINO step-9000 checkpoint is compared separately below.

### DiT: initial run versus four-node continuation

| Scope | Metric | New DiT joint DINO (2658827) | 4-node DiT final (2684032) |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.677 ± 0.427** | <u>1.700 ± 0.428</u> |
| Entire Area | CD Full ↓ | <u>0.043 ± 0.010</u> | **0.042 ± 0.010** |
| Entire Area | IoU@1m ↑ | **0.295 ± 0.106** | <u>0.294 ± 0.110</u> |
| Entire Area | DA Recall ↑ | **0.244 ± 0.111** | <u>0.239 ± 0.111</u> |
| Entire Area | DA Precision ↑ | <u>0.277 ± 0.125</u> | **0.279 ± 0.129** |
| Entire Area | DA F1 ↑ | **0.257 ± 0.117** | <u>0.255 ± 0.119</u> |
| Entire Area | MMD Loc. ↓ | **0.062 ± 0.071** | <u>0.067 ± 0.063</u> |
| Entire Area | MMD RCS ↓ | <u>0.304 ± 0.252</u> | **0.268 ± 0.232** |
| Entire Area | MMD Doppler ↓ | <u>0.275 ± 0.581</u> | **0.274 ± 0.584** |
| Foreground | CD Loc. ↓ | **1.053 ± 0.615** | <u>1.090 ± 0.768</u> |
| Foreground | CD Full ↓ | <u>0.064 ± 0.039</u> | **0.061 ± 0.040** |
| Foreground | Density Similarity ↑ | <u>0.350 ± 0.343</u> | **0.366 ± 0.353** |
| Foreground | Hit Rate ↑ | **0.710** | <u>0.706</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 980.040 ± 318.604 | 949.120 ± 292.826 |
| Additional | Point-count absolute error ↓ | **202.180 ± 194.282** | <u>226.340 ± 184.591</u> |
| Additional | PPP expected total count | 981.545 ± 308.120 | 947.683 ± 289.272 |
| Additional | PPP total-count absolute error ↓ | **202.348 ± 191.148** | <u>226.412 ± 184.251</u> |
| Additional | RCS MAE at target cells ↓ | **4.602 ± 1.666** | <u>4.704 ± 1.762</u> |
| Additional | Doppler MAE at target cells ↓ | <u>2.082 ± 1.352</u> | **2.028 ± 1.380** |
| Additional foreground | PPP expected count per box | 4.456 ± 7.789 | 4.794 ± 8.744 |
| Additional foreground | PPP expected-count absolute error per box ↓ | <u>6.314 ± 8.794</u> | **6.110 ± 8.745** |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.720 ± 0.284** | <u>0.717 ± 0.291</u> |
| Foreground (car) | MMD Loc. ↓ | **0.113** | <u>0.203</u> |
| Foreground (car) | MMD RCS ↓ | **0.035** | <u>0.081</u> |
| Foreground (car) | MMD Doppler ↓ | **0.115** | <u>0.127</u> |
| Foreground (truck) | MMD Loc. ↓ | **0.035** | <u>0.044</u> |
| Foreground (truck) | MMD RCS ↓ | **0.249** | <u>0.376</u> |
| Foreground (truck) | MMD Doppler ↓ | <u>0.472</u> | **0.247** |
| Foreground (trailer) | MMD Loc. ↓ | **0.012** | <u>0.074</u> |
| Foreground (trailer) | MMD RCS ↓ | <u>1.044</u> | **0.799** |
| Foreground (trailer) | MMD Doppler ↓ | <u>0.723</u> | **0.586** |
| Runtime | Inference time (s/frame) ↓ | <u>0.193 ± 0.005</u> | **0.170 ± 0.002** |
| Runtime | Median inference time (s/frame) ↓ | <u>0.191</u> | **0.169** |

### DINO: initial one-shot run versus four-node fine-tuning

| Scope | Metric | New DINOv3 1024 (2658284) | 4-node DINOv3 final (2684031) |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.625 ± 0.358** | <u>1.641 ± 0.392</u> |
| Entire Area | CD Full ↓ | <u>0.039 ± 0.008</u> | **0.037 ± 0.007** |
| Entire Area | IoU@1m ↑ | <u>0.312 ± 0.105</u> | **0.316 ± 0.113** |
| Entire Area | DA Recall ↑ | <u>0.295 ± 0.105</u> | **0.296 ± 0.111** |
| Entire Area | DA Precision ↑ | <u>0.307 ± 0.104</u> | **0.312 ± 0.113** |
| Entire Area | DA F1 ↑ | <u>0.298 ± 0.104</u> | **0.301 ± 0.111** |
| Entire Area | MMD Loc. ↓ | <u>0.049 ± 0.046</u> | **0.049 ± 0.044** |
| Entire Area | MMD RCS ↓ | <u>0.207 ± 0.111</u> | **0.138 ± 0.099** |
| Entire Area | MMD Doppler ↓ | **0.137 ± 0.272** | <u>0.150 ± 0.333</u> |
| Foreground | CD Loc. ↓ | <u>1.002 ± 0.683</u> | **0.962 ± 0.624** |
| Foreground | CD Full ↓ | <u>0.057 ± 0.037</u> | **0.057 ± 0.042** |
| Foreground | Density Similarity ↑ | <u>0.422 ± 0.352</u> | **0.429 ± 0.366** |
| Foreground | Hit Rate ↑ | **0.777** | <u>0.751</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1044.320 ± 310.947 | 1027.580 ± 296.780 |
| Additional | Point-count absolute error ↓ | **141.540 ± 156.163** | <u>149.880 ± 156.189</u> |
| Additional | PPP expected total count | 1047.784 ± 309.060 | 1027.576 ± 291.684 |
| Additional | PPP total-count absolute error ↓ | **141.023 ± 156.404** | <u>149.640 ± 150.669</u> |
| Additional | RCS MAE at target cells ↓ | **4.525 ± 1.607** | <u>4.788 ± 1.755</u> |
| Additional | Doppler MAE at target cells ↓ | **1.668 ± 1.186** | <u>1.757 ± 1.210</u> |
| Additional foreground | PPP expected count per box | 5.158 ± 7.863 | 4.819 ± 6.993 |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.115 ± 7.493** | <u>5.168 ± 7.892</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.783 ± 0.271** | <u>0.767 ± 0.292</u> |
| Foreground (car) | MMD Loc. ↓ | <u>0.163</u> | **0.110** |
| Foreground (car) | MMD RCS ↓ | <u>0.059</u> | **0.036** |
| Foreground (car) | MMD Doppler ↓ | **0.064** | <u>0.091</u> |
| Foreground (truck) | MMD Loc. ↓ | <u>0.152</u> | **0.081** |
| Foreground (truck) | MMD RCS ↓ | **0.248** | <u>0.252</u> |
| Foreground (truck) | MMD Doppler ↓ | **0.121** | <u>0.228</u> |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.071</u> | **0.008** |
| Foreground (trailer) | MMD RCS ↓ | <u>0.864</u> | **0.469** |
| Foreground (trailer) | MMD Doppler ↓ | <u>0.654</u> | **0.395** |
| Runtime | Inference time (s/frame) ↓ | <u>0.260 ± 0.001</u> | **0.260 ± 0.001** |
| Runtime | Median inference time (s/frame) ↓ | <u>0.260</u> | **0.260** |

### DINO: latest checkpoint (step 9971) versus step 9000

| Scope | Metric | 4-node DINOv3 final (2684031) | 4-node DINOv3 step 9000 (2684031) |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.641 ± 0.392** | <u>1.647 ± 0.382</u> |
| Entire Area | CD Full ↓ | <u>0.037 ± 0.007</u> | **0.037 ± 0.007** |
| Entire Area | IoU@1m ↑ | **0.316 ± 0.113** | <u>0.311 ± 0.114</u> |
| Entire Area | DA Recall ↑ | **0.296 ± 0.111** | <u>0.289 ± 0.110</u> |
| Entire Area | DA Precision ↑ | **0.312 ± 0.113** | <u>0.309 ± 0.116</u> |
| Entire Area | DA F1 ↑ | **0.301 ± 0.111** | <u>0.296 ± 0.112</u> |
| Entire Area | MMD Loc. ↓ | <u>0.049 ± 0.044</u> | **0.043 ± 0.042** |
| Entire Area | MMD RCS ↓ | <u>0.138 ± 0.099</u> | **0.130 ± 0.087** |
| Entire Area | MMD Doppler ↓ | <u>0.150 ± 0.333</u> | **0.144 ± 0.329** |
| Foreground | CD Loc. ↓ | **0.962 ± 0.624** | <u>1.026 ± 0.678</u> |
| Foreground | CD Full ↓ | **0.057 ± 0.042** | <u>0.062 ± 0.045</u> |
| Foreground | Density Similarity ↑ | **0.429 ± 0.366** | <u>0.387 ± 0.342</u> |
| Foreground | Hit Rate ↑ | **0.751** | <u>0.732</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.580 ± 296.780 | 1017.360 ± 296.956 |
| Additional | Point-count absolute error ↓ | **149.880 ± 156.189** | <u>154.700 ± 148.833</u> |
| Additional | PPP expected total count | 1027.576 ± 291.684 | 1024.037 ± 299.645 |
| Additional | PPP total-count absolute error ↓ | **149.640 ± 150.669** | <u>149.657 ± 143.806</u> |
| Additional | RCS MAE at target cells ↓ | <u>4.788 ± 1.755</u> | **4.766 ± 1.739** |
| Additional | Doppler MAE at target cells ↓ | <u>1.757 ± 1.210</u> | **1.751 ± 1.247** |
| Additional foreground | PPP expected count per box | 4.819 ± 6.993 | 5.014 ± 7.377 |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.168 ± 7.892** | <u>5.331 ± 7.804</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.767 ± 0.292** | <u>0.766 ± 0.296</u> |
| Foreground (car) | MMD Loc. ↓ | <u>0.110</u> | **0.046** |
| Foreground (car) | MMD RCS ↓ | **0.036** | <u>0.190</u> |
| Foreground (car) | MMD Doppler ↓ | <u>0.091</u> | **0.018** |
| Foreground (truck) | MMD Loc. ↓ | <u>0.081</u> | **0.071** |
| Foreground (truck) | MMD RCS ↓ | **0.252** | <u>0.436</u> |
| Foreground (truck) | MMD Doppler ↓ | <u>0.228</u> | **0.191** |
| Foreground (trailer) | MMD Loc. ↓ | **0.008** | <u>0.063</u> |
| Foreground (trailer) | MMD RCS ↓ | **0.469** | <u>0.591</u> |
| Foreground (trailer) | MMD Doppler ↓ | **0.395** | <u>0.825</u> |
| Runtime | Inference time (s/frame) ↓ | **0.260 ± 0.001** | <u>0.262 ± 0.001</u> |
| Runtime | Median inference time (s/frame) ↓ | **0.260** | <u>0.262</u> |

### DINO step 9000 and final checkpoint versus RadarGen

| Scope | Metric | 4-node DINOv3 step 9000 (2684031) | 4-node DINOv3 final (2684031) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <u>1.647 ± 0.382</u> | **1.641 ± 0.392** | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | **0.037 ± 0.007** | <u>0.037 ± 0.007</u> | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.311 ± 0.114 | **0.316 ± 0.113** | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | 0.289 ± 0.110 | <u>0.296 ± 0.111</u> | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | <u>0.309 ± 0.116</u> | **0.312 ± 0.113** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | <u>0.296 ± 0.112</u> | **0.301 ± 0.111** | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | **0.043 ± 0.042** | <u>0.049 ± 0.044</u> | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | <u>0.130 ± 0.087</u> | 0.138 ± 0.099 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.144 ± 0.329** | <u>0.150 ± 0.333</u> | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | <u>1.026 ± 0.678</u> | **0.962 ± 0.624** | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | <u>0.062 ± 0.045</u> | **0.057 ± 0.042** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.387 ± 0.342 | <u>0.429 ± 0.366</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.732 | <u>0.751</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1017.360 ± 296.956 | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | <u>154.700 ± 148.833</u> | **149.880 ± 156.189** | 623.040 ± 441.057 |
| Additional | PPP expected total count | 1024.037 ± 299.645 | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | <u>149.657 ± 143.806</u> | **149.640 ± 150.669** | — |
| Additional | RCS MAE at target cells ↓ | **4.766 ± 1.739** | <u>4.788 ± 1.755</u> | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | **1.751 ± 1.247** | <u>1.757 ± 1.210</u> | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 5.014 ± 7.377 | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | <u>5.331 ± 7.804</u> | **5.168 ± 7.892** | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | <u>0.766 ± 0.296</u> | **0.767 ± 0.292** | — |
| Foreground (car) | MMD Loc. ↓ | **0.046** | 0.110 | <u>0.072</u> |
| Foreground (car) | MMD RCS ↓ | 0.190 | <u>0.036</u> | **0.012** |
| Foreground (car) | MMD Doppler ↓ | **0.018** | 0.091 | <u>0.046</u> |
| Foreground (truck) | MMD Loc. ↓ | **0.071** | <u>0.081</u> | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.436 | <u>0.252</u> | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | <u>0.191</u> | 0.228 | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | 0.063 | **0.008** | <u>0.032</u> |
| Foreground (trailer) | MMD RCS ↓ | 0.591 | <u>0.469</u> | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 0.825 | <u>0.395</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | <u>0.262 ± 0.001</u> | **0.260 ± 0.001** | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | <u>0.262</u> | **0.260** | 1.178 |

### Final four-node DiT and DINO versus RadarGen

| Scope | Metric | 4-node DiT final (2684032) | 4-node DINOv3 final (2684031) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.700 ± 0.428 | **1.641 ± 0.392** | <u>1.685 ± 0.381</u> |
| Entire Area | CD Full ↓ | 0.042 ± 0.010 | **0.037 ± 0.007** | <u>0.040 ± 0.008</u> |
| Entire Area | IoU@1m ↑ | 0.294 ± 0.110 | **0.316 ± 0.113** | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | 0.239 ± 0.111 | <u>0.296 ± 0.111</u> | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | <u>0.279 ± 0.129</u> | **0.312 ± 0.113** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | <u>0.255 ± 0.119</u> | **0.301 ± 0.111** | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | 0.067 ± 0.063 | **0.049 ± 0.044** | <u>0.064 ± 0.058</u> |
| Entire Area | MMD RCS ↓ | 0.268 ± 0.232 | <u>0.138 ± 0.099</u> | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | <u>0.274 ± 0.584</u> | **0.150 ± 0.333** | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.090 ± 0.768 | **0.962 ± 0.624** | <u>1.032 ± 0.713</u> |
| Foreground | CD Full ↓ | <u>0.061 ± 0.040</u> | **0.057 ± 0.042** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.366 ± 0.353 | <u>0.429 ± 0.366</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.706 | <u>0.751</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 949.120 ± 292.826 | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | <u>226.340 ± 184.591</u> | **149.880 ± 156.189** | 623.040 ± 441.057 |
| Additional | PPP expected total count | 947.683 ± 289.272 | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | <u>226.412 ± 184.251</u> | **149.640 ± 150.669** | — |
| Additional | RCS MAE at target cells ↓ | **4.704 ± 1.762** | <u>4.788 ± 1.755</u> | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | <u>2.028 ± 1.380</u> | **1.757 ± 1.210** | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.794 ± 8.744 | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | <u>6.110 ± 8.745</u> | **5.168 ± 7.892** | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | <u>0.717 ± 0.291</u> | **0.767 ± 0.292** | — |
| Foreground (car) | MMD Loc. ↓ | 0.203 | <u>0.110</u> | **0.072** |
| Foreground (car) | MMD RCS ↓ | 0.081 | <u>0.036</u> | **0.012** |
| Foreground (car) | MMD Doppler ↓ | 0.127 | <u>0.091</u> | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | **0.044** | <u>0.081</u> | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.376 | <u>0.252</u> | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | 0.247 | <u>0.228</u> | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | 0.074 | **0.008** | <u>0.032</u> |
| Foreground (trailer) | MMD RCS ↓ | 0.799 | <u>0.469</u> | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 0.586 | <u>0.395</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | **0.170 ± 0.002** | <u>0.260 ± 0.001</u> | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | **0.169** | <u>0.260</u> | 1.178 |

### DiT: four-node checkpoint versus RCS L2 continuation

| Scope | Metric | 4-node DiT final (2684032) | DiT RCS L2 final (2719596) |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <u>1.700 ± 0.428</u> | **1.695 ± 0.413** |
| Entire Area | CD Full ↓ | <u>0.042 ± 0.010</u> | **0.042 ± 0.009** |
| Entire Area | IoU@1m ↑ | **0.294 ± 0.110** | <u>0.293 ± 0.109</u> |
| Entire Area | DA Recall ↑ | <u>0.239 ± 0.111</u> | **0.244 ± 0.115** |
| Entire Area | DA Precision ↑ | **0.279 ± 0.129** | <u>0.269 ± 0.122</u> |
| Entire Area | DA F1 ↑ | **0.255 ± 0.119** | <u>0.253 ± 0.118</u> |
| Entire Area | MMD Loc. ↓ | <u>0.067 ± 0.063</u> | **0.058 ± 0.046** |
| Entire Area | MMD RCS ↓ | **0.268 ± 0.232** | <u>0.306 ± 0.222</u> |
| Entire Area | MMD Doppler ↓ | <u>0.274 ± 0.584</u> | **0.267 ± 0.576** |
| Foreground | CD Loc. ↓ | **1.090 ± 0.768** | <u>1.126 ± 0.680</u> |
| Foreground | CD Full ↓ | **0.061 ± 0.040** | <u>0.065 ± 0.044</u> |
| Foreground | Density Similarity ↑ | **0.366 ± 0.353** | <u>0.335 ± 0.340</u> |
| Foreground | Hit Rate ↑ | **0.706** | <u>0.699</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 949.120 ± 292.826 | 1000.360 ± 310.022 |
| Additional | Point-count absolute error ↓ | <u>226.340 ± 184.591</u> | **198.140 ± 192.379** |
| Additional | PPP expected total count | 947.683 ± 289.272 | 998.130 ± 304.219 |
| Additional | PPP total-count absolute error ↓ | <u>226.412 ± 184.251</u> | **196.058 ± 185.857** |
| Additional | RCS MAE at target cells ↓ | **4.704 ± 1.762** | <u>4.760 ± 1.817</u> |
| Additional | Doppler MAE at target cells ↓ | **2.028 ± 1.380** | <u>2.053 ± 1.391</u> |
| Additional foreground | PPP expected count per box | 4.794 ± 8.744 | 4.930 ± 9.012 |
| Additional foreground | PPP expected-count absolute error per box ↓ | **6.110 ± 8.745** | <u>6.159 ± 8.834</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | <u>0.717 ± 0.291</u> | **0.721 ± 0.291** |
| Foreground (car) | MMD Loc. ↓ | <u>0.203</u> | **0.111** |
| Foreground (car) | MMD RCS ↓ | <u>0.081</u> | **0.062** |
| Foreground (car) | MMD Doppler ↓ | <u>0.127</u> | **0.101** |
| Foreground (truck) | MMD Loc. ↓ | **0.044** | <u>0.120</u> |
| Foreground (truck) | MMD RCS ↓ | <u>0.376</u> | **0.037** |
| Foreground (truck) | MMD Doppler ↓ | <u>0.247</u> | **0.230** |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.074</u> | **0.013** |
| Foreground (trailer) | MMD RCS ↓ | <u>0.799</u> | **0.393** |
| Foreground (trailer) | MMD Doppler ↓ | **0.586** | <u>1.073</u> |
| Runtime | Inference time (s/frame) ↓ | **0.170 ± 0.002** | <u>0.254 ± 0.016</u> |
| Runtime | Median inference time (s/frame) ↓ | **0.169** | <u>0.257</u> |

### DINO: four-node checkpoint versus RCS L2 continuation

| Scope | Metric | 4-node DINOv3 final (2684031) | DINO RCS L2 final (2719598) |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.641 ± 0.392** | <u>1.685 ± 0.403</u> |
| Entire Area | CD Full ↓ | **0.037 ± 0.007** | <u>0.038 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | **0.316 ± 0.113** | <u>0.305 ± 0.113</u> |
| Entire Area | DA Recall ↑ | **0.296 ± 0.111** | <u>0.280 ± 0.109</u> |
| Entire Area | DA Precision ↑ | **0.312 ± 0.113** | <u>0.301 ± 0.116</u> |
| Entire Area | DA F1 ↑ | **0.301 ± 0.111** | <u>0.287 ± 0.112</u> |
| Entire Area | MMD Loc. ↓ | <u>0.049 ± 0.044</u> | **0.046 ± 0.047** |
| Entire Area | MMD RCS ↓ | <u>0.138 ± 0.099</u> | **0.136 ± 0.088** |
| Entire Area | MMD Doppler ↓ | <u>0.150 ± 0.333</u> | **0.145 ± 0.331** |
| Foreground | CD Loc. ↓ | **0.962 ± 0.624** | <u>1.027 ± 0.615</u> |
| Foreground | CD Full ↓ | **0.057 ± 0.042** | <u>0.063 ± 0.044</u> |
| Foreground | Density Similarity ↑ | **0.429 ± 0.366** | <u>0.393 ± 0.339</u> |
| Foreground | Hit Rate ↑ | <u>0.751</u> | **0.762** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.580 ± 296.780 | 1014.320 ± 295.757 |
| Additional | Point-count absolute error ↓ | **149.880 ± 156.189** | <u>162.500 ± 149.579</u> |
| Additional | PPP expected total count | 1027.576 ± 291.684 | 1016.721 ± 296.091 |
| Additional | PPP total-count absolute error ↓ | **149.640 ± 150.669** | <u>154.532 ± 152.374</u> |
| Additional | RCS MAE at target cells ↓ | **4.788 ± 1.755** | <u>4.981 ± 1.885</u> |
| Additional | Doppler MAE at target cells ↓ | **1.757 ± 1.210** | <u>1.811 ± 1.266</u> |
| Additional foreground | PPP expected count per box | 4.819 ± 6.993 | 4.909 ± 7.409 |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.168 ± 7.892** | <u>5.269 ± 7.909</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.767 ± 0.292** | <u>0.762 ± 0.296</u> |
| Foreground (car) | MMD Loc. ↓ | **0.110** | <u>0.133</u> |
| Foreground (car) | MMD RCS ↓ | **0.036** | <u>0.057</u> |
| Foreground (car) | MMD Doppler ↓ | **0.091** | <u>0.116</u> |
| Foreground (truck) | MMD Loc. ↓ | **0.081** | <u>0.296</u> |
| Foreground (truck) | MMD RCS ↓ | <u>0.252</u> | **0.113** |
| Foreground (truck) | MMD Doppler ↓ | <u>0.228</u> | **0.084** |
| Foreground (trailer) | MMD Loc. ↓ | **0.008** | <u>0.037</u> |
| Foreground (trailer) | MMD RCS ↓ | <u>0.469</u> | **0.328** |
| Foreground (trailer) | MMD Doppler ↓ | **0.395** | <u>0.678</u> |
| Runtime | Inference time (s/frame) ↓ | <u>0.260 ± 0.001</u> | **0.251 ± 0.001** |
| Runtime | Median inference time (s/frame) ↓ | <u>0.260</u> | **0.251** |

### RCS L2 DiT and DINO versus previous four-node models and RadarGen

| Scope | Metric | DiT RCS L2 final (2719596) | DINO RCS L2 final (2719598) | 4-node DiT final (2684032) | 4-node DINOv3 final (2684031) | RadarGen |
| --- | --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.695 ± 0.413 | <u>1.685 ± 0.403</u> | 1.700 ± 0.428 | **1.641 ± 0.392** | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | 0.042 ± 0.009 | <u>0.038 ± 0.007</u> | 0.042 ± 0.010 | **0.037 ± 0.007** | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.293 ± 0.109 | 0.305 ± 0.113 | 0.294 ± 0.110 | **0.316 ± 0.113** | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | 0.244 ± 0.115 | 0.280 ± 0.109 | 0.239 ± 0.111 | <u>0.296 ± 0.111</u> | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | 0.269 ± 0.122 | <u>0.301 ± 0.116</u> | 0.279 ± 0.129 | **0.312 ± 0.113** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | 0.253 ± 0.118 | <u>0.287 ± 0.112</u> | 0.255 ± 0.119 | **0.301 ± 0.111** | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | 0.058 ± 0.046 | **0.046 ± 0.047** | 0.067 ± 0.063 | <u>0.049 ± 0.044</u> | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | 0.306 ± 0.222 | <u>0.136 ± 0.088</u> | 0.268 ± 0.232 | 0.138 ± 0.099 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | 0.267 ± 0.576 | **0.145 ± 0.331** | 0.274 ± 0.584 | <u>0.150 ± 0.333</u> | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.126 ± 0.680 | <u>1.027 ± 0.615</u> | 1.090 ± 0.768 | **0.962 ± 0.624** | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | 0.065 ± 0.044 | 0.063 ± 0.044 | <u>0.061 ± 0.040</u> | **0.057 ± 0.042** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.335 ± 0.340 | 0.393 ± 0.339 | 0.366 ± 0.353 | <u>0.429 ± 0.366</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.699 | <u>0.762</u> | 0.706 | 0.751 | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1000.360 ± 310.022 | 1014.320 ± 295.757 | 949.120 ± 292.826 | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | 198.140 ± 192.379 | <u>162.500 ± 149.579</u> | 226.340 ± 184.591 | **149.880 ± 156.189** | 623.040 ± 441.057 |
| Additional | PPP expected total count | 998.130 ± 304.219 | 1016.721 ± 296.091 | 947.683 ± 289.272 | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | 196.058 ± 185.857 | <u>154.532 ± 152.374</u> | 226.412 ± 184.251 | **149.640 ± 150.669** | — |
| Additional | RCS MAE at target cells ↓ | <u>4.760 ± 1.817</u> | 4.981 ± 1.885 | **4.704 ± 1.762** | 4.788 ± 1.755 | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | 2.053 ± 1.391 | <u>1.811 ± 1.266</u> | 2.028 ± 1.380 | **1.757 ± 1.210** | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.930 ± 9.012 | 4.909 ± 7.409 | 4.794 ± 8.744 | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | 6.159 ± 8.834 | <u>5.269 ± 7.909</u> | 6.110 ± 8.745 | **5.168 ± 7.892** | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.721 ± 0.291 | <u>0.762 ± 0.296</u> | 0.717 ± 0.291 | **0.767 ± 0.292** | — |
| Foreground (car) | MMD Loc. ↓ | 0.111 | 0.133 | 0.203 | <u>0.110</u> | **0.072** |
| Foreground (car) | MMD RCS ↓ | 0.062 | 0.057 | 0.081 | <u>0.036</u> | **0.012** |
| Foreground (car) | MMD Doppler ↓ | 0.101 | 0.116 | 0.127 | <u>0.091</u> | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | 0.120 | 0.296 | **0.044** | <u>0.081</u> | 0.101 |
| Foreground (truck) | MMD RCS ↓ | <u>0.037</u> | 0.113 | 0.376 | 0.252 | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | 0.230 | **0.084** | 0.247 | 0.228 | <u>0.108</u> |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.013</u> | 0.037 | 0.074 | **0.008** | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | 0.393 | <u>0.328</u> | 0.799 | 0.469 | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 1.073 | 0.678 | 0.586 | <u>0.395</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | 0.254 ± 0.016 | <u>0.251 ± 0.001</u> | **0.170 ± 0.002** | 0.260 ± 0.001 | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | 0.257 | <u>0.251</u> | **0.169** | 0.260 | 1.178 |

### Best new DINO and DiT versus previous best DiT and RadarGen

| Scope | Metric | 4-node DINOv3 final (2684031) | New DiT joint DINO (2658827) | Best previous DiT (2356875) | RadarGen |
| --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <u>1.641 ± 0.392</u> | 1.677 ± 0.427 | **1.626 ± 0.345** | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | **0.037 ± 0.007** | 0.043 ± 0.010 | 0.041 ± 0.008 | <u>0.040 ± 0.008</u> |
| Entire Area | IoU@1m ↑ | **0.316 ± 0.113** | 0.295 ± 0.106 | 0.302 ± 0.102 | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | <u>0.296 ± 0.111</u> | 0.244 ± 0.111 | 0.270 ± 0.103 | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | **0.312 ± 0.113** | 0.277 ± 0.125 | <u>0.278 ± 0.111</u> | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | **0.301 ± 0.111** | 0.257 ± 0.117 | <u>0.272 ± 0.106</u> | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | <u>0.049 ± 0.044</u> | 0.062 ± 0.071 | **0.040 ± 0.038** | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | <u>0.138 ± 0.099</u> | 0.304 ± 0.252 | 0.267 ± 0.177 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.150 ± 0.333** | 0.275 ± 0.581 | <u>0.256 ± 0.567</u> | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | **0.962 ± 0.624** | 1.053 ± 0.615 | 1.094 ± 0.634 | <u>1.032 ± 0.713</u> |
| Foreground | CD Full ↓ | **0.057 ± 0.042** | 0.064 ± 0.039 | <u>0.064 ± 0.041</u> | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | <u>0.429 ± 0.366</u> | 0.350 ± 0.343 | 0.394 ± 0.336 | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.751 | 0.710 | <u>0.770</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.580 ± 296.780 | 980.040 ± 318.604 | 1092.400 ± 323.830 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | **149.880 ± 156.189** | 202.180 ± 194.282 | <u>170.340 ± 190.410</u> | 623.040 ± 441.057 |
| Additional | PPP expected total count | 1027.576 ± 291.684 | 981.545 ± 308.120 | 1092.612 ± 322.232 | — |
| Additional | PPP total-count absolute error ↓ | **149.640 ± 150.669** | 202.348 ± 191.148 | <u>171.862 ± 192.052</u> | — |
| Additional | RCS MAE at target cells ↓ | 4.788 ± 1.755 | <u>4.602 ± 1.666</u> | **4.533 ± 1.661** | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | **1.757 ± 1.210** | 2.082 ± 1.352 | <u>1.985 ± 1.197</u> | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.819 ± 6.993 | 4.456 ± 7.789 | 4.456 ± 6.365 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.168 ± 7.892** | 6.314 ± 8.794 | <u>5.319 ± 7.368</u> | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | <u>0.767 ± 0.292</u> | 0.720 ± 0.284 | **0.777 ± 0.259** | — |
| Foreground (car) | MMD Loc. ↓ | <u>0.110</u> | 0.113 | 0.150 | **0.072** |
| Foreground (car) | MMD RCS ↓ | 0.036 | <u>0.035</u> | 0.047 | **0.012** |
| Foreground (car) | MMD Doppler ↓ | 0.091 | 0.115 | <u>0.069</u> | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | <u>0.081</u> | **0.035** | 0.113 | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.252 | 0.249 | <u>0.211</u> | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | 0.228 | 0.472 | <u>0.178</u> | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | **0.008** | <u>0.012</u> | 0.024 | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | <u>0.469</u> | 1.044 | 1.100 | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | <u>0.395</u> | 0.723 | 0.568 | **0.343** |
| Runtime | Inference time (s/frame) ↓ | <u>0.260 ± 0.001</u> | **0.193 ± 0.005** | — | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | <u>0.260</u> | **0.191** | — | 1.178 |

### Best new model versus RadarGen

| Scope | Metric | 4-node DINOv3 final (2684031) | RadarGen |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.641 ± 0.392** | <u>1.685 ± 0.381</u> |
| Entire Area | CD Full ↓ | **0.037 ± 0.007** | <u>0.040 ± 0.008</u> |
| Entire Area | IoU@1m ↑ | **0.316 ± 0.113** | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | <u>0.296 ± 0.111</u> | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | **0.312 ± 0.113** | <u>0.197 ± 0.099</u> |
| Entire Area | DA F1 ↑ | **0.301 ± 0.111** | <u>0.234 ± 0.107</u> |
| Entire Area | MMD Loc. ↓ | **0.049 ± 0.044** | <u>0.064 ± 0.058</u> |
| Entire Area | MMD RCS ↓ | <u>0.138 ± 0.099</u> | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.150 ± 0.333** | <u>0.357 ± 0.958</u> |
| Foreground | CD Loc. ↓ | **0.962 ± 0.624** | <u>1.032 ± 0.713</u> |
| Foreground | CD Full ↓ | **0.057 ± 0.042** | <u>0.071 ± 0.051</u> |
| Foreground | Density Similarity ↑ | <u>0.429 ± 0.366</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | <u>0.751</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | **149.880 ± 156.189** | <u>623.040 ± 441.057</u> |
| Additional | PPP expected total count | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | **149.640 ± 150.669** | — |
| Additional | RCS MAE at target cells ↓ | **4.788 ± 1.755** | <u>6.093 ± 1.993</u> |
| Additional | Doppler MAE at target cells ↓ | **1.757 ± 1.210** | <u>3.142 ± 1.766</u> |
| Additional foreground | PPP expected count per box | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | **5.168 ± 7.892** | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.767 ± 0.292** | — |
| Foreground (car) | MMD Loc. ↓ | <u>0.110</u> | **0.072** |
| Foreground (car) | MMD RCS ↓ | <u>0.036</u> | **0.012** |
| Foreground (car) | MMD Doppler ↓ | <u>0.091</u> | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | **0.081** | <u>0.101</u> |
| Foreground (truck) | MMD RCS ↓ | <u>0.252</u> | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | <u>0.228</u> | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | **0.008** | <u>0.032</u> |
| Foreground (trailer) | MMD RCS ↓ | <u>0.469</u> | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | <u>0.395</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | **0.260 ± 0.001** | <u>1.185 ± 0.028</u> |
| Runtime | Median inference time (s/frame) ↓ | **0.260** | <u>1.178</u> |

### RadarGen versus four-node DINO and DINO RCS L2: independent pairwise comparisons

**Bold** marks the better value between RadarGen and four-node DINO (2684031). <u>Underline</u> marks the better value between RadarGen and DINO RCS L2 (2719598). Each pair is compared independently using unrounded values; exact ties mark both values. A RadarGen value is both bold and underlined when it wins both comparisons. Rows without arrows and pairs with unavailable values are not highlighted.

| Scope | Metric | RadarGen | 4-node DINOv3 final (2684031) | DINO RCS L2 final (2719598) |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.685 ± 0.381 | <strong>1.641 ± 0.392</strong> | <u>1.685 ± 0.403</u> |
| Entire Area | CD Full ↓ | 0.040 ± 0.008 | <strong>0.037 ± 0.007</strong> | <u>0.038 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | <u>0.312 ± 0.105</u> | <strong>0.316 ± 0.113</strong> | 0.305 ± 0.113 |
| Entire Area | DA Recall ↑ | <u><strong>0.302 ± 0.122</strong></u> | 0.296 ± 0.111 | 0.280 ± 0.109 |
| Entire Area | DA Precision ↑ | 0.197 ± 0.099 | <strong>0.312 ± 0.113</strong> | <u>0.301 ± 0.116</u> |
| Entire Area | DA F1 ↑ | 0.234 ± 0.107 | <strong>0.301 ± 0.111</strong> | <u>0.287 ± 0.112</u> |
| Entire Area | MMD Loc. ↓ | 0.064 ± 0.058 | <strong>0.049 ± 0.044</strong> | <u>0.046 ± 0.047</u> |
| Entire Area | MMD RCS ↓ | <u><strong>0.097 ± 0.162</strong></u> | 0.138 ± 0.099 | 0.136 ± 0.088 |
| Entire Area | MMD Doppler ↓ | 0.357 ± 0.958 | <strong>0.150 ± 0.333</strong> | <u>0.145 ± 0.331</u> |
| Foreground | CD Loc. ↓ | 1.032 ± 0.713 | <strong>0.962 ± 0.624</strong> | <u>1.027 ± 0.615</u> |
| Foreground | CD Full ↓ | 0.071 ± 0.051 | <strong>0.057 ± 0.042</strong> | <u>0.063 ± 0.044</u> |
| Foreground | Density Similarity ↑ | <u><strong>0.450 ± 0.347</strong></u> | 0.429 ± 0.366 | 0.393 ± 0.339 |
| Foreground | Hit Rate ↑ | <u><strong>0.781</strong></u> | 0.751 | 0.762 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1717.860 ± 462.256 | 1027.580 ± 296.780 | 1014.320 ± 295.757 |
| Additional | Point-count absolute error ↓ | 623.040 ± 441.057 | <strong>149.880 ± 156.189</strong> | <u>162.500 ± 149.579</u> |
| Additional | PPP expected total count | — | 1027.576 ± 291.684 | 1016.721 ± 296.091 |
| Additional | PPP total-count absolute error ↓ | — | 149.640 ± 150.669 | 154.532 ± 152.374 |
| Additional | RCS MAE at target cells ↓ | 6.093 ± 1.993 | <strong>4.788 ± 1.755</strong> | <u>4.981 ± 1.885</u> |
| Additional | Doppler MAE at target cells ↓ | 3.142 ± 1.766 | <strong>1.757 ± 1.210</strong> | <u>1.811 ± 1.266</u> |
| Additional foreground | PPP expected count per box | — | 4.819 ± 6.993 | 4.909 ± 7.409 |
| Additional foreground | PPP expected-count absolute error per box ↓ | — | 5.168 ± 7.892 | 5.269 ± 7.909 |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | — | 0.767 ± 0.292 | 0.762 ± 0.296 |
| Foreground (car) | MMD Loc. ↓ | <u><strong>0.072</strong></u> | 0.110 | 0.133 |
| Foreground (car) | MMD RCS ↓ | <u><strong>0.012</strong></u> | 0.036 | 0.057 |
| Foreground (car) | MMD Doppler ↓ | <u><strong>0.046</strong></u> | 0.091 | 0.116 |
| Foreground (truck) | MMD Loc. ↓ | <u>0.101</u> | <strong>0.081</strong> | 0.296 |
| Foreground (truck) | MMD RCS ↓ | <u><strong>0.020</strong></u> | 0.252 | 0.113 |
| Foreground (truck) | MMD Doppler ↓ | <strong>0.108</strong> | 0.228 | <u>0.084</u> |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.032</u> | <strong>0.008</strong> | 0.037 |
| Foreground (trailer) | MMD RCS ↓ | <u><strong>0.049</strong></u> | 0.469 | 0.328 |
| Foreground (trailer) | MMD Doppler ↓ | <u><strong>0.343</strong></u> | 0.395 | 0.678 |
| Runtime | Inference time (s/frame) ↓ | 1.185 ± 0.028 | <strong>0.260 ± 0.001</strong> | <u>0.251 ± 0.001</u> |
| Runtime | Median inference time (s/frame) ↓ | 1.178 | <strong>0.260</strong> | <u>0.251</u> |

RCS L2 runs use identity-linked RCS with mean squared error and tenfold lower constant learning rates (DiT 1e-6; DINO 1e-5). These tables compare physical validation metrics, not L1 versus L2 training losses. Archived validation metrics are retained as originally computed; no historical runs were re-evaluated.

The new models use evaluator version 6 and continuous detection targets. RadarGen uses version 5; previous best DiT uses version 3 and binary targets. Mark MAE retains different target-cell versus detection-level semantics. Foreground Chamfer includes only boxes with both clouds nonempty. Previous best DiT inference timing was not recorded. Runtime excludes loading, metrics, and plotting.

Checkpoints: initial DiT `epoch_17_step_18730.pth`; initial DINO `epoch_11_step_10000.pth`; four-node DiT `epoch_64_step_17822.pth` (validation 2701616); four-node DINO latest `epoch_45_step_9971.pth` (validation 2701615), compared with `epoch_41_step_9000.pth` (validation 2701633). RCS L2 DiT `epoch_65_step_18059.pth` (validation 2735087); RCS L2 DINO `epoch_45_step_9979.pth` (validation 2735088). DINO uses a 1024×1024 output grid; new DiT uses 512×512.

Sources:

- RadarGen: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
- Best previous DiT (2356875): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2369104_PPP_validation_finetune_2356875/summary.json).
- New DiT joint DINO (2658827): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680507_PPP_validation_ss50_2658827/summary.json).
- New DINOv3 1024 (2658284): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680332_PPP_validation_ss50_2658284/summary.json).
- 4-node DiT final (2684032): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701616_PPP_validation_ss50_2684032/summary.json).
- 4-node DINOv3 final (2684031): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json).
- 4-node DINOv3 step 9000 (2684031): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701633_PPP_validation_ss50_dino_step9000/summary.json).
- DiT RCS L2 final (2719596): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2735087_PPP_validation_ss50_2719596/summary.json).
- DINO RCS L2 final (2719598): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2735088_PPP_validation_ss50_2719598/summary.json).

## Latest optimizer-resume runs versus four-node runs and RadarGen

Both optimizer-resume final runs evaluated the same 50 frames in the same order as their four-node baselines and RadarGen (verified). Mean ± population standard deviation; bold marks best and underline second-best distinct values using unrounded metrics. All foreground, per-class MMD, count, mark-error, and runtime metrics are included.

### DINO: optimizer-resume final versus four-node versus RadarGen

| Scope | Metric | DINO optimizer resume final (2773475) | 4-node DINO (2684031) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.697 ± 0.411 | **1.641 ± 0.392** | <u>1.685 ± 0.381</u> |
| Entire Area | CD Full ↓ | <u>0.038 ± 0.008</u> | **0.037 ± 0.007** | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | 0.306 ± 0.113 | **0.316 ± 0.113** | <u>0.312 ± 0.105</u> |
| Entire Area | DA Recall ↑ | 0.278 ± 0.111 | <u>0.296 ± 0.111</u> | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | <u>0.307 ± 0.121</u> | **0.312 ± 0.113** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | <u>0.290 ± 0.116</u> | **0.301 ± 0.111** | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | **0.045 ± 0.040** | <u>0.049 ± 0.044</u> | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | <u>0.106 ± 0.095</u> | 0.138 ± 0.099 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.143 ± 0.331** | <u>0.150 ± 0.333</u> | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | <u>1.027 ± 0.649</u> | **0.962 ± 0.624** | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | <u>0.060 ± 0.039</u> | **0.057 ± 0.042** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.417 ± 0.359 | <u>0.429 ± 0.366</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.736 | <u>0.751</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 983.700 ± 285.639 | 1027.580 ± 296.780 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | <u>172.320 ± 147.856</u> | **149.880 ± 156.189** | 623.040 ± 441.057 |
| Additional | PPP expected total count | 990.822 ± 283.798 | 1027.576 ± 291.684 | — |
| Additional | PPP total-count absolute error ↓ | <u>167.971 ± 150.400</u> | **149.640 ± 150.669** | — |
| Additional | RCS MAE at target cells ↓ | <u>5.019 ± 1.877</u> | **4.788 ± 1.755** | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | <u>1.831 ± 1.342</u> | **1.757 ± 1.210** | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.755 ± 7.502 | 4.819 ± 6.993 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | <u>5.264 ± 7.883</u> | **5.168 ± 7.892** | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | <u>0.732 ± 0.316</u> | **0.767 ± 0.292** | — |
| Foreground (car) | MMD Loc. ↓ | 0.168 | <u>0.110</u> | **0.072** |
| Foreground (car) | MMD RCS ↓ | 0.072 | <u>0.036</u> | **0.012** |
| Foreground (car) | MMD Doppler ↓ | 0.152 | <u>0.091</u> | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | 0.206 | **0.081** | <u>0.101</u> |
| Foreground (truck) | MMD RCS ↓ | <u>0.136</u> | 0.252 | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | 0.282 | <u>0.228</u> | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.018</u> | **0.008** | 0.032 |
| Foreground (trailer) | MMD RCS ↓ | 0.502 | <u>0.469</u> | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 0.531 | <u>0.395</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | **0.258 ± 0.001** | <u>0.260 ± 0.001</u> | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | **0.259** | <u>0.260</u> | 1.178 |

Sources: [DINO optimizer resume final (2773475)](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json), [4-node DINO (2684031)](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json), [RadarGen](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).

### DiT: optimizer-resume final versus four-node versus RadarGen

| Scope | Metric | DiT optimizer resume final (2773477) | 4-node DiT (2684032) | RadarGen |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | 1.704 ± 0.460 | <u>1.700 ± 0.428</u> | **1.685 ± 0.381** |
| Entire Area | CD Full ↓ | <u>0.042 ± 0.009</u> | 0.042 ± 0.010 | **0.040 ± 0.008** |
| Entire Area | IoU@1m ↑ | <u>0.298 ± 0.113</u> | 0.294 ± 0.110 | **0.312 ± 0.105** |
| Entire Area | DA Recall ↑ | <u>0.247 ± 0.116</u> | 0.239 ± 0.111 | **0.302 ± 0.122** |
| Entire Area | DA Precision ↑ | <u>0.278 ± 0.127</u> | **0.279 ± 0.129** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | **0.258 ± 0.121** | <u>0.255 ± 0.119</u> | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | <u>0.065 ± 0.063</u> | 0.067 ± 0.063 | **0.064 ± 0.058** |
| Entire Area | MMD RCS ↓ | <u>0.248 ± 0.217</u> | 0.268 ± 0.232 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.269 ± 0.580** | <u>0.274 ± 0.584</u> | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | 1.144 ± 0.774 | <u>1.090 ± 0.768</u> | **1.032 ± 0.713** |
| Foreground | CD Full ↓ | <u>0.063 ± 0.047</u> | **0.061 ± 0.040** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.354 ± 0.359 | <u>0.366 ± 0.353</u> | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | 0.669 | <u>0.706</u> | **0.781** |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 980.380 ± 305.957 | 949.120 ± 292.826 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | **210.760 ± 182.916** | <u>226.340 ± 184.591</u> | 623.040 ± 441.057 |
| Additional | PPP expected total count | 976.778 ± 300.772 | 947.683 ± 289.272 | — |
| Additional | PPP total-count absolute error ↓ | **211.093 ± 179.453** | <u>226.412 ± 184.251</u> | — |
| Additional | RCS MAE at target cells ↓ | <u>4.806 ± 1.826</u> | **4.704 ± 1.762** | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | **2.027 ± 1.386** | <u>2.028 ± 1.380</u> | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 4.821 ± 8.810 | 4.794 ± 8.744 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | **6.091 ± 8.703** | <u>6.110 ± 8.745</u> | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.717 ± 0.295** | <u>0.717 ± 0.291</u> | — |
| Foreground (car) | MMD Loc. ↓ | <u>0.159</u> | 0.203 | **0.072** |
| Foreground (car) | MMD RCS ↓ | <u>0.027</u> | 0.081 | **0.012** |
| Foreground (car) | MMD Doppler ↓ | <u>0.082</u> | 0.127 | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | 0.196 | **0.044** | <u>0.101</u> |
| Foreground (truck) | MMD RCS ↓ | <u>0.097</u> | 0.376 | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | **0.100** | 0.247 | <u>0.108</u> |
| Foreground (trailer) | MMD Loc. ↓ | 0.085 | <u>0.074</u> | **0.032** |
| Foreground (trailer) | MMD RCS ↓ | <u>0.327</u> | 0.799 | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 1.104 | <u>0.586</u> | **0.343** |
| Runtime | Inference time (s/frame) ↓ | <u>0.178 ± 0.004</u> | **0.170 ± 0.002** | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | <u>0.179</u> | **0.169** | 1.178 |

Sources: [DiT optimizer resume final (2773477)](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788376_PPP_validation_ss50_2773477/summary.json), [4-node DiT (2684032)](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701616_PPP_validation_ss50_2684032/summary.json), [RadarGen](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
