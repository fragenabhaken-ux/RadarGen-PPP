# Run 2684031: checkpoints with most metric wins versus RadarGen

Steps **3000, 6000, 7000, 8000** tie for the most wins: **17 of 27 comparable metric rows** each. Each checkpoint is compared independently with RadarGen below.

Selection searched all 10 saved checkpoints of run 2684031, excluding the initial-training baseline. Each directional row with values for both models counts equally, using unrounded means/scalars. Rows without a direction and unavailable RadarGen PPP metrics are excluded. Mean and median runtime count as two rows, matching the existing comparison table.

All evaluations contain the same 50 frames in the same order (verified). Selection is on SS50. RadarGen uses evaluator version 5 and DINO version 6; mark MAE retains detection-level versus target-cell semantics. Win counts describe the recorded metrics and do not imply statistical significance.

Mean ± population standard deviation. Bold compares run 2684031 with RadarGen; underline compares step 15000 (run 2773475) with run 2684031. Each pair is ranked independently using unrounded values; exact ties mark both. A run 2684031 value can be both bold and underlined. ↓ lower is better; ↑ higher is better; — unavailable.

## Step 3000 versus RadarGen

Epoch 13; checkpoint `epoch_13_step_3000.pth`. **17 wins, 10 losses, 0 ties**; whole-area DA F1 0.298994.

| Scope | Metric | DINO step 3000 (2684031) | RadarGen | DINO step 15000 (2773475) |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <strong>1.629 ± 0.372</strong> | 1.685 ± 0.381 | <u>1.526 ± 0.321</u> |
| Entire Area | CD Full ↓ | <strong>0.038 ± 0.008</strong> | 0.040 ± 0.008 | <u>0.037 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | <strong>0.315 ± 0.112</strong> | 0.312 ± 0.105 | <u>0.336 ± 0.107</u> |
| Entire Area | DA Recall ↑ | 0.293 ± 0.110 | <strong>0.302 ± 0.122</strong> | <u>0.381 ± 0.128</u> |
| Entire Area | DA Precision ↑ | <u><strong>0.310 ± 0.113</strong></u> | 0.197 ± 0.099 | 0.263 ± 0.090 |
| Entire Area | DA F1 ↑ | <strong>0.299 ± 0.111</strong> | 0.234 ± 0.107 | <u>0.309 ± 0.104</u> |
| Entire Area | MMD Loc. ↓ | <strong>0.045 ± 0.050</strong> | 0.064 ± 0.058 | <u>0.040 ± 0.035</u> |
| Entire Area | MMD RCS ↓ | <u>0.172 ± 0.099</u> | <strong>0.097 ± 0.162</strong> | 0.263 ± 0.163 |
| Entire Area | MMD Doppler ↓ | <strong>0.141 ± 0.323</strong> | 0.357 ± 0.958 | <u>0.134 ± 0.312</u> |
| Foreground | CD Loc. ↓ | 1.034 ± 0.722 | <strong>1.032 ± 0.713</strong> | <u>0.925 ± 0.549</u> |
| Foreground | CD Full ↓ | <u><strong>0.058 ± 0.040</strong></u> | 0.071 ± 0.051 | 0.060 ± 0.046 |
| Foreground | Density Similarity ↑ | 0.433 ± 0.357 | <strong>0.450 ± 0.347</strong> | <u>0.445 ± 0.347</u> |
| Foreground | Hit Rate ↑ | <strong>0.788</strong> | 0.781 | <u>0.814</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1025.740 ± 297.012 | 1717.860 ± 462.256 | 1576.300 ± 420.567 |
| Additional | Point-count absolute error ↓ | <u><strong>146.000 ± 141.210</strong></u> | 623.040 ± 441.057 | 485.680 ± 260.657 |
| Additional | PPP expected total count | 1028.383 ± 294.164 | — | 1571.546 ± 419.999 |
| Additional | PPP total-count absolute error ↓ | <u>148.603 ± 140.502</u> | — | 481.026 ± 260.719 |
| Additional | RCS MAE at target cells ↓ | <u><strong>4.632 ± 1.690</strong></u> | 6.093 ± 1.993 | 4.846 ± 1.841 |
| Additional | Doppler MAE at target cells ↓ | <u><strong>1.690 ± 1.214</strong></u> | 3.142 ± 1.766 | 1.815 ± 1.291 |
| Additional foreground | PPP expected count per box | 5.232 ± 8.267 | — | 6.938 ± 10.344 |
| Additional foreground | PPP expected-count absolute error per box ↓ | <u>4.969 ± 7.578</u> | — | 5.015 ± 7.313 |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.774 ± 0.282 | — | <u>0.807 ± 0.279</u> |
| Foreground (car) | MMD Loc. ↓ | <u><strong>0.028</strong></u> | 0.072 | 0.056 |
| Foreground (car) | MMD RCS ↓ | <u>0.258</u> | <strong>0.012</strong> | 0.273 |
| Foreground (car) | MMD Doppler ↓ | <u><strong>0.026</strong></u> | 0.046 | 0.068 |
| Foreground (truck) | MMD Loc. ↓ | 0.222 | <strong>0.101</strong> | <u>0.072</u> |
| Foreground (truck) | MMD RCS ↓ | 0.420 | <strong>0.020</strong> | <u>0.157</u> |
| Foreground (truck) | MMD Doppler ↓ | <u>0.137</u> | <strong>0.108</strong> | 0.200 |
| Foreground (trailer) | MMD Loc. ↓ | <u><strong>0.029</strong></u> | 0.032 | 0.042 |
| Foreground (trailer) | MMD RCS ↓ | 0.817 | <strong>0.049</strong> | <u>0.515</u> |
| Foreground (trailer) | MMD Doppler ↓ | 0.506 | <strong>0.343</strong> | <u>0.384</u> |
| Runtime | Inference time (s/frame) ↓ | <u><strong>0.257 ± 0.001</strong></u> | 1.185 ± 0.028 | 0.257 ± 0.001 |
| Runtime | Median inference time (s/frame) ↓ | <u><strong>0.257</strong></u> | 1.178 | 0.257 |

## Step 6000 versus RadarGen

Epoch 27; checkpoint `epoch_27_step_6000.pth`. **17 wins, 10 losses, 0 ties**; whole-area DA F1 0.298381.

| Scope | Metric | DINO step 6000 (2684031) | RadarGen | DINO step 15000 (2773475) |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <strong>1.634 ± 0.368</strong> | 1.685 ± 0.381 | <u>1.526 ± 0.321</u> |
| Entire Area | CD Full ↓ | <strong>0.038 ± 0.007</strong> | 0.040 ± 0.008 | <u>0.037 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | <strong>0.313 ± 0.111</strong> | 0.312 ± 0.105 | <u>0.336 ± 0.107</u> |
| Entire Area | DA Recall ↑ | 0.293 ± 0.110 | <strong>0.302 ± 0.122</strong> | <u>0.381 ± 0.128</u> |
| Entire Area | DA Precision ↑ | <u><strong>0.310 ± 0.110</strong></u> | 0.197 ± 0.099 | 0.263 ± 0.090 |
| Entire Area | DA F1 ↑ | <strong>0.298 ± 0.109</strong> | 0.234 ± 0.107 | <u>0.309 ± 0.104</u> |
| Entire Area | MMD Loc. ↓ | <strong>0.045 ± 0.048</strong> | 0.064 ± 0.058 | <u>0.040 ± 0.035</u> |
| Entire Area | MMD RCS ↓ | <u>0.165 ± 0.111</u> | <strong>0.097 ± 0.162</strong> | 0.263 ± 0.163 |
| Entire Area | MMD Doppler ↓ | <strong>0.147 ± 0.325</strong> | 0.357 ± 0.958 | <u>0.134 ± 0.312</u> |
| Foreground | CD Loc. ↓ | <strong>1.000 ± 0.621</strong> | 1.032 ± 0.713 | <u>0.925 ± 0.549</u> |
| Foreground | CD Full ↓ | <strong>0.061 ± 0.046</strong> | 0.071 ± 0.051 | <u>0.060 ± 0.046</u> |
| Foreground | Density Similarity ↑ | 0.420 ± 0.357 | <strong>0.450 ± 0.347</strong> | <u>0.445 ± 0.347</u> |
| Foreground | Hit Rate ↑ | 0.766 | <strong>0.781</strong> | <u>0.814</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1027.280 ± 308.366 | 1717.860 ± 462.256 | 1576.300 ± 420.567 |
| Additional | Point-count absolute error ↓ | <u><strong>158.660 ± 144.889</strong></u> | 623.040 ± 441.057 | 485.680 ± 260.657 |
| Additional | PPP expected total count | 1032.389 ± 302.240 | — | 1571.546 ± 419.999 |
| Additional | PPP total-count absolute error ↓ | <u>152.720 ± 147.730</u> | — | 481.026 ± 260.719 |
| Additional | RCS MAE at target cells ↓ | <u><strong>4.681 ± 1.682</strong></u> | 6.093 ± 1.993 | 4.846 ± 1.841 |
| Additional | Doppler MAE at target cells ↓ | <u><strong>1.756 ± 1.232</strong></u> | 3.142 ± 1.766 | 1.815 ± 1.291 |
| Additional foreground | PPP expected count per box | 5.247 ± 8.034 | — | 6.938 ± 10.344 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.209 ± 7.964 | — | <u>5.015 ± 7.313</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.770 ± 0.291 | — | <u>0.807 ± 0.279</u> |
| Foreground (car) | MMD Loc. ↓ | <strong>0.056</strong> | 0.072 | <u>0.056</u> |
| Foreground (car) | MMD RCS ↓ | <u>0.106</u> | <strong>0.012</strong> | 0.273 |
| Foreground (car) | MMD Doppler ↓ | <u><strong>0.037</strong></u> | 0.046 | 0.068 |
| Foreground (truck) | MMD Loc. ↓ | <strong>0.088</strong> | 0.101 | <u>0.072</u> |
| Foreground (truck) | MMD RCS ↓ | 0.739 | <strong>0.020</strong> | <u>0.157</u> |
| Foreground (truck) | MMD Doppler ↓ | <u>0.137</u> | <strong>0.108</strong> | 0.200 |
| Foreground (trailer) | MMD Loc. ↓ | 0.060 | <strong>0.032</strong> | <u>0.042</u> |
| Foreground (trailer) | MMD RCS ↓ | 1.143 | <strong>0.049</strong> | <u>0.515</u> |
| Foreground (trailer) | MMD Doppler ↓ | 0.522 | <strong>0.343</strong> | <u>0.384</u> |
| Runtime | Inference time (s/frame) ↓ | <u><strong>0.254 ± 0.001</strong></u> | 1.185 ± 0.028 | 0.257 ± 0.001 |
| Runtime | Median inference time (s/frame) ↓ | <u><strong>0.254</strong></u> | 1.178 | 0.257 |

## Step 7000 versus RadarGen

Epoch 32; checkpoint `epoch_32_step_7000.pth`. **17 wins, 10 losses, 0 ties**; whole-area DA F1 0.298027.

| Scope | Metric | DINO step 7000 (2684031) | RadarGen | DINO step 15000 (2773475) |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <strong>1.630 ± 0.352</strong> | 1.685 ± 0.381 | <u>1.526 ± 0.321</u> |
| Entire Area | CD Full ↓ | <strong>0.037 ± 0.007</strong> | 0.040 ± 0.008 | <u>0.037 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | <strong>0.314 ± 0.111</strong> | 0.312 ± 0.105 | <u>0.336 ± 0.107</u> |
| Entire Area | DA Recall ↑ | 0.294 ± 0.106 | <strong>0.302 ± 0.122</strong> | <u>0.381 ± 0.128</u> |
| Entire Area | DA Precision ↑ | <u><strong>0.307 ± 0.110</strong></u> | 0.197 ± 0.099 | 0.263 ± 0.090 |
| Entire Area | DA F1 ↑ | <strong>0.298 ± 0.107</strong> | 0.234 ± 0.107 | <u>0.309 ± 0.104</u> |
| Entire Area | MMD Loc. ↓ | <strong>0.042 ± 0.036</strong> | 0.064 ± 0.058 | <u>0.040 ± 0.035</u> |
| Entire Area | MMD RCS ↓ | <u>0.143 ± 0.097</u> | <strong>0.097 ± 0.162</strong> | 0.263 ± 0.163 |
| Entire Area | MMD Doppler ↓ | <strong>0.149 ± 0.325</strong> | 0.357 ± 0.958 | <u>0.134 ± 0.312</u> |
| Foreground | CD Loc. ↓ | <strong>0.993 ± 0.596</strong> | 1.032 ± 0.713 | <u>0.925 ± 0.549</u> |
| Foreground | CD Full ↓ | <u><strong>0.058 ± 0.046</strong></u> | 0.071 ± 0.051 | 0.060 ± 0.046 |
| Foreground | Density Similarity ↑ | 0.441 ± 0.361 | <strong>0.450 ± 0.347</strong> | <u>0.445 ± 0.347</u> |
| Foreground | Hit Rate ↑ | <strong>0.792</strong> | 0.781 | <u>0.814</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1039.560 ± 286.006 | 1717.860 ± 462.256 | 1576.300 ± 420.567 |
| Additional | Point-count absolute error ↓ | <u><strong>152.500 ± 156.784</strong></u> | 623.040 ± 441.057 | 485.680 ± 260.657 |
| Additional | PPP expected total count | 1039.501 ± 284.454 | — | 1571.546 ± 419.999 |
| Additional | PPP total-count absolute error ↓ | <u>150.637 ± 150.620</u> | — | 481.026 ± 260.719 |
| Additional | RCS MAE at target cells ↓ | <u><strong>4.729 ± 1.717</strong></u> | 6.093 ± 1.993 | 4.846 ± 1.841 |
| Additional | Doppler MAE at target cells ↓ | <u><strong>1.734 ± 1.207</strong></u> | 3.142 ± 1.766 | 1.815 ± 1.291 |
| Additional foreground | PPP expected count per box | 5.249 ± 7.969 | — | 6.938 ± 10.344 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.160 ± 7.768 | — | <u>5.015 ± 7.313</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.778 ± 0.285 | — | <u>0.807 ± 0.279</u> |
| Foreground (car) | MMD Loc. ↓ | 0.122 | <strong>0.072</strong> | <u>0.056</u> |
| Foreground (car) | MMD RCS ↓ | <u>0.029</u> | <strong>0.012</strong> | 0.273 |
| Foreground (car) | MMD Doppler ↓ | <u><strong>0.046</strong></u> | 0.046 | 0.068 |
| Foreground (truck) | MMD Loc. ↓ | <u><strong>0.025</strong></u> | 0.101 | 0.072 |
| Foreground (truck) | MMD RCS ↓ | 0.250 | <strong>0.020</strong> | <u>0.157</u> |
| Foreground (truck) | MMD Doppler ↓ | 0.270 | <strong>0.108</strong> | <u>0.200</u> |
| Foreground (trailer) | MMD Loc. ↓ | <u>0.041</u> | <strong>0.032</strong> | 0.042 |
| Foreground (trailer) | MMD RCS ↓ | 0.808 | <strong>0.049</strong> | <u>0.515</u> |
| Foreground (trailer) | MMD Doppler ↓ | <u>0.372</u> | <strong>0.343</strong> | 0.384 |
| Runtime | Inference time (s/frame) ↓ | <u><strong>0.257 ± 0.001</strong></u> | 1.185 ± 0.028 | 0.257 ± 0.001 |
| Runtime | Median inference time (s/frame) ↓ | <u><strong>0.257</strong></u> | 1.178 | 0.257 |

## Step 8000 versus RadarGen

Epoch 36; checkpoint `epoch_36_step_8000.pth`. **17 wins, 10 losses, 0 ties**; whole-area DA F1 0.300330.

| Scope | Metric | DINO step 8000 (2684031) | RadarGen | DINO step 15000 (2773475) |
| --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | <strong>1.627 ± 0.384</strong> | 1.685 ± 0.381 | <u>1.526 ± 0.321</u> |
| Entire Area | CD Full ↓ | <strong>0.037 ± 0.007</strong> | 0.040 ± 0.008 | <u>0.037 ± 0.007</u> |
| Entire Area | IoU@1m ↑ | <strong>0.318 ± 0.115</strong> | 0.312 ± 0.105 | <u>0.336 ± 0.107</u> |
| Entire Area | DA Recall ↑ | 0.298 ± 0.112 | <strong>0.302 ± 0.122</strong> | <u>0.381 ± 0.128</u> |
| Entire Area | DA Precision ↑ | <u><strong>0.308 ± 0.111</strong></u> | 0.197 ± 0.099 | 0.263 ± 0.090 |
| Entire Area | DA F1 ↑ | <strong>0.300 ± 0.111</strong> | 0.234 ± 0.107 | <u>0.309 ± 0.104</u> |
| Entire Area | MMD Loc. ↓ | <strong>0.047 ± 0.052</strong> | 0.064 ± 0.058 | <u>0.040 ± 0.035</u> |
| Entire Area | MMD RCS ↓ | <u>0.133 ± 0.088</u> | <strong>0.097 ± 0.162</strong> | 0.263 ± 0.163 |
| Entire Area | MMD Doppler ↓ | <strong>0.144 ± 0.336</strong> | 0.357 ± 0.958 | <u>0.134 ± 0.312</u> |
| Foreground | CD Loc. ↓ | <strong>1.001 ± 0.626</strong> | 1.032 ± 0.713 | <u>0.925 ± 0.549</u> |
| Foreground | CD Full ↓ | <strong>0.060 ± 0.045</strong> | 0.071 ± 0.051 | <u>0.060 ± 0.046</u> |
| Foreground | Density Similarity ↑ | 0.415 ± 0.348 | <strong>0.450 ± 0.347</strong> | <u>0.445 ± 0.347</u> |
| Foreground | Hit Rate ↑ | <strong>0.792</strong> | 0.781 | <u>0.814</u> |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1049.660 ± 308.211 | 1717.860 ± 462.256 | 1576.300 ± 420.567 |
| Additional | Point-count absolute error ↓ | <u><strong>150.040 ± 149.880</strong></u> | 623.040 ± 441.057 | 485.680 ± 260.657 |
| Additional | PPP expected total count | 1054.923 ± 309.274 | — | 1571.546 ± 419.999 |
| Additional | PPP total-count absolute error ↓ | <u>142.583 ± 150.867</u> | — | 481.026 ± 260.719 |
| Additional | RCS MAE at target cells ↓ | <u><strong>4.757 ± 1.754</strong></u> | 6.093 ± 1.993 | 4.846 ± 1.841 |
| Additional | Doppler MAE at target cells ↓ | <u><strong>1.772 ± 1.251</strong></u> | 3.142 ± 1.766 | 1.815 ± 1.291 |
| Additional foreground | PPP expected count per box | 5.257 ± 7.958 | — | 6.938 ± 10.344 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.161 ± 7.777 | — | <u>5.015 ± 7.313</u> |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.766 ± 0.295 | — | <u>0.807 ± 0.279</u> |
| Foreground (car) | MMD Loc. ↓ | 0.142 | <strong>0.072</strong> | <u>0.056</u> |
| Foreground (car) | MMD RCS ↓ | <u>0.112</u> | <strong>0.012</strong> | 0.273 |
| Foreground (car) | MMD Doppler ↓ | <u><strong>0.011</strong></u> | 0.046 | 0.068 |
| Foreground (truck) | MMD Loc. ↓ | <strong>0.076</strong> | 0.101 | <u>0.072</u> |
| Foreground (truck) | MMD RCS ↓ | 0.432 | <strong>0.020</strong> | <u>0.157</u> |
| Foreground (truck) | MMD Doppler ↓ | 0.277 | <strong>0.108</strong> | <u>0.200</u> |
| Foreground (trailer) | MMD Loc. ↓ | 0.056 | <strong>0.032</strong> | <u>0.042</u> |
| Foreground (trailer) | MMD RCS ↓ | <u>0.463</u> | <strong>0.049</strong> | 0.515 |
| Foreground (trailer) | MMD Doppler ↓ | 1.034 | <strong>0.343</strong> | <u>0.384</u> |
| Runtime | Inference time (s/frame) ↓ | <strong>0.258 ± 0.001</strong> | 1.185 ± 0.028 | <u>0.257 ± 0.001</u> |
| Runtime | Median inference time (s/frame) ↓ | <strong>0.258</strong> | 1.178 | <u>0.257</u> |

## Sources

- RadarGen: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
- DINO step 15000 (2773475): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json); checkpoint `epoch_68_step_15000.pth`.
- DINO step 3000: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_003000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_13_step_3000.pth`.
- DINO step 6000: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_006000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_27_step_6000.pth`.
- DINO step 7000: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_007000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_32_step_7000.pth`.
- DINO step 8000: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2684031_checkpoint_sweep/step_008000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684031_PPP_continue_dino_2658284_4nodes/checkpoints/epoch_36_step_8000.pth`.
- Selection candidates: [sweep manifest](../evaluation/configs/SS50_dino_2684031_checkpoint_sweep/manifest.json).
