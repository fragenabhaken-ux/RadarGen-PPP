# DINO step 15000 versus RadarGen: SS50

DINO optimizer-resume run `2773475`, checkpoint `epoch_68_step_15000.pth`. Both runs completed the same 50 frames in the same order (verified). Mean ± population standard deviation; bold marks better unrounded values; ↓ lower is better, ↑ higher is better. Step 15000 was selected by F1 on this SS50 set, so these are checkpoint-selection results, not independent full-validation results.

| Scope | Metric | DINO step 15000 | RadarGen |
| --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.526 ± 0.321** | 1.685 ± 0.381 |
| Entire Area | CD Full ↓ | **0.037 ± 0.007** | 0.040 ± 0.008 |
| Entire Area | IoU@1m ↑ | **0.336 ± 0.107** | 0.312 ± 0.105 |
| Entire Area | DA Recall ↑ | **0.381 ± 0.128** | 0.302 ± 0.122 |
| Entire Area | DA Precision ↑ | **0.263 ± 0.090** | 0.197 ± 0.099 |
| Entire Area | DA F1 ↑ | **0.309 ± 0.104** | 0.234 ± 0.107 |
| Entire Area | MMD Loc. ↓ | **0.040 ± 0.035** | 0.064 ± 0.058 |
| Entire Area | MMD RCS ↓ | 0.263 ± 0.163 | **0.097 ± 0.162** |
| Entire Area | MMD Doppler ↓ | **0.134 ± 0.312** | 0.357 ± 0.958 |
| Foreground | CD Loc. ↓ | **0.925 ± 0.549** | 1.032 ± 0.713 |
| Foreground | CD Full ↓ | **0.060 ± 0.046** | 0.071 ± 0.051 |
| Foreground | Density Similarity ↑ | 0.445 ± 0.347 | **0.450 ± 0.347** |
| Foreground | Hit Rate ↑ | **0.814** | 0.781 |
| Additional | GT point count | 1095.540 ± 320.639 | 1095.540 ± 320.639 |
| Additional | Generated point count | 1576.300 ± 420.567 | 1717.860 ± 462.256 |
| Additional | Point-count absolute error ↓ | **485.680 ± 260.657** | 623.040 ± 441.057 |
| Additional | PPP expected total count | 1571.546 ± 419.999 | — |
| Additional | PPP total-count absolute error ↓ | 481.026 ± 260.719 | — |
| Additional | RCS MAE at target cells ↓ | **4.846 ± 1.841** | 6.093 ± 1.993 |
| Additional | Doppler MAE at target cells ↓ | **1.815 ± 1.291** | 3.142 ± 1.766 |
| Additional foreground | PPP expected count per box | 6.938 ± 10.344 | — |
| Additional foreground | PPP expected-count absolute error per box ↓ | 5.015 ± 7.313 | — |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | 0.807 ± 0.279 | — |
| Foreground (car) | MMD Loc. ↓ | **0.056** | 0.072 |
| Foreground (car) | MMD RCS ↓ | 0.273 | **0.012** |
| Foreground (car) | MMD Doppler ↓ | 0.068 | **0.046** |
| Foreground (truck) | MMD Loc. ↓ | **0.072** | 0.101 |
| Foreground (truck) | MMD RCS ↓ | 0.157 | **0.020** |
| Foreground (truck) | MMD Doppler ↓ | 0.200 | **0.108** |
| Foreground (trailer) | MMD Loc. ↓ | 0.042 | **0.032** |
| Foreground (trailer) | MMD RCS ↓ | 0.515 | **0.049** |
| Foreground (trailer) | MMD Doppler ↓ | 0.384 | **0.343** |
| Runtime | Inference time (s/frame) ↓ | **0.257 ± 0.001** | 1.185 ± 0.028 |
| Runtime | Median inference time (s/frame) ↓ | **0.257** | 1.178 |

Sources:
- [SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json)
- [2404702_RadarGen_validation/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json)
