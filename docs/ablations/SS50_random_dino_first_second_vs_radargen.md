# Random DINO: first versus second final versus RadarGen

Best unrounded mean: **bold**. Second-best distinct mean: <u>underlined</u>. All three columns use the same SS50 scene/frame/seed tuples in identical order, verified. Values are means; unavailable metrics are —. Descriptive counts and hardware-dependent timings have no best/worst ranking. Ties use the same formatting; rankings are determined before rounding.

¹ Legacy RadarGen target-cell mark MAEs use different semantics from detection-wise PPP mark errors; their numerical rankings should be interpreted cautiously. Foreground geometry can have different valid-box denominators. ² Inference timing is not a controlled hardware comparison. Training histories and compute differ.

## Random DINO

First final: **2847666 / 15,000**. Second final: **2863217 / 29,915**.

### Global metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Chamfer, normalized ↓ | 0.0416 | <u>0.0405</u> | **0.0396** |
| Chamfer XY ↓ | <u>1.7388</u> | 1.7523 | **1.6850** |
| Count absolute error ↓ | **179.7800** | <u>182.1000</u> | 623.0400 |
| Detection F1 ↑ | <u>0.2669</u> | **0.2670** | 0.2344 |
| Precision ↑ | **0.2795** | <u>0.2773</u> | 0.1967 |
| Recall ↑ | 0.2606 | <u>0.2640</u> | **0.3017** |
| Doppler MAE at detections ↓ | **1.9523** | <u>1.9854</u> | — |
| Doppler MAE at target cells¹ ↓ | **1.9523** | <u>1.9854</u> | 3.1415 |
| Generated count | 998.5600 | 1025.3600 | 1717.8600 |
| GT count | 1095.5400 | 1095.5400 | 1095.5400 |
| Mean inference time (s)² | 0.6993 | 0.2654 | 1.1845 |
| IoU ↑ | 0.2933 | <u>0.2937</u> | **0.3123** |
| MMD Doppler ↓ | <u>0.2117</u> | **0.2097** | 0.3567 |
| MMD RCS ↓ | 0.2576 | <u>0.1525</u> | **0.0974** |
| MMD XY ↓ | <u>0.0591</u> | **0.0589** | 0.0639 |
| PPP expected count | 1003.4167 | 1024.0393 | — |
| PPP expected-count absolute error ↓ | **175.1338** | <u>175.2953</u> | — |
| RCS MAE at detections ↓ | **4.8640** | <u>5.2352</u> | — |
| RCS MAE at target cells¹ ↓ | **4.8640** | <u>5.2352</u> | 6.0925 |
| Target collisions | 12.5600 | 12.5600 | — |
| Median inference time (s)² | 0.6652 | 0.2653 | 1.1778 |

### Foreground metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Per-box Chamfer, normalized ↓ | **0.0657** | <u>0.0675</u> | 0.0707 |
| Per-box Chamfer XY ↓ | <u>1.0365</u> | 1.0689 | **1.0319** |
| Per-box point-count similarity ↑ | <u>0.4072</u> | 0.4040 | **0.4499** |
| PPP expected count per box | 5.0142 | 5.0092 | — |
| PPP expected-count error per box ↓ | <u>5.3055</u> | **5.1766** | — |
| PPP hit probability, GT-positive ↑ | **0.7710** | <u>0.7547</u> | — |
| MMD xy car ↓ | <u>0.0891</u> | 0.1607 | **0.0718** |
| MMD rcs car ↓ | <u>0.0538</u> | 0.0665 | **0.0124** |
| MMD doppler car ↓ | 0.1048 | **0.0379** | <u>0.0461</u> |
| MMD xy truck ↓ | 0.4537 | **0.0373** | <u>0.1014</u> |
| MMD rcs truck ↓ | <u>0.2968</u> | 0.3201 | **0.0197** |
| MMD doppler truck ↓ | <u>0.2632</u> | 0.3149 | **0.1082** |
| MMD xy trailer ↓ | 0.0572 | <u>0.0397</u> | **0.0322** |
| MMD rcs trailer ↓ | <u>0.6246</u> | 0.6398 | **0.0492** |
| MMD doppler trailer ↓ | 0.7512 | <u>0.4339</u> | **0.3430** |
| Sampled hit rate ↑ | <u>0.7435</u> | 0.7212 | **0.7807** |
| GT-positive boxes | 269 | 269 | 269 |
| Evaluated boxes | 299 | 299 | 299 |

Sources:

- [First-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/summary.json).
- [Second-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2863217_step29915_checkpoint_audit/summary.json).
- [RadarGen SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).

