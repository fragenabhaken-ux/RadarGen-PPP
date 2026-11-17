# SS50: first-final and second-final ablations versus RadarGen

Best unrounded mean: **bold**. Second-best distinct mean: <u>underlined</u>. All three columns use the same SS50 scene/frame/seed tuples in identical order, verified. Values are means; unavailable metrics are —. Descriptive counts and hardware-dependent timings have no best/worst ranking. Ties use the same formatting; rankings are determined before rounding.

¹ Legacy RadarGen target-cell mark MAEs use different semantics from detection-wise PPP mark errors; their numerical rankings should be interpreted cautiously. Foreground geometry can have different valid-box denominators. ² Inference timing is not a controlled hardware comparison. Training histories and compute differ.

## DiT without bottleneck

First final: **2849053 / 28,537**. Second final: **2867524 / 58,057**.

### Global metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Chamfer, normalized ↓ | <u>0.0390</u> | **0.0383** | 0.0396 |
| Chamfer XY ↓ | **1.6732** | 1.7153 | <u>1.6850</u> |
| Count absolute error ↓ | **200.9400** | <u>219.8400</u> | 623.0400 |
| Detection F1 ↑ | **0.2681** | <u>0.2616</u> | 0.2344 |
| Precision ↑ | **0.2892** | <u>0.2881</u> | 0.1967 |
| Recall ↑ | <u>0.2563</u> | 0.2453 | **0.3017** |
| Doppler MAE at detections ↓ | <u>2.1927</u> | **2.1709** | — |
| Doppler MAE at target cells¹ ↓ | <u>2.1927</u> | **2.1709** | 3.1415 |
| Generated count | 986.1200 | 950.6600 | 1717.8600 |
| GT count | 1095.5400 | 1095.5400 | 1095.5400 |
| Mean inference time (s)² | 0.1826 | 0.1736 | 1.1845 |
| IoU ↑ | **0.3127** | 0.3062 | <u>0.3123</u> |
| MMD Doppler ↓ | <u>0.2847</u> | **0.2843** | 0.3567 |
| MMD RCS ↓ | 0.1862 | <u>0.1578</u> | **0.0974** |
| MMD XY ↓ | **0.0484** | <u>0.0551</u> | 0.0639 |
| PPP expected count | 991.9875 | 952.4641 | — |
| PPP expected-count absolute error ↓ | **194.4460** | <u>213.4580</u> | — |
| RCS MAE at detections ↓ | **4.9222** | <u>5.0367</u> | — |
| RCS MAE at target cells¹ ↓ | **4.9222** | <u>5.0367</u> | 6.0925 |
| Target collisions | 40.3200 | 40.3200 | — |
| Median inference time (s)² | 0.1788 | 0.1726 | 1.1778 |

### Foreground metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Per-box Chamfer, normalized ↓ | <u>0.0658</u> | **0.0571** | 0.0707 |
| Per-box Chamfer XY ↓ | 1.1006 | **0.9755** | <u>1.0319</u> |
| Per-box point-count similarity ↑ | <u>0.4359</u> | 0.4049 | **0.4499** |
| PPP expected count per box | 5.2583 | 5.1144 | — |
| PPP expected-count error per box ↓ | **5.1708** | <u>5.4721</u> | — |
| PPP hit probability, GT-positive ↑ | **0.7389** | <u>0.7099</u> | — |
| MMD xy car ↓ | <u>0.0582</u> | **0.0232** | 0.0718 |
| MMD rcs car ↓ | <u>0.0149</u> | 0.0201 | **0.0124** |
| MMD doppler car ↓ | 0.0710 | **0.0420** | <u>0.0461</u> |
| MMD xy truck ↓ | **0.0473** | <u>0.0867</u> | 0.1014 |
| MMD rcs truck ↓ | 0.0861 | <u>0.0434</u> | **0.0197** |
| MMD doppler truck ↓ | <u>0.3324</u> | 0.6099 | **0.1082** |
| MMD xy trailer ↓ | <u>0.0871</u> | 0.0896 | **0.0322** |
| MMD rcs trailer ↓ | <u>0.1466</u> | 0.1656 | **0.0492** |
| MMD doppler trailer ↓ | 0.8043 | <u>0.7470</u> | **0.3430** |
| Sampled hit rate ↑ | <u>0.7546</u> | 0.6914 | **0.7807** |
| GT-positive boxes | 269 | 269 | 269 |
| Evaluated boxes | 299 | 299 | 299 |

Sources:

- [First-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json).
- [Second-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2867524_step58057_checkpoint_audit/summary.json).
- [RadarGen SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).

## Compact DINO

First final: **2849062 / 10,372**. Second final: **2867528 / 20,595**.

### Global metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Chamfer, normalized ↓ | 0.0427 | <u>0.0415</u> | **0.0396** |
| Chamfer XY ↓ | <u>1.6742</u> | **1.6460** | 1.6850 |
| Count absolute error ↓ | **155.2000** | <u>161.3400</u> | 623.0400 |
| Detection F1 ↑ | <u>0.2711</u> | **0.2827** | 0.2344 |
| Precision ↑ | <u>0.2745</u> | **0.2893** | 0.1967 |
| Recall ↑ | 0.2727 | <u>0.2813</u> | **0.3017** |
| Doppler MAE at detections ↓ | <u>1.7829</u> | **1.7798** | — |
| Doppler MAE at target cells¹ ↓ | <u>1.7829</u> | **1.7798** | 3.1415 |
| Generated count | 1087.3000 | 1070.2800 | 1717.8600 |
| GT count | 1095.5400 | 1095.5400 | 1095.5400 |
| Mean inference time (s)² | 0.2656 | 0.2640 | 1.1845 |
| IoU ↑ | 0.2847 | <u>0.3013</u> | **0.3123** |
| MMD Doppler ↓ | **0.1621** | <u>0.1663</u> | 0.3567 |
| MMD RCS ↓ | 0.3097 | <u>0.2737</u> | **0.0974** |
| MMD XY ↓ | **0.0425** | <u>0.0443</u> | 0.0639 |
| PPP expected count | 1086.6916 | 1072.6043 | — |
| PPP expected-count absolute error ↓ | **153.7751** | <u>164.0588</u> | — |
| RCS MAE at detections ↓ | **4.5723** | <u>4.5994</u> | — |
| RCS MAE at target cells¹ ↓ | **4.5723** | <u>4.5994</u> | 6.0925 |
| Target collisions | 12.5600 | 12.5600 | — |
| Median inference time (s)² | 0.2654 | 0.2640 | 1.1778 |

### Foreground metrics

| Metric | First final | Second final | RadarGen |
|---|---:|---:|---:|
| Per-box Chamfer, normalized ↓ | <u>0.0618</u> | **0.0609** | 0.0707 |
| Per-box Chamfer XY ↓ | <u>1.0995</u> | 1.1012 | **1.0319** |
| Per-box point-count similarity ↑ | <u>0.4255</u> | 0.3990 | **0.4499** |
| PPP expected count per box | 4.5853 | 4.8007 | — |
| PPP expected-count error per box ↓ | <u>5.2986</u> | **5.2748** | — |
| PPP hit probability, GT-positive ↑ | <u>0.7663</u> | **0.7785** | — |
| MMD xy car ↓ | 0.1530 | <u>0.1044</u> | **0.0718** |
| MMD rcs car ↓ | 0.2035 | <u>0.0780</u> | **0.0124** |
| MMD doppler car ↓ | <u>0.0566</u> | 0.0924 | **0.0461** |
| MMD xy truck ↓ | <u>0.0726</u> | **0.0164** | 0.1014 |
| MMD rcs truck ↓ | <u>0.3519</u> | 0.4056 | **0.0197** |
| MMD doppler truck ↓ | 0.3747 | <u>0.2289</u> | **0.1082** |
| MMD xy trailer ↓ | **0.0207** | <u>0.0261</u> | 0.0322 |
| MMD rcs trailer ↓ | <u>0.8273</u> | 1.2662 | **0.0492** |
| MMD doppler trailer ↓ | <u>0.7979</u> | 0.8170 | **0.3430** |
| Sampled hit rate ↑ | 0.7472 | <u>0.7621</u> | **0.7807** |
| GT-positive boxes | 269 | 269 | 269 |
| Evaluated boxes | 299 | 299 | 299 |

Sources:

- [First-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json).
- [Second-final SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2867528_step20595_checkpoint_audit/summary.json).
- [RadarGen SS50](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).

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

