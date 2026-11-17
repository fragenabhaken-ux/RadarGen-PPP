# Compact DINO: first versus second final versus RadarGen

Best unrounded mean: **bold**. Second-best distinct mean: <u>underlined</u>. All three columns use the same SS50 scene/frame/seed tuples in identical order, verified. Values are means; unavailable metrics are —. Descriptive counts and hardware-dependent timings have no best/worst ranking. Ties use the same formatting; rankings are determined before rounding.

¹ Legacy RadarGen target-cell mark MAEs use different semantics from detection-wise PPP mark errors; their numerical rankings should be interpreted cautiously. Foreground geometry can have different valid-box denominators. ² Inference timing is not a controlled hardware comparison. Training histories and compute differ.

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

