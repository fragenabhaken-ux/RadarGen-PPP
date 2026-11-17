# DiT without bottleneck: first versus second final versus RadarGen

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

