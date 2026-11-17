# SS50: first versus second final checkpoints

Same 50 scene/frame/seed tuples in the same order are required. Best unrounded means are bold. Lower is better except F1/precision/recall/IoU, similarity and hit probabilities/rates. Descriptive counts and timing are not ranked.

## DiT without bottleneck

F1: **0.2681 → 0.2616**; delta -0.0066.

### Paired uncertainty check

Second minus first; 95% percentile intervals from 10,000 paired frame bootstrap resamples (seed 42). Exploratory, without multiple-comparison correction.

| Metric | Mean difference | 95% interval |
|---|---:|---|
| distance_attr_f1 | -0.0066 | [-0.0124, -0.0007] |
| chamfer_distance_xy | +0.0421 | [+0.0144, +0.0686] |
| count_abs_error | +18.9000 | [+3.5795, +34.2405] |
| rcs_mae_at_detections | +0.1145 | [+0.0641, +0.1562] |
| doppler_mae_at_detections | -0.0218 | [-0.0635, +0.0206] |

### metrics

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| chamfer_distance_all_norm | 0.0390 | **0.0383** | -0.0007 |
| chamfer_distance_xy | **1.6732** | 1.7153 | +0.0421 |
| count_abs_error | **200.9400** | 219.8400 | +18.9000 |
| distance_attr_f1 | **0.2681** | 0.2616 | -0.0066 |
| distance_attr_precision | **0.2892** | 0.2881 | -0.0011 |
| distance_attr_recall | **0.2563** | 0.2453 | -0.0110 |
| doppler_mae_at_detections | 2.1927 | **2.1709** | -0.0218 |
| doppler_mae_at_target_cells | 2.1927 | **2.1709** | -0.0218 |
| generated_count | 986.1200 | 950.6600 | -35.4600 |
| gt_count | 1095.5400 | 1095.5400 | +0.0000 |
| inference_time_seconds | 0.1826 | 0.1736 | -0.0090 |
| iou_th1 | **0.3127** | 0.3062 | -0.0065 |
| pcl_mmd_doppler | 0.2847 | **0.2843** | -0.0004 |
| pcl_mmd_rcs | 0.1862 | **0.1578** | -0.0285 |
| pcl_mmd_xy | **0.0484** | 0.0551 | +0.0067 |
| ppp_expected_total_count | 991.9875 | 952.4641 | -39.5234 |
| ppp_total_count_abs_error | **194.4460** | 213.4580 | +19.0120 |
| rcs_mae_at_detections | **4.9222** | 5.0367 | +0.1145 |
| rcs_mae_at_target_cells | **4.9222** | 5.0367 | +0.1145 |
| target_collision_count | 40.3200 | 40.3200 | +0.0000 |

### foreground

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | 0.0658 | **0.0571** | -0.0088 |
| per_box_chamfer_distance_xy | 1.1006 | **0.9755** | -0.1251 |
| per_box_num_points_similarity | **0.4359** | 0.4049 | -0.0311 |
| ppp_expected_count | 5.2583 | 5.1144 | -0.1439 |
| ppp_expected_count_abs_error | **5.1708** | 5.4721 | +0.3012 |
| ppp_hit_probability_gt_positive | **0.7389** | 0.7099 | -0.0290 |
| invariant_mmd_xy_car | 0.0582 | **0.0232** | -0.0349 |
| invariant_mmd_rcs_car | **0.0149** | 0.0201 | +0.0052 |
| invariant_mmd_doppler_car | 0.0710 | **0.0420** | -0.0290 |
| invariant_mmd_xy_truck | **0.0473** | 0.0867 | +0.0394 |
| invariant_mmd_rcs_truck | 0.0861 | **0.0434** | -0.0427 |
| invariant_mmd_doppler_truck | **0.3324** | 0.6099 | +0.2775 |
| invariant_mmd_xy_trailer | **0.0871** | 0.0896 | +0.0026 |
| invariant_mmd_rcs_trailer | **0.1466** | 0.1656 | +0.0190 |
| invariant_mmd_doppler_trailer | 0.8043 | **0.7470** | -0.0573 |
| hit_rate | **0.7546** | 0.6914 | -0.0632 |
| gt_positive_boxes | 269.0000 | 269.0000 | +0.0000 |
| evaluated_boxes | 299.0000 | 299.0000 | +0.0000 |

- [First summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json).
- [Second summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2867524_step58057_checkpoint_audit/summary.json).

## Compact DINO

F1: **0.2711 → 0.2827**; delta +0.0116.

### Paired uncertainty check

Second minus first; 95% percentile intervals from 10,000 paired frame bootstrap resamples (seed 42). Exploratory, without multiple-comparison correction.

| Metric | Mean difference | 95% interval |
|---|---:|---|
| distance_attr_f1 | +0.0116 | [+0.0028, +0.0205] |
| chamfer_distance_xy | -0.0282 | [-0.0568, +0.0003] |
| count_abs_error | +6.1400 | [-17.3405, +29.1805] |
| rcs_mae_at_detections | +0.0271 | [-0.0246, +0.0802] |
| doppler_mae_at_detections | -0.0032 | [-0.0366, +0.0318] |

### metrics

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| chamfer_distance_all_norm | 0.0427 | **0.0415** | -0.0013 |
| chamfer_distance_xy | 1.6742 | **1.6460** | -0.0282 |
| count_abs_error | **155.2000** | 161.3400 | +6.1400 |
| distance_attr_f1 | 0.2711 | **0.2827** | +0.0116 |
| distance_attr_precision | 0.2745 | **0.2893** | +0.0148 |
| distance_attr_recall | 0.2727 | **0.2813** | +0.0087 |
| doppler_mae_at_detections | 1.7829 | **1.7798** | -0.0032 |
| doppler_mae_at_target_cells | 1.7829 | **1.7798** | -0.0032 |
| generated_count | 1087.3000 | 1070.2800 | -17.0200 |
| gt_count | 1095.5400 | 1095.5400 | +0.0000 |
| inference_time_seconds | 0.2656 | 0.2640 | -0.0016 |
| iou_th1 | 0.2847 | **0.3013** | +0.0165 |
| pcl_mmd_doppler | **0.1621** | 0.1663 | +0.0043 |
| pcl_mmd_rcs | 0.3097 | **0.2737** | -0.0360 |
| pcl_mmd_xy | **0.0425** | 0.0443 | +0.0018 |
| ppp_expected_total_count | 1086.6916 | 1072.6043 | -14.0873 |
| ppp_total_count_abs_error | **153.7751** | 164.0588 | +10.2837 |
| rcs_mae_at_detections | **4.5723** | 4.5994 | +0.0271 |
| rcs_mae_at_target_cells | **4.5723** | 4.5994 | +0.0271 |
| target_collision_count | 12.5600 | 12.5600 | +0.0000 |

### foreground

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | 0.0618 | **0.0609** | -0.0009 |
| per_box_chamfer_distance_xy | **1.0995** | 1.1012 | +0.0017 |
| per_box_num_points_similarity | **0.4255** | 0.3990 | -0.0266 |
| ppp_expected_count | 4.5853 | 4.8007 | +0.2154 |
| ppp_expected_count_abs_error | 5.2986 | **5.2748** | -0.0239 |
| ppp_hit_probability_gt_positive | 0.7663 | **0.7785** | +0.0122 |
| invariant_mmd_xy_car | 0.1530 | **0.1044** | -0.0486 |
| invariant_mmd_rcs_car | 0.2035 | **0.0780** | -0.1255 |
| invariant_mmd_doppler_car | **0.0566** | 0.0924 | +0.0358 |
| invariant_mmd_xy_truck | 0.0726 | **0.0164** | -0.0562 |
| invariant_mmd_rcs_truck | **0.3519** | 0.4056 | +0.0537 |
| invariant_mmd_doppler_truck | 0.3747 | **0.2289** | -0.1458 |
| invariant_mmd_xy_trailer | **0.0207** | 0.0261 | +0.0054 |
| invariant_mmd_rcs_trailer | **0.8273** | 1.2662 | +0.4389 |
| invariant_mmd_doppler_trailer | **0.7979** | 0.8170 | +0.0190 |
| hit_rate | 0.7472 | **0.7621** | +0.0149 |
| gt_positive_boxes | 269.0000 | 269.0000 | +0.0000 |
| evaluated_boxes | 299.0000 | 299.0000 | +0.0000 |

- [First summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json).
- [Second summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2867528_step20595_checkpoint_audit/summary.json).

## Random DINO

F1: **0.2669 → 0.2670**; delta +0.0002.

### Paired uncertainty check

Second minus first; 95% percentile intervals from 10,000 paired frame bootstrap resamples (seed 42). Exploratory, without multiple-comparison correction.

| Metric | Mean difference | 95% interval |
|---|---:|---|
| distance_attr_f1 | +0.0002 | [-0.0070, +0.0070] |
| chamfer_distance_xy | +0.0135 | [-0.0199, +0.0456] |
| count_abs_error | +2.3200 | [-19.5800, +23.6200] |
| rcs_mae_at_detections | +0.3712 | [+0.2532, +0.5105] |
| doppler_mae_at_detections | +0.0331 | [-0.0381, +0.0998] |

### metrics

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| chamfer_distance_all_norm | 0.0416 | **0.0405** | -0.0011 |
| chamfer_distance_xy | **1.7388** | 1.7523 | +0.0135 |
| count_abs_error | **179.7800** | 182.1000 | +2.3200 |
| distance_attr_f1 | 0.2669 | **0.2670** | +0.0002 |
| distance_attr_precision | **0.2795** | 0.2773 | -0.0022 |
| distance_attr_recall | 0.2606 | **0.2640** | +0.0034 |
| doppler_mae_at_detections | **1.9523** | 1.9854 | +0.0331 |
| doppler_mae_at_target_cells | **1.9523** | 1.9854 | +0.0331 |
| generated_count | 998.5600 | 1025.3600 | +26.8000 |
| gt_count | 1095.5400 | 1095.5400 | +0.0000 |
| inference_time_seconds | 0.6993 | 0.2654 | -0.4339 |
| iou_th1 | 0.2933 | **0.2937** | +0.0004 |
| pcl_mmd_doppler | 0.2117 | **0.2097** | -0.0020 |
| pcl_mmd_rcs | 0.2576 | **0.1525** | -0.1051 |
| pcl_mmd_xy | 0.0591 | **0.0589** | -0.0002 |
| ppp_expected_total_count | 1003.4167 | 1024.0393 | +20.6226 |
| ppp_total_count_abs_error | **175.1338** | 175.2953 | +0.1615 |
| rcs_mae_at_detections | **4.8640** | 5.2352 | +0.3712 |
| rcs_mae_at_target_cells | **4.8640** | 5.2352 | +0.3712 |
| target_collision_count | 12.5600 | 12.5600 | +0.0000 |

### foreground

| Metric | First final | Second final | Difference |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0657** | 0.0675 | +0.0018 |
| per_box_chamfer_distance_xy | **1.0365** | 1.0689 | +0.0324 |
| per_box_num_points_similarity | **0.4072** | 0.4040 | -0.0032 |
| ppp_expected_count | 5.0142 | 5.0092 | -0.0050 |
| ppp_expected_count_abs_error | 5.3055 | **5.1766** | -0.1288 |
| ppp_hit_probability_gt_positive | **0.7710** | 0.7547 | -0.0163 |
| invariant_mmd_xy_car | **0.0891** | 0.1607 | +0.0716 |
| invariant_mmd_rcs_car | **0.0538** | 0.0665 | +0.0127 |
| invariant_mmd_doppler_car | 0.1048 | **0.0379** | -0.0670 |
| invariant_mmd_xy_truck | 0.4537 | **0.0373** | -0.4164 |
| invariant_mmd_rcs_truck | **0.2968** | 0.3201 | +0.0233 |
| invariant_mmd_doppler_truck | **0.2632** | 0.3149 | +0.0517 |
| invariant_mmd_xy_trailer | 0.0572 | **0.0397** | -0.0175 |
| invariant_mmd_rcs_trailer | **0.6246** | 0.6398 | +0.0152 |
| invariant_mmd_doppler_trailer | 0.7512 | **0.4339** | -0.3173 |
| hit_rate | **0.7435** | 0.7212 | -0.0223 |
| gt_positive_boxes | 269.0000 | 269.0000 | +0.0000 |
| evaluated_boxes | 299.0000 | 299.0000 | +0.0000 |

- [First summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/summary.json).
- [Second summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2863217_step29915_checkpoint_audit/summary.json).

## ResNet-50

First-final SS50 already available at step 12,575. Second-run training was still active when this audit was started; a final-second comparison must wait for that allocation to complete.

