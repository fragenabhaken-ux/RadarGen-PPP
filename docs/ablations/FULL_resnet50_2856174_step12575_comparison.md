# Full validation: ResNet-50, first-run final

Identical annotated-frame identities, order and seeds verified. Best unrounded means are bold; descriptive counts and hardware-dependent timings are not ranked. The 15,000-step DINO reference was selected for best SS50 F1, not necessarily best full-validation F1.

## metrics

| Metric | ResNet-50, first-run final | Final DINO 19,914 | Best-F1 DINO 15,000 (SS50-selected) | RadarGen |
|---|---:|---:|---:|---:|
| chamfer_distance_all_norm | 0.0377 | 0.0370 | **0.0369** | 0.0395 |
| chamfer_distance_xy | 1.5948 | 1.6538 | **1.5062** | 1.6678 |
| count_abs_error | **150.8153** | 166.9984 | 511.4307 | 643.0142 |
| distance_attr_f1 | 0.3116 | 0.3057 | **0.3196** | 0.2387 |
| distance_attr_precision | **0.3242** | 0.3227 | 0.2730 | 0.2012 |
| distance_attr_recall | 0.3053 | 0.2959 | **0.3942** | 0.3051 |
| doppler_mae_at_detections | **1.7121** | 1.7888 | 1.7716 | 3.1704 |
| doppler_mae_at_target_cells | **1.7121** | 1.7888 | 1.7716 | 3.1704 |
| generated_count | 1064.7723 | 1027.3059 | 1622.6123 | 1757.8305 |
| gt_count | 1123.7692 | 1123.7692 | 1123.7692 | 1123.7692 |
| inference_time_seconds | 0.2434 | 0.2569 | 0.2569 | 1.0686 |
| iou_th1 | 0.3348 | 0.3276 | **0.3552** | 0.3224 |
| pcl_mmd_doppler | 0.1236 | **0.1018** | 0.1073 | 0.3114 |
| pcl_mmd_rcs | 0.1661 | 0.1076 | 0.2721 | **0.0867** |
| pcl_mmd_xy | 0.0472 | 0.0495 | **0.0432** | 0.0581 |
| ppp_expected_total_count | 1065.5614 | 1028.1222 | 1622.1120 | — |
| ppp_total_count_abs_error | **148.1983** | 163.7958 | 510.7488 | — |
| rcs_mae_at_detections | **4.7324** | 5.0699 | 4.9027 | 6.0497 |
| rcs_mae_at_target_cells | **4.7324** | 5.0699 | 4.9027 | 6.0497 |
| target_collision_count | 12.3463 | 12.3463 | 12.3463 | 40.6275 |

## foreground

| Metric | ResNet-50, first-run final | Final DINO 19,914 | Best-F1 DINO 15,000 (SS50-selected) | RadarGen |
|---|---:|---:|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0563** | 0.0582 | 0.0582 | 0.0656 |
| per_box_chamfer_distance_xy | 0.9351 | 0.9548 | **0.8999** | 0.9167 |
| per_box_num_points_similarity | 0.4414 | 0.4223 | **0.4637** | 0.4590 |
| ppp_expected_count | 5.8277 | 5.5240 | 7.9661 | — |
| ppp_expected_count_abs_error | **4.9326** | 5.2175 | 5.0787 | — |
| ppp_hit_probability_gt_positive | 0.7889 | 0.7517 | **0.8189** | — |
| invariant_mmd_xy_car | **0.0279** | 0.0341 | 0.0293 | 0.0298 |
| invariant_mmd_rcs_car | 0.0400 | 0.0442 | 0.1637 | **0.0050** |
| invariant_mmd_doppler_car | 0.0136 | 0.0309 | 0.0546 | **0.0122** |
| invariant_mmd_xy_truck | 0.0271 | 0.0202 | **0.0137** | 0.0262 |
| invariant_mmd_rcs_truck | 0.1879 | 0.0471 | 0.2662 | **0.0308** |
| invariant_mmd_doppler_truck | 0.0260 | **0.0084** | 0.0518 | 0.0375 |
| invariant_mmd_xy_trailer | 0.0025 | 0.0024 | **0.0011** | 0.0072 |
| invariant_mmd_rcs_trailer | 0.2411 | 0.0794 | 0.3454 | **0.0379** |
| invariant_mmd_doppler_trailer | 0.0184 | **0.0150** | 0.0394 | 0.0654 |
| hit_rate | 0.7890 | 0.7546 | **0.8208** | 0.7842 |
| gt_positive_boxes | 10874 | 10874 | 10874 | 10874 |
| evaluated_boxes | 11891 | 11891 | 11891 | 11891 |

Legacy RadarGen target-cell MAEs differ from detection-based mark MAEs; foreground geometry denominators can differ. Training histories and compute differ. Timing is not hardware-controlled.

Sources:

- [ResNet-50, first-run final](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_ablation_bundles/resnet50_backbone_2856174_step12575/metrics/summary.json).
- [Final DINO 19,914](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2773475/summary.json).
- [Best-F1 DINO 15,000 (SS50-selected)](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2773475_step15000/summary.json).
- [RadarGen](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/summary.json).
