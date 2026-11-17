# SS50: ResNet-50 versus RadarGen

ResNet job **2856174**, final step **12,575**. Identical 50 scene/frame/seed tuples in identical order verified. Best unrounded mean is bold; descriptive counts and hardware-dependent timing are not ranked. Legacy RadarGen target-cell MAEs differ in semantics from detection-based PPP MAEs. Training histories differ.

## metrics

| Metric | ResNet-50 | RadarGen |
|---|---:|---:|
| chamfer_distance_all_norm | **0.0378** | 0.0396 |
| chamfer_distance_xy | **1.6062** | 1.6850 |
| count_abs_error | **160.0400** | 623.0400 |
| distance_attr_f1 | **0.3028** | 0.2344 |
| distance_attr_precision | **0.3157** | 0.1967 |
| distance_attr_recall | 0.2955 | **0.3017** |
| doppler_mae_at_detections | 1.7636 | — |
| doppler_mae_at_target_cells | **1.7636** | 3.1415 |
| generated_count | 1026.1400 | 1717.8600 |
| gt_count | 1095.5400 | 1095.5400 |
| inference_time_seconds | 0.2474 | 1.1845 |
| iou_th1 | **0.3232** | 0.3123 |
| pcl_mmd_doppler | **0.1678** | 0.3567 |
| pcl_mmd_rcs | 0.1827 | **0.0974** |
| pcl_mmd_xy | **0.0442** | 0.0639 |
| ppp_expected_total_count | 1031.8664 | — |
| ppp_total_count_abs_error | 155.4333 | — |
| rcs_mae_at_detections | 4.7122 | — |
| rcs_mae_at_target_cells | **4.7122** | 6.0925 |
| target_collision_count | 12.5600 | — |

## foreground

| Metric | ResNet-50 | RadarGen |
|---|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0592** | 0.0707 |
| per_box_chamfer_distance_xy | **1.0007** | 1.0319 |
| per_box_num_points_similarity | 0.4368 | **0.4499** |
| ppp_expected_count | 5.5546 | — |
| ppp_expected_count_abs_error | 4.9726 | — |
| ppp_hit_probability_gt_positive | 0.7900 | — |
| invariant_mmd_xy_car | 0.0825 | **0.0718** |
| invariant_mmd_rcs_car | 0.0509 | **0.0124** |
| invariant_mmd_doppler_car | 0.0801 | **0.0461** |
| invariant_mmd_xy_truck | **0.0496** | 0.1014 |
| invariant_mmd_rcs_truck | 0.2053 | **0.0197** |
| invariant_mmd_doppler_truck | 0.1947 | **0.1082** |
| invariant_mmd_xy_trailer | **0.0277** | 0.0322 |
| invariant_mmd_rcs_trailer | 0.6348 | **0.0492** |
| invariant_mmd_doppler_trailer | 0.4691 | **0.3430** |
| hit_rate | **0.8067** | 0.7807 |
| gt_positive_boxes | 269 | 269 |
| evaluated_boxes | 299 | 299 |

Inference median (seconds): ResNet: 0.24706876548589207, RadarGen: 1.1778035850147717. Timings are not a controlled hardware comparison.

Foreground valid-box counts differ for geometry metrics; consult source summaries.

Sources:

- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
