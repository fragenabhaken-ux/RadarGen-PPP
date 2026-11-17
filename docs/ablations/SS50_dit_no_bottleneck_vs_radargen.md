# SS50: DiT without bottleneck versus RadarGen

DiT job 2849053, final step 28,537. Identical 50 scene/frame/seed tuples and order verified. Best unrounded mean is bold; descriptive counts and inference timing are not ranked. Values are mean ± standard deviation where available. Missing values are —. RadarGen target-cell mark MAEs have legacy semantics and should not be interpreted as detection-based MAEs. Inference timing differs in hardware/load conditions.

## metrics

| Metric | DiT without bottleneck | RadarGen |
|---|---:|---:|
| chamfer_distance_all_norm | **0.0390 ± 0.0087** | 0.0396 ± 0.0079 |
| chamfer_distance_xy | **1.6732 ± 0.4408** | 1.6850 ± 0.3808 |
| count_abs_error | **200.9400 ± 176.8336** | 623.0400 ± 441.0573 |
| distance_attr_f1 | **0.2681 ± 0.1297** | 0.2344 ± 0.1069 |
| distance_attr_precision | **0.2892 ± 0.1342** | 0.1967 ± 0.0990 |
| distance_attr_recall | 0.2563 ± 0.1270 | **0.3017 ± 0.1224** |
| doppler_mae_at_detections | 2.1927 ± 1.6217 | — |
| doppler_mae_at_target_cells | **2.1927 ± 1.6217** | 3.1415 ± 1.7657 |
| generated_count | 986.1200 ± 309.7470 | 1717.8600 ± 462.2564 |
| gt_count | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 |
| inference_time_seconds | 0.1826 ± 0.0101 | 1.1845 ± 0.0277 |
| iou_th1 | **0.3127 ± 0.1126** | 0.3123 ± 0.1054 |
| pcl_mmd_doppler | **0.2847 ± 0.5896** | 0.3567 ± 0.9575 |
| pcl_mmd_rcs | 0.1862 ± 0.1729 | **0.0974 ± 0.1618** |
| pcl_mmd_xy | **0.0484 ± 0.0432** | 0.0639 ± 0.0578 |
| ppp_expected_total_count | 991.9875 ± 312.5200 | — |
| ppp_total_count_abs_error | 194.4460 ± 177.0410 | — |
| rcs_mae_at_detections | 4.9222 ± 1.8962 | — |
| rcs_mae_at_target_cells | **4.9222 ± 1.8962** | 6.0925 ± 1.9929 |
| target_collision_count | 40.3200 ± 19.1399 | — |

## foreground

| Metric | DiT without bottleneck | RadarGen |
|---|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0658 ± 0.0497** | 0.0707 ± 0.0507 |
| per_box_chamfer_distance_xy | 1.1006 ± 0.7502 | **1.0319 ± 0.7133** |
| per_box_num_points_similarity | 0.4359 ± 0.3660 | **0.4499 ± 0.3471** |
| ppp_expected_count | 5.2583 ± 8.4740 | — |
| ppp_expected_count_abs_error | 5.1708 ± 7.0894 | — |
| ppp_hit_probability_gt_positive | 0.7389 ± 0.3255 | — |
| invariant_mmd_xy_car | **0.0582** | 0.0718 |
| invariant_mmd_rcs_car | 0.0149 | **0.0124** |
| invariant_mmd_doppler_car | 0.0710 | **0.0461** |
| invariant_mmd_xy_truck | **0.0473** | 0.1014 |
| invariant_mmd_rcs_truck | 0.0861 | **0.0197** |
| invariant_mmd_doppler_truck | 0.3324 | **0.1082** |
| invariant_mmd_xy_trailer | 0.0871 | **0.0322** |
| invariant_mmd_rcs_trailer | 0.1466 | **0.0492** |
| invariant_mmd_doppler_trailer | 0.8043 | **0.3430** |
| hit_rate | 0.7546 | **0.7807** |
| gt_positive_boxes | 269 | 269 |
| evaluated_boxes | 299 | 299 |

Sources:

- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
