# SS50: ResNet first-final versus final DINO and RadarGen

ResNet **2856174 / 12,575**; final DINO **2773475 / 19,914** (not the best-F1 intermediate checkpoint at 15,000); RadarGen released checkpoint. Identical 50 scene/frame/seed tuples in identical order verified. Best unrounded mean is bold, second-best distinct mean underlined. Descriptive counts and hardware-dependent timings are unranked.

## metrics

| Metric | ResNet first-final | Final DINO | RadarGen |
|---|---:|---:|---:|
| chamfer_distance_all_norm | <u>0.0378</u> | **0.0375** | 0.0396 |
| chamfer_distance_xy | **1.6062** | 1.6972 | <u>1.6850</u> |
| count_abs_error | **160.0400** | <u>172.3200</u> | 623.0400 |
| distance_attr_f1 | **0.3028** | <u>0.2895</u> | 0.2344 |
| distance_attr_precision | **0.3157** | <u>0.3072</u> | 0.1967 |
| distance_attr_recall | <u>0.2955</u> | 0.2782 | **0.3017** |
| doppler_mae_at_detections | **1.7636** | <u>1.8310</u> | — |
| doppler_mae_at_target_cells | **1.7636** | <u>1.8310</u> | 3.1415 |
| generated_count | 1026.1400 | 983.7000 | 1717.8600 |
| gt_count | 1095.5400 | 1095.5400 | 1095.5400 |
| inference_time_seconds | 0.2474 | 0.2584 | 1.1845 |
| iou_th1 | **0.3232** | 0.3063 | <u>0.3123</u> |
| pcl_mmd_doppler | <u>0.1678</u> | **0.1430** | 0.3567 |
| pcl_mmd_rcs | 0.1827 | <u>0.1061</u> | **0.0974** |
| pcl_mmd_xy | **0.0442** | <u>0.0454</u> | 0.0639 |
| ppp_expected_total_count | 1031.8664 | 990.8224 | — |
| ppp_total_count_abs_error | **155.4333** | <u>167.9707</u> | — |
| rcs_mae_at_detections | **4.7122** | <u>5.0186</u> | — |
| rcs_mae_at_target_cells | **4.7122** | <u>5.0186</u> | 6.0925 |
| target_collision_count | 12.5600 | 12.5600 | — |
| median_inference_seconds | 0.2471 | 0.2585 | 1.1778 |

## foreground

| Metric | ResNet first-final | Final DINO | RadarGen |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0592** | <u>0.0603</u> | 0.0707 |
| per_box_chamfer_distance_xy | **1.0007** | <u>1.0274</u> | 1.0319 |
| per_box_num_points_similarity | <u>0.4368</u> | 0.4174 | **0.4499** |
| ppp_expected_count | 5.5546 | 4.7550 | — |
| ppp_expected_count_abs_error | **4.9726** | <u>5.2643</u> | — |
| ppp_hit_probability_gt_positive | **0.7900** | <u>0.7320</u> | — |
| invariant_mmd_xy_car | <u>0.0825</u> | 0.1682 | **0.0718** |
| invariant_mmd_rcs_car | <u>0.0509</u> | 0.0718 | **0.0124** |
| invariant_mmd_doppler_car | <u>0.0801</u> | 0.1515 | **0.0461** |
| invariant_mmd_xy_truck | **0.0496** | 0.2057 | <u>0.1014</u> |
| invariant_mmd_rcs_truck | 0.2053 | <u>0.1360</u> | **0.0197** |
| invariant_mmd_doppler_truck | <u>0.1947</u> | 0.2822 | **0.1082** |
| invariant_mmd_xy_trailer | <u>0.0277</u> | **0.0176** | 0.0322 |
| invariant_mmd_rcs_trailer | 0.6348 | <u>0.5018</u> | **0.0492** |
| invariant_mmd_doppler_trailer | <u>0.4691</u> | 0.5306 | **0.3430** |
| hit_rate | **0.8067** | 0.7361 | <u>0.7807</u> |
| gt_positive_boxes | 269 | 269 | 269 |
| evaluated_boxes | 299 | 299 | 299 |

Legacy RadarGen target-cell MAEs differ from detection-based PPP mark errors. Foreground geometry can have different valid-box counts. Timing is not hardware-controlled. Training histories and compute differ. Rankings are mean rankings, not statistical-significance claims.

Sources:

- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
