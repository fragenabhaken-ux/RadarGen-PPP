# SS50: ResNet first-final versus best-F1 DINO and RadarGen

ResNet **2856174 / 12,575**; DINO **2773475 / 15,000** (best SS50 F1, not final checkpoint); RadarGen released checkpoint. Identical 50 scene/frame/seed tuples and order verified. ResNet second-final/full validation remains in progress; do not confuse these completed SS50 results with partial full-set means. Training histories and budgets differ. Best unrounded means are bold, second-best distinct means underlined. Descriptive counts and uncontrolled timings are unranked.

## metrics

| Metric | ResNet first-final | Best-F1 DINO | RadarGen |
|---|---:|---:|---:|
| chamfer_distance_all_norm | <u>0.0378</u> | **0.0368** | 0.0396 |
| chamfer_distance_xy | <u>1.6062</u> | **1.5256** | 1.6850 |
| count_abs_error | **160.0400** | <u>485.6800</u> | 623.0400 |
| distance_attr_f1 | <u>0.3028</u> | **0.3086** | 0.2344 |
| distance_attr_precision | **0.3157** | <u>0.2631</u> | 0.1967 |
| distance_attr_recall | 0.2955 | **0.3810** | <u>0.3017</u> |
| doppler_mae_at_detections | **1.7636** | <u>1.8152</u> | — |
| doppler_mae_at_target_cells | **1.7636** | <u>1.8152</u> | 3.1415 |
| generated_count | 1026.1400 | 1576.3000 | 1717.8600 |
| gt_count | 1095.5400 | 1095.5400 | 1095.5400 |
| inference_time_seconds | 0.2474 | 0.2573 | 1.1845 |
| iou_th1 | <u>0.3232</u> | **0.3364** | 0.3123 |
| pcl_mmd_doppler | <u>0.1678</u> | **0.1345** | 0.3567 |
| pcl_mmd_rcs | <u>0.1827</u> | 0.2632 | **0.0974** |
| pcl_mmd_xy | <u>0.0442</u> | **0.0399** | 0.0639 |
| ppp_expected_total_count | 1031.8664 | 1571.5458 | — |
| ppp_total_count_abs_error | **155.4333** | <u>481.0261</u> | — |
| rcs_mae_at_detections | **4.7122** | <u>4.8462</u> | — |
| rcs_mae_at_target_cells | **4.7122** | <u>4.8462</u> | 6.0925 |
| target_collision_count | 12.5600 | 12.5600 | — |

## foreground

| Metric | ResNet first-final | Best-F1 DINO | RadarGen |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | **0.0592** | <u>0.0600</u> | 0.0707 |
| per_box_chamfer_distance_xy | <u>1.0007</u> | **0.9245** | 1.0319 |
| per_box_num_points_similarity | 0.4368 | <u>0.4454</u> | **0.4499** |
| ppp_expected_count | 5.5546 | 6.9375 | — |
| ppp_expected_count_abs_error | **4.9726** | <u>5.0153</u> | — |
| ppp_hit_probability_gt_positive | <u>0.7900</u> | **0.8072** | — |
| invariant_mmd_xy_car | 0.0825 | **0.0559** | <u>0.0718</u> |
| invariant_mmd_rcs_car | <u>0.0509</u> | 0.2732 | **0.0124** |
| invariant_mmd_doppler_car | 0.0801 | <u>0.0680</u> | **0.0461** |
| invariant_mmd_xy_truck | **0.0496** | <u>0.0718</u> | 0.1014 |
| invariant_mmd_rcs_truck | 0.2053 | <u>0.1574</u> | **0.0197** |
| invariant_mmd_doppler_truck | <u>0.1947</u> | 0.2002 | **0.1082** |
| invariant_mmd_xy_trailer | **0.0277** | 0.0416 | <u>0.0322</u> |
| invariant_mmd_rcs_trailer | 0.6348 | <u>0.5155</u> | **0.0492** |
| invariant_mmd_doppler_trailer | 0.4691 | <u>0.3839</u> | **0.3430** |
| hit_rate | <u>0.8067</u> | **0.8141** | 0.7807 |
| gt_positive_boxes | 269 | 269 | 269 |
| evaluated_boxes | 299 | 299 | 299 |

Legacy RadarGen target-cell mark MAEs differ from detection-based mark errors. Foreground geometry can use different valid-box denominators. Timings were measured under different hardware/load conditions. Rankings are mean rankings, not significance claims.

Sources:

- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_resnet50_backbone_2856174_step12575/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dino_2773475_checkpoint_sweep/step_015000/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).
