# SS50: four-model comparison

DiT without bottleneck: **2849053 / 28,537**; compact DINO: **2849062 / 10,372**; random DINO: **2847666 / 15,000**; final DINO: **2773475 / 19,914** (last checkpoint, not best-F1 intermediate). Identical 50 scene/frame/seed tuples and order verified. Best unrounded means are bold; descriptive counts and hardware-dependent timing are not ranked. Training histories and compute differ.

## metrics

| Metric | DiT no bottleneck | Compact DINO | Random DINO | Final DINO |
|---|---:|---:|---:|---:|
| chamfer_distance_all_norm | 0.0390 | 0.0427 | 0.0416 | **0.0375** |
| chamfer_distance_xy | **1.6732** | 1.6742 | 1.7388 | 1.6972 |
| count_abs_error | 200.9400 | **155.2000** | 179.7800 | 172.3200 |
| distance_attr_f1 | 0.2681 | 0.2711 | 0.2669 | **0.2895** |
| distance_attr_precision | 0.2892 | 0.2745 | 0.2795 | **0.3072** |
| distance_attr_recall | 0.2563 | 0.2727 | 0.2606 | **0.2782** |
| doppler_mae_at_detections | 2.1927 | **1.7829** | 1.9523 | 1.8310 |
| doppler_mae_at_target_cells | 2.1927 | **1.7829** | 1.9523 | 1.8310 |
| generated_count | 986.1200 | 1087.3000 | 998.5600 | 983.7000 |
| gt_count | 1095.5400 | 1095.5400 | 1095.5400 | 1095.5400 |
| inference_time_seconds | 0.1826 | 0.2656 | 0.6993 | 0.2584 |
| iou_th1 | **0.3127** | 0.2847 | 0.2933 | 0.3063 |
| pcl_mmd_doppler | 0.2847 | 0.1621 | 0.2117 | **0.1430** |
| pcl_mmd_rcs | 0.1862 | 0.3097 | 0.2576 | **0.1061** |
| pcl_mmd_xy | 0.0484 | **0.0425** | 0.0591 | 0.0454 |
| ppp_expected_total_count | 991.9875 | 1086.6916 | 1003.4167 | 990.8224 |
| ppp_total_count_abs_error | 194.4460 | **153.7751** | 175.1338 | 167.9707 |
| rcs_mae_at_detections | 4.9222 | **4.5723** | 4.8640 | 5.0186 |
| rcs_mae_at_target_cells | 4.9222 | **4.5723** | 4.8640 | 5.0186 |
| target_collision_count | 40.3200 | 12.5600 | 12.5600 | 12.5600 |

## foreground

| Metric | DiT no bottleneck | Compact DINO | Random DINO | Final DINO |
|---|---:|---:|---:|---:|
| per_box_chamfer_distance_all_norm | 0.0658 | 0.0618 | 0.0657 | **0.0603** |
| per_box_chamfer_distance_xy | 1.1006 | 1.0995 | 1.0365 | **1.0274** |
| per_box_num_points_similarity | **0.4359** | 0.4255 | 0.4072 | 0.4174 |
| ppp_expected_count | 5.2583 | 4.5853 | 5.0142 | 4.7550 |
| ppp_expected_count_abs_error | **5.1708** | 5.2986 | 5.3055 | 5.2643 |
| ppp_hit_probability_gt_positive | 0.7389 | 0.7663 | **0.7710** | 0.7320 |
| invariant_mmd_xy_car | **0.0582** | 0.1530 | 0.0891 | 0.1682 |
| invariant_mmd_rcs_car | **0.0149** | 0.2035 | 0.0538 | 0.0718 |
| invariant_mmd_doppler_car | 0.0710 | **0.0566** | 0.1048 | 0.1515 |
| invariant_mmd_xy_truck | **0.0473** | 0.0726 | 0.4537 | 0.2057 |
| invariant_mmd_rcs_truck | **0.0861** | 0.3519 | 0.2968 | 0.1360 |
| invariant_mmd_doppler_truck | 0.3324 | 0.3747 | **0.2632** | 0.2822 |
| invariant_mmd_xy_trailer | 0.0871 | 0.0207 | 0.0572 | **0.0176** |
| invariant_mmd_rcs_trailer | **0.1466** | 0.8273 | 0.6246 | 0.5018 |
| invariant_mmd_doppler_trailer | 0.8043 | 0.7979 | 0.7512 | **0.5306** |
| hit_rate | **0.7546** | 0.7472 | 0.7435 | 0.7361 |
| gt_positive_boxes | 269 | 269 | 269 | 269 |
| evaluated_boxes | 299 | 299 | 299 | 299 |

DiT output is 512, DINO output is 1024, explaining differing target-cell collision counts. Foreground geometry metrics have differing valid-box counts; see source summaries. Inference times are not a controlled hardware comparison.

Sources:

- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/summary.json).
- [Summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json).
