# SS50: DiT without bottleneck and compact DINO versus final DINO

Checkpoints: DiT **2849053 / 28,537**; compact DINO **2849062 / 10,372**; final DINO **2773475 / 19,914**. Identical 50 scene/frame/seed tuples in identical order verified. Final DINO here is the last checkpoint, not its best-F1 intermediate checkpoint at step 15,000.

Best unrounded mean is **bold**. Values are mean ± standard deviation where available; descriptive counts and hardware-dependent timings are not ranked. Model training histories and compute differ. Inference speed is not a controlled hardware comparison.

## Global metrics

| Metric | DiT no bottleneck | Compact DINO | Final DINO |
|---|---:|---:|---:|
| chamfer_distance_all_norm | 0.0390 ± 0.0087 | 0.0427 ± 0.0091 | **0.0375 ± 0.0076** |
| chamfer_distance_xy | **1.6732 ± 0.4408** | 1.6742 ± 0.3373 | 1.6972 ± 0.4109 |
| count_abs_error | 200.9400 ± 176.8336 | **155.2000 ± 160.5002** | 172.3200 ± 147.8555 |
| distance_attr_f1 | 0.2681 ± 0.1297 | 0.2711 ± 0.0994 | **0.2895 ± 0.1157** |
| distance_attr_precision | 0.2892 ± 0.1342 | 0.2745 ± 0.1021 | **0.3072 ± 0.1213** |
| distance_attr_recall | 0.2563 ± 0.1270 | 0.2727 ± 0.0986 | **0.2782 ± 0.1114** |
| doppler_mae_at_detections | 2.1927 ± 1.6217 | **1.7829 ± 1.0498** | 1.8310 ± 1.3418 |
| doppler_mae_at_target_cells | 2.1927 ± 1.6217 | **1.7829 ± 1.0498** | 1.8310 ± 1.3418 |
| generated_count | 986.1200 ± 309.7470 | 1087.3000 ± 299.3163 | 983.7000 ± 285.6385 |
| gt_count | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 |
| inference_time_seconds | 0.1826 ± 0.0101 | 0.2656 ± 0.0014 | 0.2584 ± 0.0008 |
| iou_th1 | **0.3127 ± 0.1126** | 0.2847 ± 0.1026 | 0.3063 ± 0.1125 |
| pcl_mmd_doppler | 0.2847 ± 0.5896 | 0.1621 ± 0.3473 | **0.1430 ± 0.3315** |
| pcl_mmd_rcs | 0.1862 ± 0.1729 | 0.3097 ± 0.1919 | **0.1061 ± 0.0950** |
| pcl_mmd_xy | 0.0484 ± 0.0432 | **0.0425 ± 0.0352** | 0.0454 ± 0.0403 |
| ppp_expected_total_count | 991.9875 ± 312.5200 | 1086.6916 ± 298.3436 | 990.8224 ± 283.7980 |
| ppp_total_count_abs_error | 194.4460 ± 177.0410 | **153.7751 ± 160.3152** | 167.9707 ± 150.4003 |
| rcs_mae_at_detections | 4.9222 ± 1.8962 | **4.5723 ± 1.6264** | 5.0186 ± 1.8774 |
| rcs_mae_at_target_cells | 4.9222 ± 1.8962 | **4.5723 ± 1.6264** | 5.0186 ± 1.8774 |
| target_collision_count | 40.3200 ± 19.1399 | 12.5600 ± 6.7651 | 12.5600 ± 6.7651 |

## Foreground metrics

| Metric | DiT no bottleneck | Compact DINO | Final DINO |
|---|---:|---:|---:|
| per_box_chamfer_distance_all_norm | 0.0658 ± 0.0497 | 0.0618 ± 0.0380 | **0.0603 ± 0.0387** |
| per_box_chamfer_distance_xy | 1.1006 ± 0.7502 | 1.0995 ± 0.6643 | **1.0274 ± 0.6486** |
| per_box_num_points_similarity | **0.4359 ± 0.3660** | 0.4255 ± 0.3639 | 0.4174 ± 0.3586 |
| ppp_expected_count | 5.2583 ± 8.4740 | 4.5853 ± 7.2423 | 4.7550 ± 7.5022 |
| ppp_expected_count_abs_error | **5.1708 ± 7.0894** | 5.2986 ± 7.5010 | 5.2643 ± 7.8830 |
| ppp_hit_probability_gt_positive | 0.7389 ± 0.3255 | **0.7663 ± 0.2766** | 0.7320 ± 0.3159 |
| invariant_mmd_xy_car | **0.0582** | 0.1530 | 0.1682 |
| invariant_mmd_rcs_car | **0.0149** | 0.2035 | 0.0718 |
| invariant_mmd_doppler_car | 0.0710 | **0.0566** | 0.1515 |
| invariant_mmd_xy_truck | **0.0473** | 0.0726 | 0.2057 |
| invariant_mmd_rcs_truck | **0.0861** | 0.3519 | 0.1360 |
| invariant_mmd_doppler_truck | 0.3324 | 0.3747 | **0.2822** |
| invariant_mmd_xy_trailer | 0.0871 | 0.0207 | **0.0176** |
| invariant_mmd_rcs_trailer | **0.1466** | 0.8273 | 0.5018 |
| invariant_mmd_doppler_trailer | 0.8043 | 0.7979 | **0.5306** |
| hit_rate | **0.7546** | 0.7472 | 0.7361 |
| gt_positive_boxes | 269 | 269 | 269 |
| evaluated_boxes | 299 | 299 | 299 |

Foreground geometry uses differing valid-box counts; consult source summaries for denominators. All runs evaluated 299 boxes, including 269 GT-positive boxes. Foreground MMD sampling: Existing evaluator policy: pool nonempty object-local clouds per class and frame; use every third nonempty frame, up to 200 frames per class and side.

Sources:

- [DiT without bottleneck](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dit_dcae_no_bottleneck_2849053_step28537/summary.json).
- [Compact DINO](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_neck_compact_2849062_step10372/summary.json).
- [Final DINO](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json).
