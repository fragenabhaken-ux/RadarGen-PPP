# Random-init DINO: final first-allocation SS50 comparison

All four evaluations contain the same 50 scene/frame/seed tuples in the same order (verified). Random DINO uses its final step 15,000; the pretrained DINO final checkpoint is step 19,914, distinct from the best-F1 intermediate step 15,000.

| Metric | Random DINO 15k | Pretrained DINO first 4-node run 9,971 | Pretrained DINO final 19,914 | RadarGen |
|---|---:|---:|---:|---:|
| Detection F1 ↑ | 0.2669 | **0.3011** | 0.2895 | 0.2344 |
| Precision ↑ | 0.2795 | **0.3125** | 0.3072 | 0.1967 |
| Recall ↑ | 0.2606 | 0.2958 | 0.2782 | **0.3017** |
| XY Chamfer ↓ | 1.7388 | **1.6406** | 1.6972 | 1.6850 |
| Count absolute error ↓ | 179.78 | **149.88** | 172.32 | 623.04 |
| MMD XY ↓ | 0.0591 | 0.0488 | **0.0454** | 0.0639 |
| MMD RCS ↓ | 0.2576 | 0.1384 | 0.1061 | **0.0974** |
| MMD Doppler ↓ | 0.2117 | 0.1501 | **0.1430** | 0.3567 |

The pretrained lineage already inherited earlier PPP training: run 2684031 started from run 2658284 step 10,000, whose backbone/LoRA came from an earlier PPP checkpoint. Random DINO started fresh. This compares the available trained models; it does not isolate ImageNet/DINO pretraining at equal total training compute.

The best-F1 pretrained intermediate checkpoint (2773475 step 15,000) has F1 0.3086, XY Chamfer 1.5256, count absolute error 485.68; it is not the final checkpoint.

Sources:

- Random-init DINO, 2847666 / 15,000: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_ablation_dino_random_init_2847666_step15000/summary.json).
- Pretrained DINO, first 4-node run, 2684031 / 9,971: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701615_PPP_validation_ss50_2684031/summary.json).
- Pretrained DINO, final checkpoint, 2773475 / 19,914: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2788375_PPP_validation_ss50_2773475/summary.json).
- RadarGen: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2404702_RadarGen_validation/summary.json).


## All saved metrics

Best mean per metric is **bold**, selected from unrounded values. Descriptive counts and hardware-dependent timings are not ranked.

Values are mean ± standard deviation; scalar foreground metrics have no saved standard deviation. Missing metrics are shown as —. Inference timings come from different hardware/load conditions and are not a controlled speed comparison. “At target cells” is retained separately from “at detections”; the legacy RadarGen values use target-cell semantics.

### Global metrics

| Metric | Random DINO 15k | Pretrained DINO first 12h | Pretrained DINO final | RadarGen |
|---|---:|---:|---:|---:|
| chamfer_distance_all_norm | 0.0416 ± 0.0097 | **0.0375 ± 0.0073** | 0.0375 ± 0.0076 | 0.0396 ± 0.0079 |
| chamfer_distance_xy | 1.7388 ± 0.4467 | **1.6406 ± 0.3918** | 1.6972 ± 0.4109 | 1.6850 ± 0.3808 |
| count_abs_error | 179.7800 ± 178.8280 | **149.8800 ± 156.1895** | 172.3200 ± 147.8555 | 623.0400 ± 441.0573 |
| distance_attr_f1 | 0.2669 ± 0.1149 | **0.3011 ± 0.1113** | 0.2895 ± 0.1157 | 0.2344 ± 0.1069 |
| distance_attr_precision | 0.2795 ± 0.1216 | **0.3125 ± 0.1127** | 0.3072 ± 0.1213 | 0.1967 ± 0.0990 |
| distance_attr_recall | 0.2606 ± 0.1119 | 0.2958 ± 0.1112 | 0.2782 ± 0.1114 | **0.3017 ± 0.1224** |
| doppler_mae_at_detections | 1.9523 ± 1.4034 | **1.7572 ± 1.2101** | 1.8310 ± 1.3418 | — |
| doppler_mae_at_target_cells | 1.9523 ± 1.4034 | **1.7572 ± 1.2101** | 1.8310 ± 1.3418 | 3.1415 ± 1.7657 |
| generated_count | 998.5600 ± 273.1361 | 1027.5800 ± 296.7802 | 983.7000 ± 285.6385 | 1717.8600 ± 462.2564 |
| gt_count | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 | 1095.5400 ± 320.6391 |
| inference_time_seconds | 0.6993 ± 0.1525 | 0.2598 ± 0.0007 | 0.2584 ± 0.0008 | 1.1845 ± 0.0277 |
| iou_th1 | 0.2933 ± 0.1102 | **0.3155 ± 0.1131** | 0.3063 ± 0.1125 | 0.3123 ± 0.1054 |
| pcl_mmd_doppler | 0.2117 ± 0.4208 | 0.1501 ± 0.3329 | **0.1430 ± 0.3315** | 0.3567 ± 0.9575 |
| pcl_mmd_rcs | 0.2576 ± 0.2331 | 0.1384 ± 0.0987 | 0.1061 ± 0.0950 | **0.0974 ± 0.1618** |
| pcl_mmd_xy | 0.0591 ± 0.0503 | 0.0488 ± 0.0440 | **0.0454 ± 0.0403** | 0.0639 ± 0.0578 |
| ppp_expected_total_count | 1003.4167 ± 280.6326 | 1027.5756 ± 291.6845 | 990.8224 ± 283.7980 | — |
| ppp_total_count_abs_error | 175.1338 ± 185.2009 | **149.6401 ± 150.6686** | 167.9707 ± 150.4003 | — |
| rcs_mae_at_detections | 4.8640 ± 1.8816 | **4.7883 ± 1.7547** | 5.0186 ± 1.8774 | — |
| rcs_mae_at_target_cells | 4.8640 ± 1.8816 | **4.7883 ± 1.7547** | 5.0186 ± 1.8774 | 6.0925 ± 1.9929 |
| target_collision_count | 12.5600 ± 6.7651 | 12.5600 ± 6.7651 | 12.5600 ± 6.7651 | — |

Inference median (seconds): Random DINO: 0.6651908619969618; First 12h: 0.25994129599712323; Final: 0.25851011101622134; RadarGen: 1.1778035850147717.

### Foreground metrics

| Metric | Random DINO 15k | Pretrained DINO first 12h | Pretrained DINO final | RadarGen |
|---|---:|---:|---:|---:|
| per_box_chamfer_distance_all_norm | 0.0657 ± 0.0510 | **0.0568 ± 0.0416** | 0.0603 ± 0.0387 | 0.0707 ± 0.0507 |
| per_box_chamfer_distance_xy | 1.0365 ± 0.5970 | **0.9621 ± 0.6240** | 1.0274 ± 0.6486 | 1.0319 ± 0.7133 |
| per_box_num_points_similarity | 0.4072 ± 0.3599 | 0.4294 ± 0.3660 | 0.4174 ± 0.3586 | **0.4499 ± 0.3471** |
| ppp_expected_count | 5.0142 ± 8.4490 | 4.8187 ± 6.9934 | 4.7550 ± 7.5022 | — |
| ppp_expected_count_abs_error | 5.3055 ± 7.8537 | **5.1677 ± 7.8921** | 5.2643 ± 7.8830 | — |
| ppp_hit_probability_gt_positive | **0.7710 ± 0.2759** | 0.7669 ± 0.2918 | 0.7320 ± 0.3159 | — |
| invariant_mmd_xy_car | 0.0891 | 0.1103 | 0.1682 | **0.0718** |
| invariant_mmd_rcs_car | 0.0538 | 0.0357 | 0.0718 | **0.0124** |
| invariant_mmd_doppler_car | 0.1048 | 0.0914 | 0.1515 | **0.0461** |
| invariant_mmd_xy_truck | 0.4537 | **0.0806** | 0.2057 | 0.1014 |
| invariant_mmd_rcs_truck | 0.2968 | 0.2524 | 0.1360 | **0.0197** |
| invariant_mmd_doppler_truck | 0.2632 | 0.2279 | 0.2822 | **0.1082** |
| invariant_mmd_xy_trailer | 0.0572 | **0.0081** | 0.0176 | 0.0322 |
| invariant_mmd_rcs_trailer | 0.6246 | 0.4689 | 0.5018 | **0.0492** |
| invariant_mmd_doppler_trailer | 0.7512 | 0.3949 | 0.5306 | **0.3430** |
| hit_rate | 0.7435 | 0.7509 | 0.7361 | **0.7807** |
| gt_positive_boxes | 269 | 269 | 269 | 269 |
| evaluated_boxes | 299 | 299 | 299 | 299 |

Foreground MMD sampling: Existing evaluator policy: pool nonempty object-local clouds per class and frame; use every third nonempty frame, up to 200 frames per class and side.

All models evaluated 50/50 samples, with zero empty-GT and zero empty-generated samples. Foreground geometry metrics have differing valid-box counts, recorded in the source summaries.
