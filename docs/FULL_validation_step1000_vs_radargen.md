# Full validation: DINO step 1000 versus RadarGen

All annotated validation keyframes, without SS50 subsampling. No PNGs are generated.

| Model | Evaluated | Expected | Complete |
| --- | --- | --- | --- |
| DINO finetuning step 1000 (2744861) | 1906 | 1906 | True |
| RadarGen | 1906 | 1906 | True |

Verified identical 1906 frames, order, seeds, and dataset inventory.

Mean ± population standard deviation where available. **Bold** marks the better mean; ↓ lower is better, ↑ higher is better. — unavailable. Metric names retain the evaluator keys.

| Scope | Metric | DINO step 1000 | RadarGen |
| --- | --- | --- | --- |
| metrics | chamfer_distance_all_norm ↓ | **0.037654 ± 0.008543** | 0.039462 ± 0.008206 |
| metrics | chamfer_distance_xy ↓ | **1.630797 ± 0.465006** | 1.667791 ± 0.384385 |
| metrics | count_abs_error ↓ | **166.259182 ± 146.745509** | 643.014166 ± 409.667483 |
| metrics | distance_attr_f1 ↑ | **0.308516 ± 0.117239** | 0.238661 ± 0.115003 |
| metrics | distance_attr_precision ↑ | **0.326541 ± 0.118680** | 0.201191 ± 0.105330 |
| metrics | distance_attr_recall ↑ | 0.298154 ± 0.118061 | **0.305050 ± 0.136883** |
| metrics | doppler_mae_at_detections ↓ | **1.740924 ± 1.137649** | 3.170428 ± 1.662411 |
| metrics | doppler_mae_at_target_cells ↓ | **1.740924 ± 1.137649** | 3.170428 ± 1.662411 |
| metrics | generated_count | 1021.838405 ± 290.652820 | 1757.830535 ± 470.695169 |
| metrics | gt_count | 1123.769150 ± 316.886723 | 1123.769150 ± 316.886723 |
| metrics | inference_time_seconds ↓ | **0.258978 ± 0.000552** | 1.068550 ± 0.005953 |
| metrics | inference_time_seconds (median) ↓ | 0.258937 | 1.068251 |
| metrics | iou_th1 ↑ | **0.330020 ± 0.121964** | 0.322366 ± 0.109072 |
| metrics | pcl_mmd_doppler ↓ | **0.099101 ± 0.225649** | 0.311366 ± 0.735431 |
| metrics | pcl_mmd_rcs ↓ | 0.137585 ± 0.097799 | **0.086685 ± 0.154601** |
| metrics | pcl_mmd_xy ↓ | **0.047790 ± 0.054620** | 0.058053 ± 0.058772 |
| metrics | ppp_expected_total_count | 1021.516388 ± 289.081131 | — |
| metrics | ppp_total_count_abs_error ↓ | 163.844989 ± 146.017792 | — |
| metrics | rcs_mae_at_detections ↓ | **4.838153 ± 1.851399** | 6.049682 ± 2.101065 |
| metrics | rcs_mae_at_target_cells ↓ | **4.838153 ± 1.851399** | 6.049682 ± 2.101065 |
| metrics | target_collision_count | 12.346275 ± 6.142905 | 40.627492 ± 17.385746 |
| foreground | per_box_chamfer_distance_all_norm ↓ | **0.057493 ± 0.041016** | 0.065590 ± 0.045677 |
| foreground | per_box_chamfer_distance_xy ↓ | 0.950769 ± 0.616814 | **0.916734 ± 0.609247** |
| foreground | per_box_num_points_similarity ↑ | 0.437488 ± 0.352489 | **0.459043 ± 0.346241** |
| foreground | ppp_expected_count | 5.698658 ± 8.506913 | — |
| foreground | ppp_expected_count_abs_error ↓ | 5.016609 ± 7.159088 | — |
| foreground | ppp_hit_probability_gt_positive ↑ | 0.776534 ± 0.295113 | — |
| foreground | invariant_mmd_xy_car ↓ | 0.039344 | **0.029769** |
| foreground | invariant_mmd_rcs_car ↓ | 0.041642 | **0.005008** |
| foreground | invariant_mmd_doppler_car ↓ | 0.026177 | **0.012225** |
| foreground | invariant_mmd_xy_truck ↓ | **0.016089** | 0.026230 |
| foreground | invariant_mmd_rcs_truck ↓ | 0.073655 | **0.030830** |
| foreground | invariant_mmd_doppler_truck ↓ | **0.017259** | 0.037501 |
| foreground | invariant_mmd_xy_trailer ↓ | **0.000555** | 0.007247 |
| foreground | invariant_mmd_rcs_trailer ↓ | 0.146980 | **0.037851** |
| foreground | invariant_mmd_doppler_trailer ↓ | **0.039573** | 0.065383 |
| foreground | hit_rate ↑ | 0.774416 | **0.784164** |
| foreground | gt_positive_boxes | 10874.000000 | 10874.000000 |
| foreground | evaluated_boxes | 11891.000000 | 11891.000000 |

Foreground Chamfer excludes boxes where either cloud is empty. Class-wise MMD uses the existing every-third-nonempty-frame policy, up to 200 frames per class and side. PPP detection-level mark MAE and RadarGen target-cell mark MAE have different target semantics. Runtime excludes loading, metrics, and rendering.

## Sources

- DINO finetuning step 1000 (2744861): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2744861_step1000/summary.json), [provenance](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2744861_step1000/validation_config.json), [predictions](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2744861_step1000/predictions.h5).
- RadarGen: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/summary.json), [provenance](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/validation_config.json), [predictions](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/predictions.h5).
