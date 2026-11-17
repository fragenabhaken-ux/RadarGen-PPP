# Full validation: DINO step 15000 versus RadarGen

All annotated validation keyframes, without SS50 subsampling. No PNGs are generated.

| Model | Evaluated | Expected | Complete |
| --- | --- | --- | --- |
| DINO finetuning step 15000 (2773475) | 1906 | 1906 | True |
| RadarGen | 1906 | 1906 | True |

Verified identical 1906 frames, order, seeds, and dataset inventory.

Mean ± population standard deviation where available. **Bold** marks the better mean; ↓ lower is better, ↑ higher is better. — unavailable. Metric names retain the evaluator keys.

| Scope | Metric | DINO step 15000 | RadarGen |
| --- | --- | --- | --- |
| metrics | chamfer_distance_all_norm ↓ | **0.036909 ± 0.008411** | 0.039462 ± 0.008206 |
| metrics | chamfer_distance_xy ↓ | **1.506165 ± 0.401184** | 1.667791 ± 0.384385 |
| metrics | count_abs_error ↓ | **511.430745 ± 245.593768** | 643.014166 ± 409.667483 |
| metrics | distance_attr_f1 ↑ | **0.319565 ± 0.109117** | 0.238661 ± 0.115003 |
| metrics | distance_attr_precision ↑ | **0.272984 ± 0.092660** | 0.201191 ± 0.105330 |
| metrics | distance_attr_recall ↑ | **0.394170 ± 0.137414** | 0.305050 ± 0.136883 |
| metrics | doppler_mae_at_detections ↓ | **1.771583 ± 1.126411** | 3.170428 ± 1.662411 |
| metrics | doppler_mae_at_target_cells ↓ | **1.771583 ± 1.126411** | 3.170428 ± 1.662411 |
| metrics | generated_count | 1622.612277 ± 425.093151 | 1757.830535 ± 470.695169 |
| metrics | gt_count | 1123.769150 ± 316.886723 | 1123.769150 ± 316.886723 |
| metrics | inference_time_seconds ↓ | **0.256913 ± 0.001623** | 1.068550 ± 0.005953 |
| metrics | inference_time_seconds (median) ↓ | 0.256360 | 1.068251 |
| metrics | iou_th1 ↑ | **0.355231 ± 0.120290** | 0.322366 ± 0.109072 |
| metrics | pcl_mmd_doppler ↓ | **0.107329 ± 0.244602** | 0.311366 ± 0.735431 |
| metrics | pcl_mmd_rcs ↓ | 0.272095 ± 0.158865 | **0.086685 ± 0.154601** |
| metrics | pcl_mmd_xy ↓ | **0.043171 ± 0.048842** | 0.058053 ± 0.058772 |
| metrics | ppp_expected_total_count | 1622.111981 ± 422.459905 | — |
| metrics | ppp_total_count_abs_error ↓ | 510.748769 ± 241.044861 | — |
| metrics | rcs_mae_at_detections ↓ | **4.902714 ± 1.979395** | 6.049682 ± 2.101065 |
| metrics | rcs_mae_at_target_cells ↓ | **4.902714 ± 1.979395** | 6.049682 ± 2.101065 |
| metrics | target_collision_count | 12.346275 ± 6.142905 | 40.627492 ± 17.385746 |
| foreground | per_box_chamfer_distance_all_norm ↓ | **0.058182 ± 0.041667** | 0.065590 ± 0.045677 |
| foreground | per_box_chamfer_distance_xy ↓ | **0.899925 ± 0.585292** | 0.916734 ± 0.609247 |
| foreground | per_box_num_points_similarity ↑ | **0.463668 ± 0.340246** | 0.459043 ± 0.346241 |
| foreground | ppp_expected_count | 7.966142 ± 12.129716 | — |
| foreground | ppp_expected_count_abs_error ↓ | 5.078693 ± 7.225392 | — |
| foreground | ppp_hit_probability_gt_positive ↑ | 0.818911 ± 0.271443 | — |
| foreground | invariant_mmd_xy_car ↓ | **0.029348** | 0.029769 |
| foreground | invariant_mmd_rcs_car ↓ | 0.163697 | **0.005008** |
| foreground | invariant_mmd_doppler_car ↓ | 0.054589 | **0.012225** |
| foreground | invariant_mmd_xy_truck ↓ | **0.013709** | 0.026230 |
| foreground | invariant_mmd_rcs_truck ↓ | 0.266175 | **0.030830** |
| foreground | invariant_mmd_doppler_truck ↓ | 0.051840 | **0.037501** |
| foreground | invariant_mmd_xy_trailer ↓ | **0.001082** | 0.007247 |
| foreground | invariant_mmd_rcs_trailer ↓ | 0.345366 | **0.037851** |
| foreground | invariant_mmd_doppler_trailer ↓ | **0.039415** | 0.065383 |
| foreground | hit_rate ↑ | **0.820765** | 0.784164 |
| foreground | gt_positive_boxes | 10874.000000 | 10874.000000 |
| foreground | evaluated_boxes | 11891.000000 | 11891.000000 |

Foreground Chamfer excludes boxes where either cloud is empty. Class-wise MMD uses the existing every-third-nonempty-frame policy, up to 200 frames per class and side. PPP detection-level mark MAE and RadarGen target-cell mark MAE have different target semantics. Runtime excludes loading, metrics, and rendering.

## Sources

- DINO finetuning step 15000 (2773475): [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2773475_step15000/summary.json), [provenance](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2773475_step15000/validation_config.json), [predictions](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/dino_2773475_step15000/predictions.h5).
- RadarGen: [summary](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/summary.json), [provenance](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/validation_config.json), [predictions](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/FULL_validation_20261108/radargen/predictions.h5).
