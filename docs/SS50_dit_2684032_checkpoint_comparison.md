# DiT four-node run 2684032: all-checkpoint SS50 comparison

Training run: `2684032`. Previous-training baseline: initial DiT run `2658827`, final step 18,730. Only run 2684032 is swept; failed runs 2744861/2744862 are excluded.

**Selection criterion:** highest whole-area distance-and-attribute detection F1 on the fixed SS50 benchmark. Exact ties share first place. Other metrics show trade-offs and do not change this selection rule.

New evaluations produce metrics only: no figures, PNGs, TensorBoard output, or prediction exports. Existing matched final-checkpoint and baseline evaluations are reused.

**Sweep incomplete: 2/19 evaluations complete.** Rankings below are provisional; the overall winner will be declared after every checkpoint finishes.

## All metrics

Mean ± population standard deviation; hit rate, class-wise MMD, and median runtime are scalars. Bold marks first place within each metric using unrounded values. Rows without arrows are not ranked; — unavailable or pending. Rankings remain provisional until completion.

| Scope | Metric | DiT initial final (2658827) | DiT four-node step 1000 | DiT four-node step 2000 | DiT four-node step 3000 | DiT four-node step 4000 | DiT four-node step 5000 | DiT four-node step 6000 | DiT four-node step 7000 | DiT four-node step 8000 | DiT four-node step 9000 | DiT four-node step 10000 | DiT four-node step 11000 | DiT four-node step 12000 | DiT four-node step 13000 | DiT four-node step 14000 | DiT four-node step 15000 | DiT four-node step 16000 | DiT four-node step 17000 | DiT four-node step 17822 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entire Area | CD Loc. ↓ | **1.677 ± 0.427** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.700 ± 0.428 |
| Entire Area | CD Full ↓ | 0.043 ± 0.010 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.042 ± 0.010** |
| Entire Area | IoU@1m ↑ | **0.295 ± 0.106** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.294 ± 0.110 |
| Entire Area | DA Recall ↑ | **0.244 ± 0.111** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.239 ± 0.111 |
| Entire Area | DA Precision ↑ | 0.277 ± 0.125 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.279 ± 0.129** |
| Entire Area | DA F1 ↑ | **0.257 ± 0.117** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.255 ± 0.119 |
| Entire Area | MMD Loc. ↓ | **0.062 ± 0.071** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.067 ± 0.063 |
| Entire Area | MMD RCS ↓ | 0.304 ± 0.252 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.268 ± 0.232** |
| Entire Area | MMD Doppler ↓ | 0.275 ± 0.581 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.274 ± 0.584** |
| Foreground | CD Loc. ↓ | **1.053 ± 0.615** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.090 ± 0.768 |
| Foreground | CD Full ↓ | 0.064 ± 0.039 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.061 ± 0.040** |
| Foreground | Density Similarity ↑ | 0.350 ± 0.343 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.366 ± 0.353** |
| Foreground | Hit Rate ↑ | **0.710** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.706 |
| Additional | GT point count | 1095.540 ± 320.639 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1095.540 ± 320.639 |
| Additional | Generated point count | 980.040 ± 318.604 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 949.120 ± 292.826 |
| Additional | Point-count absolute error ↓ | **202.180 ± 194.282** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 226.340 ± 184.591 |
| Additional | PPP expected total count | 981.545 ± 308.120 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 947.683 ± 289.272 |
| Additional | PPP total-count absolute error ↓ | **202.348 ± 191.148** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 226.412 ± 184.251 |
| Additional | RCS MAE at target cells ↓ | **4.602 ± 1.666** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.704 ± 1.762 |
| Additional | Doppler MAE at target cells ↓ | 2.082 ± 1.352 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.028 ± 1.380** |
| Additional foreground | PPP expected count per box | 4.456 ± 7.789 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.794 ± 8.744 |
| Additional foreground | PPP expected-count absolute error per box ↓ | 6.314 ± 8.794 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **6.110 ± 8.745** |
| Additional foreground | PPP hit probability (GT-positive boxes) ↑ | **0.720 ± 0.284** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.717 ± 0.291 |
| Foreground (car) | MMD Loc. ↓ | **0.113** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.203 |
| Foreground (car) | MMD RCS ↓ | **0.035** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.081 |
| Foreground (car) | MMD Doppler ↓ | **0.115** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.127 |
| Foreground (truck) | MMD Loc. ↓ | **0.035** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.044 |
| Foreground (truck) | MMD RCS ↓ | **0.249** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.376 |
| Foreground (truck) | MMD Doppler ↓ | 0.472 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.247** |
| Foreground (trailer) | MMD Loc. ↓ | **0.012** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.074 |
| Foreground (trailer) | MMD RCS ↓ | 1.044 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.799** |
| Foreground (trailer) | MMD Doppler ↓ | 0.723 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.586** |
| Runtime | Inference time (s/frame) ↓ | 0.193 ± 0.005 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.170 ± 0.002** |
| Runtime | Median inference time (s/frame) ↓ | 0.191 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.169** |

## Sources

- DiT initial final (2658827): [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680507_PPP_validation_ss50_2658827/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2680507_PPP_validation_ss50_2658827/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2658827_PPP_train_dit_joint_dino_512_constant_1e5/checkpoints/epoch_17_step_18730.pth`.
- DiT four-node step 1000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_001000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_001000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_3_step_1000.pth`.
- DiT four-node step 2000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_002000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_002000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_7_step_2000.pth`.
- DiT four-node step 3000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_003000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_003000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_10_step_3000.pth`.
- DiT four-node step 4000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_004000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_004000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_14_step_4000.pth`.
- DiT four-node step 5000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_005000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_005000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_18_step_5000.pth`.
- DiT four-node step 6000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_006000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_006000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_21_step_6000.pth`.
- DiT four-node step 7000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_007000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_007000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_25_step_7000.pth`.
- DiT four-node step 8000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_008000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_008000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_29_step_8000.pth`.
- DiT four-node step 9000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_009000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_009000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_32_step_9000.pth`.
- DiT four-node step 10000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_010000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_010000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_36_step_10000.pth`.
- DiT four-node step 11000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_011000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_011000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_39_step_11000.pth`.
- DiT four-node step 12000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_012000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_012000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_43_step_12000.pth`.
- DiT four-node step 13000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_013000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_013000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_47_step_13000.pth`.
- DiT four-node step 14000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_014000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_014000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_50_step_14000.pth`.
- DiT four-node step 15000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_015000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_015000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_54_step_15000.pth`.
- DiT four-node step 16000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_016000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_016000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_58_step_16000.pth`.
- DiT four-node step 17000: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_017000/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep/step_017000/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_61_step_17000.pth`.
- DiT four-node step 17822: [/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701616_PPP_validation_ss50_2684032/summary.json](/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/2701616_PPP_validation_ss50_2684032/summary.json); checkpoint `/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/train/2684032_PPP_continue_dit_2658827_4nodes/checkpoints/epoch_64_step_17822.pth`.
