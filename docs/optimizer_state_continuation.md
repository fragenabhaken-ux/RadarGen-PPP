# DINO and DiT continuation with optimizer state

Continue the original four-node final checkpoints, preserving model parameters, optimizer moments and internal counters, global progress, data position, and all 16 ranks' RNG states.

| Model | Source checkpoint | Learning rate for additional training |
| --- | --- | --- |
| DINO | `2684031`, `epoch_45_step_9971.pth` | Fresh cosine from `1e-4` to `1e-5`, no warmup |
| DiT | `2684032`, `epoch_64_step_17822.pth` | Continue constant `1e-5`; saved warmup already complete |

DINO starts its **new scheduler at local update zero**, while its global training counter continues from 9,971. The cosine has the same 45-epoch decay horizon as the previous weights-only run (`decay_end_fraction: 0.045` of a 1000-epoch scheduler budget); it then holds at `1e-5`. The original training/data/optimizer/loss contract is retained; only the scheduler fields may change.

DiT uses exact resume from global step 17,822, including its existing constant scheduler state. Its 100-update historical warmup remains in the configuration for strict compatibility and does not run again.

Both retain four nodes, 16 GPU workers total, the original per-GPU batch sizes (DINO 24, DiT 19), cubic mark links and original L1 objectives, and a 12-hour allocation with an 11.8-hour training guard. DINO's terminal epoch ceiling is extended from 65 to 1000 so the original limit does not terminate the additional allocation early. Work directories are new and include the Slurm job ID; source checkpoints are unchanged.

## Launch scripts

- `configs/PPP_resume_dino_2684031_optimizer_4nodes.yaml`
- `jsc_jupiter/PPP_resume_dino_2684031_optimizer_4nodes.sbatch`
- `configs/PPP_resume_dit_2684032_optimizer_4nodes.yaml`
- `jsc_jupiter/PPP_resume_dit_2684032_optimizer_4nodes.sbatch`

The DINO launch uses `--resume SOURCE --replace-scheduler`; the DiT launch uses `--resume SOURCE`. Replacement is explicitly limited to a constant-to-cosine transition with zero warmup. All other contract fields must still match exactly. The replacement loads optimizer state and progress, skips the old scheduler state, and applies the freshly constructed cosine's starting LR.

## Subsequent interruption and resume

Checkpoints produced by the DINO continuation already contain the new cosine state and new scheduler contract. Resume those with the same new configuration and `--resume NEW_CHECKPOINT`, **without** `--replace-scheduler`, to preserve the cosine position rather than restart it. The replacement flag intentionally rejects checkpoints that already use cosine.

Global save cadence remains every 1000 updates. TensorBoard and training prediction behavior are inherited from each original source run.

## Verification

Actual source checkpoints contain populated optimizer state (628 parameter states for DINO, 606 for DiT), saved schedulers, progress and 16 per-rank RNG states. The current training manifest matches their stored manifest identity. Prepared configurations preserve the strict original contracts except for the explicit DINO scheduler replacement. CPU tests check retained moments/counters and progress, zero-warmup cosine behavior, normal resume of newly generated checkpoints, rejection of unrelated contract changes, and existing uninterrupted-versus-resumed training/data-order equivalence.

## Submitted runs

- DINO: Slurm `2773475`.
- DiT: Slurm `2773477`.

Both submitted on 2026-11-08 after the 12 targeted CPU tests and checkpoint contract preflight passed.
