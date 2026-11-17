# SS50 DINO sampling-budget sweep

Checkpoint: DINO run 2773475, `epoch_91_step_19914.pth` (final checkpoint).

The exact 50 frame identities and order come from validation job 2788375. Saved dense fields are read from the completed full-validation archive. Frame source paths and checkpoint identity are checked against the SS50 run. RadarGen references come from job 2404702, with identical frame identities/order verified.

For target expected mean budget B, multiply every frame's cell masses by the same B / mean_SS50(original total cell mass). This preserves predicted density differences between scenes. Draw the Poisson total and then independent categorical cell locations, retaining repeated cells and original physical RCS/Doppler marks. This has the same law as independent per-cell Poisson sampling. Each budget/frame/seed uses an independent reproducible NumPy seed.

Sweep: 100–2000 inclusive, step 100; seeds 42, 43, 44. All calculations use CPU and the existing evaluation metric functions, including foreground/class MMD and analytical PPP metrics. Analytical counts/hit probabilities are recomputed for scaled intensities. Field-based mark errors and collision counts remain unchanged. Model inference time is excluded; sampling time and complete evaluation wall time are recorded separately.

Plots show seed means and seed standard deviations (not frame uncertainty or confidence intervals). Ground truth mean count is a vertical line; original DINO and RadarGen metrics are markers at their original realized average generated counts. These baselines use variable counts per frame. The curve x-coordinate is the expected average count; each summary also records the realized average count. Frame standard deviations and original baseline metrics are retained in the JSON reports.

Implementation: `scripts/ss50_sampling_budget.py`.
Submit: `sbatch jsc_jupiter/SS50_sampling_budget_cpu.sbatch`.
Outputs: `output/ss50_sampling_budget/` (`experiment.json`, `timing.json`, per-budget/seed metric records and summaries, `metrics.png/pdf`, `foreground.png/pdf`, and baseline summaries).

CPU pilot on the entry host, four PyTorch threads, three frames:

| Expected mean points | Seconds/frame, including report aggregation |
| --- | --- |
| 100 | 0.221 |
| 1000 | 0.297 |
| 3000 | 0.755 |

Setup took 47.7 seconds. These are preliminary timings from three frames, not the final SS50 cost.

The queued jobs 2858168 and 2858186 were cancelled while pending. At the user's request, the full sweep now runs directly on the entry node with **24 single-thread workers**, `nice -n 10`, no GPUs, and one BLAS/PyTorch thread per worker. The scheduled script remains available with 48 workers for future compute-node runs.

Entry-node capacity check: 72 physical ARM cores (one thread/core), CPUs 0–71 accessible; 572 GiB RAM, about 177 GiB available at launch. Load averages were 66.85 / 70.11 / 52.73. Reduced scheduling priority lets other work take precedence while the sweep progresses without Slurm.

Launch script: `jsc_jupiter/SS50_sampling_budget_entry.sh`. Live output is in `output/ss50_sampling_budget/entry.log`; final process CPU time, wall time, and peak resident memory are recorded in `resource_usage.json`. Per-evaluation elapsed timings go to `timing.json`.

Pilot artifacts are in `output/ss50_sampling_budget_pilot/`; their baseline markers represent the full SS50 while pilot curves cover only three frames, so use the full-sweep figures for scientific comparisons.
