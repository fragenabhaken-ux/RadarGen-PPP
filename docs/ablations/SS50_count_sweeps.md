# SS50 ablation count sweeps versus RadarGen

Three separate comparisons, each containing only one requested ablation and RadarGen:

| Model | Training run | Checkpoint step | Results |
| --- | --- | ---: | --- |
| DiT without output bottleneck | 2867524 | 58,057 | [Table](../../output/paper_material/count_sweep_evaluation/ss50_dit_no_bottleneck_count_sweep/results.md) · [Plots](../../output/paper_material/count_sweep_evaluation/ss50_dit_no_bottleneck_count_sweep/plots/README.md) |
| DINO Compact | 2867528 | 20,595 | [Table](../../output/paper_material/count_sweep_evaluation/ss50_dino_compact_count_sweep/results.md) · [Plots](../../output/paper_material/count_sweep_evaluation/ss50_dino_compact_count_sweep/plots/README.md) |
| Random-initialized DINO | 2863217 | 29,915 | [Table](../../output/paper_material/count_sweep_evaluation/ss50_dino_random_count_sweep/results.md) · [Plots](../../output/paper_material/count_sweep_evaluation/ss50_dino_random_count_sweep/plots/README.md) |

Use the exact same 50 frame identities/order as the earlier count sweeps. Continuous GT arrays are checked for exact equality with the RadarGen reference. Read saved dense fields and native points from the completed full-validation bundles; no model inference is required. The native baseline uses the original points from these same archived maps, whose per-frame RNG indices follow the full-validation run.

For each model independently, target an expected mean count of 100–2000 inclusive in steps of 100, using one shared intensity multiplier B / mean_SS50(original total cell mass). Thus scene-to-scene density differences remain intact. Sample with independent reproducible seeds 42, 43, 44, retain repeated cells, and preserve physical RCS/Doppler fields. Twenty count targets × three seeds × 50 frames = 3000 frame evaluations per model, plus its 50-frame native reference.

All existing point-cloud, foreground, per-class MMD, and analytical PPP metrics are evaluated. Original per-box PPP expected counts are integrated once; expected counts then scale linearly and Poisson hit probabilities are recomputed at each multiplier. Field MAE uses GT detection locations. Empty-cloud handling and aggregation use existing evaluator functions.

Each comparison reuses the existing RadarGen shared-threshold curve, targeting actual means 100–1700, plus its native threshold 0.1 baseline on those same fixed maps (mean 1737.30). Its fixed optimized maps support at most 1750.04 mean points, so no RadarGen curve is fabricated at 1800–2000. Historical RadarGen and other model references are excluded from these new comparisons.

Each results directory contains `experiment.json`, per-budget/seed and native metric records, `timing.json`, `results.md`, `results.csv`, and `plots/` with one PNG per recorded numeric metric. Plots use realized mean counts and three-seed mean ± population SD; PPP-only metrics explicitly state that RadarGen has no equivalent. GT/raster diagnostics and postprocessing timing are labeled separately from prediction-quality metrics.

Run all three with `bash jsc_jupiter/SS50_ablation_count_sweeps_entry.sh`. One shared pool of 24 CPU workers, one PyTorch/BLAS thread each, reduced scheduling priority, no GPUs. Model fields, annotations, and native baseline selection are held fixed. Live log: `output/paper_material/count_sweep_evaluation/ablation_count_sweeps.log`; combined actual compute/memory: `ablation_count_sweeps_resource_usage.json` in the same parent directory.
