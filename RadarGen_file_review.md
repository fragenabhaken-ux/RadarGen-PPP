# RadarGen PPP: file review and publication plan

Reviewed snapshot: `60ef461472136cbf1ebca5cd01ba2d4cd9ac6c67`, branch `preprocessing_data_input_output`.
Comparison base: `408e8cb4b15b143c926084e10f45223d17dcef80` (original RadarGen evaluation commit).
Sources: supplied `inventory(1).txt` and `changes(1).diff`. This is a file-purpose/publication review, not a runtime correctness audit or a fresh test run.

The committed comparison contains **141 files: 125 added and 16 modified**, with no listed deletions or renames. The separately generated review archive is included as item 142. The captured working-tree status was clean; it predates that archive copy. Other untracked or ignored files are not enumerated here.

## Change overview

- JUPITER environment and Slurm setup, offline preprocessing launchers and validation records.
- HDF5 preprocessing, continuous radar target capture, metadata/provenance and restart validation.
- PPP dataset loading and sparse target rasterization, replacing dense-map supervision for the PPP path.
- PPP and physical mark losses; direct conditioning-only DiT with a trainable scalar decoder.
- Training integration, accumulation, optimizer checks, checkpointing and resume.
- Seeded PPP grid sampling and inference.
- Evaluation integration with continuous HDF5 ground truth and analytical count/hit metrics.
- Small-subset overfit diagnostics and focused regression tests.
- Supporting documentation, source-transfer archives, patches and generated job outputs.

## Categories

- **Delete:** redundant delivery/review artifacts. This is a recommendation, not proof that every binary archive has no unique contents.
- **Keep privately:** useful development evidence, reference code, cluster-specific configuration or historical tooling. Move out of the publication tree, not into the trash. Keep active launchers and dependencies usable throughout ongoing JUPITER work.
- **Publish:** scientific implementation plus the tests, preprocessing utilities, configurations and documentation needed to reproduce it. Some files need relocation or portable defaults first.

| Category | Files |
|---|---:|
| Delete | 10 |
| Keep privately | 70 |
| Publish | 62 |

## Important cleanup dependencies

1. Do not delete `jsc_jupiter` wholesale. It contains the actual HDF5 writer, production runner, output validation and tests. Move reusable components into a portable preprocessing package and update imports first.
2. `test_hdf5_samples.py` imports `create_hdf5_samples.py`; `test_hdf5_4nodes.py` imports both the worker and launch-check helper. Keep or migrate those dependencies together.
3. The two 480-scene lists have identical added contents, but different launchers reference them. Consolidate references before deleting a copy.
4. Production launch code references `/e/project1/nxtaim-1/huber7/pretrained/offline_hdf5_smoke.py`, which is not included in this changed-file inventory. Capture its behavior and provide an in-repository portable setup before claiming preprocessing is reproducible from the published repository alone.
5. The two generated radar lists contribute 1,009,703 of 1,024,946 added lines, approximately 98.5%. This is data inventory, not a million lines of implementation.
6. Current PPP evaluation explicitly stops on empty sampled point clouds. Preserve that limitation in documentation until a metric policy is implemented; a successful smoke run is not full evaluation validation.
7. Keep the actual preprocessing signatures, scene selection, tested dependency versions and experiment configs as reproducibility evidence. Moving local copies privately does not eliminate the need for portable public equivalents.
8. Removing files in a new commit does not remove their old contents from Git history. Ignore rules also do not stop tracking already committed files. No history rewrite or file deletion is performed by this review.

## Complete per-file inventory

Descriptions summarize each file's purpose or net change. Categories express the intended publication outcome; the notes identify migration or verification needed first.

| # | File | Git status | Category | Change / function | Action / qualification |
|---:|---|---|---|---|---|
| 1 | `.gitignore` | Modified | **Publish** | Ignores generated HDF5 files, camera manifests and selected JUPITER logs/backups. | Extend to root job logs and review exports; ignores do not untrack committed files. |
| 2 | `.gitmodules` | Modified | **Publish** | Changes UniDepth and UFM submodule URLs from SSH to public HTTPS. | Keep for installation without SSH credentials. |
| 3 | `APPLY_WP8.txt` | Added | **Delete** | Instructions for applying the already integrated WP8 patch. | Obsolete once the patch is committed. |
| 4 | `README_WP6.md` | Added | **Keep privately** | Implementation-package instructions and WP6 inference checks. | Merge enduring instructions into public documentation before archiving this version. |
| 5 | `README_WP7.md` | Added | **Keep privately** | WP7 package application and evaluation instructions. | Merge durable evaluation guidance into public documentation. |
| 6 | `README_WP7_HDF5_fix.txt` | Added | **Keep privately** | Explains the HDF5-ground-truth evaluation correction. | Retain the rationale in public evaluation docs, archive the patch-specific instructions. |
| 7 | `README_WP8.md` | Added | **Keep privately** | Explains the overfit diagnostic package and how to run it. | Extract a portable overfit guide for publication. |
| 8 | `RadarGen_WP6.zip` | Added | **Delete** | WP6 implementation transfer archive. | Binary contents are not in the diff; confirm no unique unapplied content before removal. |
| 9 | `RadarGen_WP7.zip` | Added | **Delete** | WP7 implementation transfer archive. | Binary contents are not in the diff; confirm no unique unapplied content before removal. |
| 10 | `RadarGen_WP7_HDF5_fix.zip` | Added | **Delete** | HDF5 evaluation fix transfer archive. | Binary contents are not in the diff; confirm no unique unapplied content before removal. |
| 11 | `RadarGen_WP8_overfit.zip` | Added | **Delete** | WP8 implementation transfer archive. | Binary contents are not in the diff; confirm no unique unapplied content before removal. |
| 12 | `WP6_status.tex` | Added | **Keep privately** | Historical WP6 implementation and verification status. | Archive as development evidence; avoid multiple conflicting status sources. |
| 13 | `WP7_status.tex` | Added | **Keep privately** | Historical WP7 implementation and verification status. | Archive as development evidence. |
| 14 | `configs/RadarGen_600M_512px_TS_PPP_inference.yaml` | Added | **Publish** | Configures PPP inference, local assets and checkpoint loading. | Replace personal paths and diagnostic checkpoint defaults with documented placeholders. |
| 15 | `configs/RadarGen_600M_512px_TS_PPP_training.yaml` | Added | **Publish** | Defines PPP dataset, model, optimizer, losses and training settings. | Replace personal paths; retain a documented reproducible configuration. |
| 16 | `configs/RadarGen_PPP_WP8_overfit.yaml` | Added | **Publish** | Defines the four-example, 200-update diagnostic experiment. | Publish as an optional overfit example after making paths portable. |
| 17 | `diffusion/__init__.py` | Modified | **Publish** | Lazily imports diffusion samplers so configuration-only operations stay lightweight. | Keep with its current dependent implementation. |
| 18 | `diffusion/data/datasets/__init__.py` | Modified | **Publish** | Registers the PPP dataset wrapper. | Keep with its current dependent implementation. |
| 19 | `diffusion/data/datasets/radargen_ppp_wrapper.py` | Added | **Publish** | Builds the HDF5 PPP dataset through the existing dataset registry. | Keep with its current dependent implementation. |
| 20 | `diffusion/data/datasets/radargen_wrapper.py` | Modified | **Publish** | Defers the baseline dataset import until construction. | Preserve baseline behavior. |
| 21 | `diffusion/model/nets/__init__.py` | Modified | **Publish** | Exports and registers the PPP model. | Keep with its current dependent implementation. |
| 22 | `diffusion/model/nets/radargen_ppp.py` | Added | **Publish** | Adds conditioning-only, fixed-timestep DiT prediction with three jointly attended modality streams. | Keep with its current dependent implementation. |
| 23 | `diffusion/model_modification.py` | Modified | **Publish** | Adds pretrained input-projection adaptation from 32 to 99 channels for PPP. | Keep with its current dependent implementation. |
| 24 | `diffusion/utils/checkpoint.py` | Modified | **Publish** | Adds optional checkpoint metadata and supplied RNG state; defers model-download import. | Keep with its current dependent implementation. |
| 25 | `diffusion/utils/config.py` | Modified | **Publish** | Adds the PPP HDF5 data-root configuration field. | Keep with its current dependent implementation. |
| 26 | `evaluation/config.py` | Modified | **Publish** | Adds optional scene-token filtering for evaluation. | Keep with its current dependent implementation. |
| 27 | `evaluation/configs/truckscenes_eval_ppp.yaml` | Added | **Publish** | Configures validation-keyframe PPP evaluation. | Make paths and checkpoint selection portable. |
| 28 | `evaluation/configs/truckscenes_eval_ppp_smoke.yaml` | Added | **Publish** | Defines a restricted PPP evaluation smoke check. | Keep as an explicitly diagnostic example; generalize paths. |
| 29 | `evaluation/evaluator.py` | Modified | **Publish** | Integrates shared HDF5 ground truth, analytical PPP metrics and explicit empty-cloud rejection. | Resolve the empty-cloud metric policy before final results. |
| 30 | `evaluation/models/__init__.py` | Modified | **Publish** | Registers the PPP evaluation wrapper. | Keep with its current dependent implementation. |
| 31 | `evaluation/models/radargen_ppp_wrapper.py` | Added | **Publish** | Connects PPP prediction to evaluation and validates frame/target provenance. | Keep with its current dependent implementation. |
| 32 | `evaluation/ppp_ground_truth.py` | Added | **Publish** | Reads continuous HDF5 radar targets while retaining coordinates, marks and multiplicity. | Keep with its current dependent implementation. |
| 33 | `evaluation/ppp_metrics.py` | Added | **Publish** | Computes expected counts and box hit probabilities from grid masses and aggregates scores. | Keep with its current dependent implementation. |
| 34 | `evaluation/scripts/check_ppp_evaluation.py` | Added | **Publish** | Preflights PPP evaluation, counts and empty realizations. | Keep with its current dependent implementation. |
| 35 | `evaluation_results/truckscenes_ppp_smoke/eval_results_20261004_002609.json` | Added | **Keep privately** | Recorded smoke evaluation metrics. | Archive as validation evidence, not final benchmark results. |
| 36 | `jsc_jupiter/.gitignore` | Added | **Keep privately** | Ignores machine-local environment/log outputs. | Keep with private cluster tooling; consolidate relevant rules in public root ignores. |
| 37 | `jsc_jupiter/HDF5_PHASE1.md` | Added | **Keep privately** | Documents the original single-scene HDF5 smoke workflow. | Extract the current schema and reproducibility instructions into public docs. |
| 38 | `jsc_jupiter/RadarGen_scene_EDA.ipynb` | Added | **Keep privately** | Notebook for inspecting a staged scene and invoking EDA utilities. | Research exploration; not required by the training pipeline. |
| 39 | `jsc_jupiter/activate_jupiter.sh` | Added | **Keep privately** | Activates the JUPITER environment and adjusts runtime library paths. | Keep while running jobs; public instructions should be portable. |
| 40 | `jsc_jupiter/config_jupiter.sh` | Added | **Keep privately** | Defines JUPITER environment, repository and project locations. | Preserve outside the published source tree. |
| 41 | `jsc_jupiter/constraints_jupiter.txt` | Added | **Keep privately** | Pins dependency constraints used for JUPITER setup. | Use the tested versions to inform public installation requirements. |
| 42 | `jsc_jupiter/create_bev_condition_maps_smoke.py` | Added | **Keep privately** | Specialized BEV preprocessing smoke entry point. | Keep for baseline troubleshooting until consolidated. |
| 43 | `jsc_jupiter/create_hdf5_dataset_scene.py` | Added | **Keep privately** | Production single-scene runner used by the job-array launcher. | Overlaps the four-node runner; consolidate before retirement. |
| 44 | `jsc_jupiter/create_hdf5_samples.py` | Added | **Keep privately** | Original single-scene HDF5 staging and generation runner. | Still imported by test_hdf5_samples.py; not safe to delete independently. |
| 45 | `jsc_jupiter/create_radar_maps_smoke.py` | Added | **Keep privately** | Specialized radar-map preprocessing smoke entry point. | Keep for baseline parity checks until consolidated. |
| 46 | `jsc_jupiter/hdf5_4nodes.sbatch` | Added | **Keep privately** | Launches production preprocessing with four nodes and 16 tasks. | Machine/account-specific launcher; retain with worker dependencies. |
| 47 | `jsc_jupiter/hdf5_4nodes_launch_check.py` | Added | **Keep privately** | Checks task placement, GPU uniqueness and CPU affinity. | Cluster validation utility; retained tests currently import it. |
| 48 | `jsc_jupiter/hdf5_4nodes_launch_check.sbatch` | Added | **Keep privately** | Submits the multi-node placement/affinity validation job. | Preserve outside the published source tree. |
| 49 | `jsc_jupiter/hdf5_4nodes_runner.py` | Added | **Publish** | Stages scene inputs and generates HDF5 samples through the production preprocessing flow. | Extract the canonical portable scene runner; consolidate overlapping runners. |
| 50 | `jsc_jupiter/hdf5_4nodes_scenes.txt` | Added | **Publish** | Records the exact 480-scene production selection. | Keep one canonical scene manifest with split/version provenance; update references if relocated. |
| 51 | `jsc_jupiter/hdf5_4nodes_worker.py` | Added | **Publish** | Distributes scenes, validates exact outputs and manages signatures, restart and completion markers. | Publish the reusable validation/orchestration logic; separate cluster paths and the external offline wrapper. |
| 52 | `jsc_jupiter/hdf5_all.sbatch` | Added | **Keep privately** | Alternative scene-array production preprocessing launcher. | Uses create_hdf5_dataset_scene.py and the external offline wrapper. |
| 53 | `jsc_jupiter/hdf5_samples.py` | Added | **Publish** | Implements HDF5 serialization, annotation/temporal metadata, validation and baseline-map comparison. | Move to a portable preprocessing package and update imports. |
| 54 | `jsc_jupiter/hdf5_scene_tokens.txt` | Added | **Keep privately** | Second copy of the 480-scene list, identical to hdf5_4nodes_scenes.txt. | Retain while hdf5_all.sbatch references it; delete only after switching to the canonical manifest. |
| 55 | `jsc_jupiter/hdf5_smoke.sbatch` | Added | **Keep privately** | Submits single-scene HDF5 smoke preprocessing. | Preserve outside the published source tree. |
| 56 | `jsc_jupiter/implementation_roadmaps/RadarGen_PPP_Roadmap.tex` | Added | **Keep privately** | Development specifications, design rationale and implementation/run evidence. | Archive the full working roadmap; publish a concise current method/reproduction guide. |
| 57 | `jsc_jupiter/implementation_roadmaps/ppp_finals_unet_54k.py` | Added | **Keep privately** | Reference MMSegmentation UNet training configuration from earlier PPP work. | Reference material, not the RadarGen training entry point. |
| 58 | `jsc_jupiter/implementation_roadmaps/ppp_finals_unet_head.py` | Added | **Keep privately** | Reference MMSegmentation PPP/mark decoder head. | Preserve privately as provenance; not part of current runtime. |
| 59 | `jsc_jupiter/implementation_roadmaps/ppp_loss.py` | Added | **Keep privately** | Reference MMSegmentation PPP loss variants. | The active implementation is radargen/losses/ppp_loss.py. |
| 60 | `jsc_jupiter/list_required_radar_files.py` | Added | **Publish** | Enumerates radar inputs using the same adapter selection as preprocessing. | Generalize defaults and publish as a preprocessing utility. |
| 61 | `jsc_jupiter/modules_jupiter.sh` | Added | **Keep privately** | Loads the JUPITER module stack. | Preserve outside the published source tree. |
| 62 | `jsc_jupiter/preprocess_smoke.sbatch` | Added | **Keep privately** | Submits the original BEV/radar preprocessing smoke workflow. | Preserve outside the published source tree. |
| 63 | `jsc_jupiter/preprocess_smoke_1874930.err` | Added | **Keep privately** | stderr/warnings from run preprocess_smoke_1874930. | Archive as execution evidence outside the published source tree. |
| 64 | `jsc_jupiter/preprocess_smoke_1874930.out` | Added | **Keep privately** | stdout/progress from run preprocess_smoke_1874930. | Archive as execution evidence outside the published source tree. |
| 65 | `jsc_jupiter/preprocess_smoke_1926923.err` | Added | **Keep privately** | stderr/warnings from run preprocess_smoke_1926923. | Archive as execution evidence outside the published source tree. |
| 66 | `jsc_jupiter/preprocess_smoke_1926923.out` | Added | **Keep privately** | stdout/progress from run preprocess_smoke_1926923. | Archive as execution evidence outside the published source tree. |
| 67 | `jsc_jupiter/radar_stage_smoke_2131044.err` | Added | **Keep privately** | stderr/warnings from run radar_stage_smoke_2131044. | Archive as execution evidence outside the published source tree. |
| 68 | `jsc_jupiter/radar_stage_smoke_2131044.out` | Added | **Keep privately** | stdout/progress from run radar_stage_smoke_2131044. | Archive as execution evidence outside the published source tree. |
| 69 | `jsc_jupiter/radar_staged_smoke.sbatch` | Added | **Keep privately** | Submits radar preprocessing against extracted scene sweeps. | Preserve outside the published source tree. |
| 70 | `jsc_jupiter/radargen_radar_maps_eda.py` | Added | **Keep privately** | Plots dense radar targets and compares them to raw detections. | Preserve outside the published source tree. |
| 71 | `jsc_jupiter/radargen_scene_eda.py` | Added | **Keep privately** | Plots cameras, radar clouds, mark distributions and per-frame counts. | Preserve outside the published source tree. |
| 72 | `jsc_jupiter/required_radar_files.txt` | Added | **Keep privately** | Generated list of required radar file paths (560,962 lines). | Archive outside source control as provenance; regenerate with enumeration tooling. |
| 73 | `jsc_jupiter/required_sweep_files.txt` | Added | **Keep privately** | Generated sweep-only input list (448,741 lines). | Archive outside source control; not executable code. |
| 74 | `jsc_jupiter/requirements_jupiter.txt` | Added | **Keep privately** | Lists the JUPITER Python environment requirements. | Preserve the environment record; derive a portable public dependency specification. |
| 75 | `jsc_jupiter/setup_jupiter.sh` | Added | **Keep privately** | Creates/configures the JUPITER environment and validates dependencies. | Preserve outside the published source tree. |
| 76 | `jsc_jupiter/test_full_imports.py` | Added | **Keep privately** | Diagnoses imports across the JUPITER dependency stack. | Preserve outside the published source tree. |
| 77 | `jsc_jupiter/test_full_imports.sbatch` | Added | **Keep privately** | Runs the dependency import diagnostic as a GPU job. | Preserve outside the published source tree. |
| 78 | `jsc_jupiter/test_hdf5_4nodes.py` | Added | **Publish** | Tests production inventory validation and launch-record rejection. | Retain portable validation tests; move cluster-specific checks with their helper. |
| 79 | `jsc_jupiter/test_hdf5_samples.py` | Added | **Publish** | Tests serialization, preprocessing callback parity and staging cleanup. | Move under tests; currently imports create_hdf5_samples.py, so migrate imports before retiring it. |
| 80 | `jsc_jupiter/truckscenes_bev_condition_maps_jupiter.yaml` | Added | **Keep privately** | Machine-specific full BEV preprocessing configuration. | Preserve outside the published source tree. |
| 81 | `jsc_jupiter/truckscenes_bev_condition_maps_smoke.yaml` | Added | **Keep privately** | Machine-specific BEV smoke configuration. | Preserve outside the published source tree. |
| 82 | `jsc_jupiter/truckscenes_hdf5_all.yaml` | Added | **Publish** | Specifies the full HDF5 preprocessing contract and production settings. | Publish a portable template; retain the original machine-specific values privately. |
| 83 | `jsc_jupiter/truckscenes_hdf5_smoke.yaml` | Added | **Keep privately** | Single-scene HDF5 smoke settings and local paths. | Retain as private run record; public smoke settings can be derived from the canonical template. |
| 84 | `jsc_jupiter/truckscenes_radar_maps_jupiter.yaml` | Added | **Keep privately** | Machine-specific full radar-map preprocessing configuration. | Preserve outside the published source tree. |
| 85 | `jsc_jupiter/truckscenes_radar_maps_smoke.yaml` | Added | **Keep privately** | Staged-scene radar-map smoke configuration. | Preserve outside the published source tree. |
| 86 | `jsc_jupiter/wp5_training_smoke.sbatch` | Added | **Keep privately** | Submits real-model training and checkpoint-resume smoke checks. | Retain exact job recipe; publish portable underlying commands. |
| 87 | `jsc_jupiter/wp6_inference_smoke.sbatch` | Added | **Keep privately** | Submits real-model inference and sampling checks. | Preserve outside the published source tree. |
| 88 | `jsc_jupiter/wp7_evaluation_smoke.sbatch` | Added | **Keep privately** | Submits PPP evaluation preflight and smoke evaluation. | Preserve outside the published source tree. |
| 89 | `jsc_jupiter/wp8_overfit.sbatch` | Added | **Keep privately** | Submits the four-example overfit diagnostic and summary plots. | Preserve outside the published source tree. |
| 90 | `radargen/bev_condition_maps/core.py` | Modified | **Publish** | Adds a callback exposing native t0 depth outputs during BEV preprocessing. | Keep with its current dependent implementation. |
| 91 | `radargen/bev_condition_maps/creator.py` | Modified | **Publish** | Passes through the depth callback and corrects output-shape documentation. | Keep with its current dependent implementation. |
| 92 | `radargen/core/data_types.py` | Modified | **Publish** | Defines the PPP training-batch fields and tensor contract. | Keep with its current dependent implementation. |
| 93 | `radargen/inference/ppp_pipeline.py` | Added | **Publish** | Loads PPP components/checkpoints and produces decoded fields and sampled point clouds. | Keep with its current dependent implementation. |
| 94 | `radargen/losses/__init__.py` | Added | **Publish** | Exports PPP and mark losses. | Keep with its current dependent implementation. |
| 95 | `radargen/losses/mark_losses.py` | Added | **Publish** | Implements masked physical RCS and Doppler losses, including circular Doppler mode. | Keep with its current dependent implementation. |
| 96 | `radargen/losses/ppp_loss.py` | Added | **Publish** | Implements numerically checked PPP likelihood reductions with differentiable FP32 math. | Keep with its current dependent implementation. |
| 97 | `radargen/ppp/__init__.py` | Added | **Publish** | Exports PPP grid-mass and sampling utilities. | Keep with its current dependent implementation. |
| 98 | `radargen/ppp/sampling.py` | Added | **Publish** | Converts log-intensity to cell masses and performs seeded per-cell Poisson sampling. | Keep with its current dependent implementation. |
| 99 | `radargen/ppp_targets/__init__.py` | Added | **Publish** | Exports sparse-target rasterization utilities. | Keep with its current dependent implementation. |
| 100 | `radargen/ppp_targets/creator.py` | Added | **Publish** | Converts continuous detections into sparse training masks and physical mark targets. | Keep with its current dependent implementation. |
| 101 | `radargen/radar_maps/creator.py` | Modified | **Publish** | Adds a callback capturing filtered metric radar detections before rasterization. | Keep with its current dependent implementation. |
| 102 | `radargen/training/ppp_checkpoint.py` | Added | **Publish** | Saves/restores DiT, decoder, optimizer, scheduler, training position and RNG state. | Keep with its current dependent implementation. |
| 103 | `radargen/training/ppp_decoder.py` | Added | **Publish** | Adapts RGB output to one scalar channel, keeps the decoder differentiable/FP32 and freezes condition encoding. | Keep with its current dependent implementation. |
| 104 | `radargen/training/ppp_initialization.py` | Added | **Publish** | Validates local assets and initializes pretrained SANA/DC-AE in the required order. | Keep with its current dependent implementation. |
| 105 | `radargen/training/ppp_overfit.py` | Added | **Publish** | Produces per-sample decoded-field statistics and visual diagnostics. | Optional training diagnostic, still useful for reproducibility. |
| 106 | `radargen/training/radargen_ppp_dataset.py` | Added | **Publish** | Loads and validates HDF5 conditions and constructs sparse PPP training targets. | Keep with its current dependent implementation. |
| 107 | `radargen/training/radargen_ppp_training_model.py` | Added | **Publish** | Runs one fixed-timestep DiT call and shared scalar decoding into three output fields. | Keep with its current dependent implementation. |
| 108 | `radargen_evaluation.tar.gz` | Added | **Delete** | Evaluation-source snapshot created for review/transfer. | Binary contents are not in the diff; confirm any unique content is preserved. |
| 109 | `scripts/check_ppp_data.py` | Added | **Publish** | Checks real HDF5 loading, target geometry, validity and batching. | Keep with its current dependent implementation. |
| 110 | `scripts/check_ppp_inference.py` | Added | **Publish** | Checks real PPP inference and seeded sampling. | Keep with its current dependent implementation. |
| 111 | `scripts/check_ppp_model.py` | Added | **Publish** | Checks pretrained PPP initialization, forward pass and gradient flow. | Keep with its current dependent implementation. |
| 112 | `scripts/inspect_ppp_checkpoints.py` | Added | **Keep privately** | One-off inspection of pretrained checkpoint keys and decoder adaptation. | Useful initialization evidence; overlap with model checks makes a separate public utility optional. |
| 113 | `scripts/prepare_ppp_null_embedding.py` | Added | **Publish** | Creates the genuine empty-text embedding from a local text model. | Needed to reproduce setup; do not regenerate it for existing runs. |
| 114 | `scripts/summarize_ppp_overfit.py` | Added | **Publish** | Summarizes diagnostic JSONL records and plots learning curves. | Keep with its current dependent implementation. |
| 115 | `scripts/train_ppp.py` | Added | **Publish** | Implements configuration validation, decoded-loss training, accumulation, clipping, checkpoint resume and diagnostics. | Keep with its current dependent implementation. |
| 116 | `tests/test_ppp_ground_truth.py` | Added | **Publish** | Tests continuous target preservation, multiplicity and input validation. | Keep with its current dependent implementation. |
| 117 | `tests/test_ppp_inference.py` | Added | **Publish** | Tests sampling geometry, statistics, seeds and strict checkpoint loading. | Keep with its current dependent implementation. |
| 118 | `tests/test_ppp_losses.py` | Added | **Publish** | Tests analytical loss values, gradients, reductions, masks and numerical edge cases. | Keep with its current dependent implementation. |
| 119 | `tests/test_ppp_metrics.py` | Added | **Publish** | Tests box geometry, continuous target counts and metric aggregation. | Keep with its current dependent implementation. |
| 120 | `tests/test_ppp_model.py` | Added | **Publish** | Tests PPP model contracts, initialization and decoder gradients with small/substitute components. | Keep with its current dependent implementation. |
| 121 | `tests/test_ppp_overfit.py` | Added | **Publish** | Tests diagnostic count semantics and physical mark transformations. | Keep with its current dependent implementation. |
| 122 | `tests/test_ppp_training.py` | Added | **Publish** | Tests integrated updates, accumulation, freezing, loss weighting and save/resume with substitute models. | Keep with its current dependent implementation. |
| 123 | `wp4_check_2164430.err` | Added | **Keep privately** | stderr/warnings from run wp4_check_2164430. | Archive as execution evidence outside the published source tree. |
| 124 | `wp4_check_2164430.out` | Added | **Keep privately** | stdout/progress from run wp4_check_2164430. | Archive as execution evidence outside the published source tree. |
| 125 | `wp5_training_2164843.err` | Added | **Keep privately** | stderr/warnings from run wp5_training_2164843. | Archive as execution evidence outside the published source tree. |
| 126 | `wp5_training_2164843.out` | Added | **Keep privately** | stdout/progress from run wp5_training_2164843. | Archive as execution evidence outside the published source tree. |
| 127 | `wp6_inference_2165710.err` | Added | **Keep privately** | stderr/warnings from run wp6_inference_2165710. | Archive as execution evidence outside the published source tree. |
| 128 | `wp6_inference_2165710.out` | Added | **Keep privately** | stdout/progress from run wp6_inference_2165710. | Archive as execution evidence outside the published source tree. |
| 129 | `wp7.patch` | Added | **Delete** | Patch used to deliver the WP7 implementation. | Retire after confirming all intended hunks are in the committed source. |
| 130 | `wp7_evaluation_2166130.err` | Added | **Keep privately** | stderr/warnings from run wp7_evaluation_2166130. | Archive as execution evidence outside the published source tree. |
| 131 | `wp7_evaluation_2166130.out` | Added | **Keep privately** | stdout/progress from run wp7_evaluation_2166130. | Archive as execution evidence outside the published source tree. |
| 132 | `wp7_evaluation_2166259.err` | Added | **Keep privately** | stderr/warnings from run wp7_evaluation_2166259. | Archive as execution evidence outside the published source tree. |
| 133 | `wp7_evaluation_2166259.out` | Added | **Keep privately** | stdout/progress from run wp7_evaluation_2166259. | Archive as execution evidence outside the published source tree. |
| 134 | `wp7_evaluation_2166349.err` | Added | **Keep privately** | stderr/warnings from run wp7_evaluation_2166349. | Archive as execution evidence outside the published source tree. |
| 135 | `wp7_evaluation_2166349.out` | Added | **Keep privately** | stdout/progress from run wp7_evaluation_2166349. | Archive as execution evidence outside the published source tree. |
| 136 | `wp7_hdf5_gt.patch` | Added | **Delete** | Patch used to deliver the HDF5 ground-truth correction. | Retire after confirming all intended hunks are integrated. |
| 137 | `wp8.patch` | Added | **Delete** | Patch used to deliver overfit diagnostics. | Retire after confirming all intended hunks are integrated. |
| 138 | `wp8_overfit_2166629.err` | Added | **Keep privately** | stderr/warnings from run wp8_overfit_2166629. | Archive as execution evidence outside the published source tree. |
| 139 | `wp8_overfit_2166629.out` | Added | **Keep privately** | stdout/progress from run wp8_overfit_2166629. | Archive as execution evidence outside the published source tree. |
| 140 | `wp8_overfit_2166761.err` | Added | **Keep privately** | stderr/warnings from run wp8_overfit_2166761. | Archive as execution evidence outside the published source tree. |
| 141 | `wp8_overfit_2166761.out` | Added | **Keep privately** | stdout/progress from run wp8_overfit_2166761. | Archive as execution evidence outside the published source tree. |
| 142 | `radargen-review-ZxtBXS.tar.gz` | Generated; outside supplied Git diff | **Delete** | Inventory/diff archive created for this review and subsequently copied into the repository. | User-requested extra item; absent from the captured inventory because it was created/copied afterward. Remove after this review is preserved. |

## Suggested order

First archive private records and remove redundant delivery artifacts. Next consolidate preprocessing runners and imports, generalize paths, and document the missing offline setup dependency. Finally assemble the publication tree and run focused installation, preprocessing, training/resume and evaluation checks. Keep working on the current branch until the publication version is ready; the classifications above do not require deleting useful development tooling immediately.
