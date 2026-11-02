# Persistent validation dataset index

Validation automatically prepares a JSON manifest on its first launch and reuses it
on subsequent launches. The default location is
`output/ppp_manifests/validation_<dataset-contract-sha256>.json` in the repository.
The key includes the dataset roots, split, preprocessing signature, normalization,
and other indexing settings, so model checkpoints share an index for identical data.
The first launch still performs the full scan; an already running validation does
not publish this new index.

Reuse checks the manifest checksum and configuration plus each scene completion
record, without opening every HDF5 file. Selected samples retain file size/mtime
and metadata checks on access. The TruckScenes adapter is still initialized for
annotations and plotting. This assumes immutable preprocessed data.

Use `--dataset-index /path/to/index.json` to override the location, or set
`dataset_index_path` in the validation YAML. Use `--rebuild-dataset-index` to
explicitly repeat the full scan and atomically replace an existing index. A changed
or incompatible index fails with rebuild guidance rather than silently rescanning.
Both bounded and full-set validation use the same index.

This reuses the existing training manifest format, which requires nonempty target
frames. A dataset containing empty target frames cannot currently be indexed with
this format.
