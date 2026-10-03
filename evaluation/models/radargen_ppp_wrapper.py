"""PPP evaluation with explicit HDF5 conditioning and identity-checked fields."""
import numpy as np
from evaluation.models.registry import register_model
from evaluation.protocols import ModelWrapper


@register_model('radargen_ppp')
def create_radargen_ppp(config, adapter, **kwargs):
    return RadarGenPPPWrapper(config, adapter, evaluation_config=kwargs.get('evaluation_config'))


class RadarGenPPPWrapper(ModelWrapper):
    def __init__(self, config, adapter, *, evaluation_config=None):
        import pyrallis
        from diffusion.utils.config import SanaConfig
        from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
        with open(config.config_path) as stream:
            self.config = pyrallis.load(SanaConfig, stream)
        self.adapter = adapter
        params = config.params or {}
        self.split = evaluation_config.eval_split if evaluation_config else params.get('eval_split', 'val')
        self.hdf5_root = params.get('ppp_data_dir', self.config.data.ppp_data_dir)
        if params.get('bev_maps_dir'):
            raise ValueError('Use ppp_data_dir for HDF5; legacy separate-map inference is not implemented here')
        for key, value in vars(adapter.normalization).items():
            if self.config.data.extra.get(key) != value:
                raise ValueError(f'PPP/evaluator normalization mismatch: {key}')
        if evaluation_config and (evaluation_config.dataset_name != self.config.data.extra['dataset_name']
                                  or evaluation_config.dataset_version != self.config.data.extra['dataset_version']):
            raise ValueError('PPP/evaluator dataset mismatch')
        self.pipeline = RadarGenPPPPipeline.from_config(self.config,
            checkpoint_path=config.checkpoint_path,
            sana_checkpoint_path=params['sana_checkpoint_path'],
            null_embed_path=params['null_embed_path'], text_model_dir=params['text_model_dir'],
            device=config.device, seed=params.get('seed', 42))
        self._dataset = None
        self._scene = None
        self._cached = None

    @property
    def name(self):
        return 'RadarGenPPP'

    def _prepare_pair(self, adapter, sample_data, sample_data_next, scene_token, frame_idx):
        from radargen.training.radargen_ppp_dataset import RadarGenPPPDataset
        if adapter is not self.adapter or scene_token is None or frame_idx is None or sample_data_next is None:
            raise ValueError('PPP HDF5 evaluation requires the configured adapter and full pair identity')
        if self._scene != scene_token:
            self._dataset = RadarGenPPPDataset(adapter, ppp_data_dir=self.hdf5_root,
                split=self.split, processing_signature=self.config.data.extra['ppp_processing_signature'],
                scene_tokens=[scene_token], resolution=self.config.model.image_size)
            self._scene = scene_token
        if frame_idx < 0 or frame_idx >= len(self._dataset):
            raise ValueError('PPP frame index outside completed pair inventory')
        # Compare the actual pair, not just the scene and integer index.
        from jsc_jupiter.hdf5_samples import pair_metadata
        expected = self._dataset.frame_index[frame_idx][1]
        supplied = pair_metadata(adapter, sample_data, sample_data_next, self.split)
        for key, value in supplied.items():
            if key not in expected or not np.array_equal(expected[key], value):
                raise ValueError(f'Evaluator/HDF5 pair identity mismatch: {key}')

    def get_ground_truth_point_cloud(self, adapter, sample_data, sample_data_next,
                                     scene_token, frame_idx):
        import h5py
        from radargen.training.radargen_ppp_dataset import validate_ppp_metadata
        from evaluation.ppp_ground_truth import read_continuous_targets
        self._prepare_pair(adapter, sample_data, sample_data_next, scene_token, frame_idx)
        path, expected, size, mtime = self._dataset.frame_index[frame_idx]
        stat = path.stat()
        if path.is_symlink() or (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f'{path}: indexed immutable sample changed')
        with h5py.File(path, 'r') as handle:
            validate_ppp_metadata(handle, expected, resolution=self._dataset.resolution,
                point_limit=self._dataset.point_limit, normalization=self._dataset.normalization)
            return read_continuous_targets(handle, self._dataset.point_limit)

    def predict_point_cloud(self, adapter, sample_data, sample_data_next=None,
                            scene_token=None, frame_idx=None):
        from radargen.ppp.sampling import ppp_cell_masses
        self._cached = None
        self._prepare_pair(adapter, sample_data, sample_data_next, scene_token, frame_idx)
        fields, _ = self.pipeline.predict_dataset_sample(self._dataset, frame_idx)
        points = self.pipeline.sample_fields(fields)
        self._cached = (scene_token, frame_idx, dict(cell_mass_grid=ppp_cell_masses(fields['log_intensity']).numpy(),
                                                     coordinate_range=self.pipeline.coordinate_range))
        return points

    def get_last_ppp_fields(self, scene_token, frame_idx):
        if self._cached is None or self._cached[:2] != (scene_token, frame_idx):
            raise ValueError('Requested PPP fields do not match the last prediction')
        fields = self._cached[2]
        return dict(cell_mass_grid=fields['cell_mass_grid'].copy(), coordinate_range=fields['coordinate_range'])
