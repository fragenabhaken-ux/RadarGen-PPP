from diffusion.data.builder import DATASETS


@DATASETS.register_module()
class RadarGenPPPDatasetWrapper:
    """Registry entry; import the HDF5 implementation only when selected."""

    def __new__(cls, transform=None, resolution=512, config=None, **kwargs):
        from radargen.training.radargen_ppp_dataset import build_radargen_ppp_dataset_from_config

        return build_radargen_ppp_dataset_from_config(config, resolution=resolution)
