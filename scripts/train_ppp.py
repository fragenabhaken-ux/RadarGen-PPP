#!/usr/bin/env python3
"""WP1: validate PPP configuration; training is deliberately not implemented."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyrallis

from diffusion.utils.config import SanaConfig


def validate_ppp_config(config: SanaConfig) -> None:
    """Check the roadmap's configuration contract without runtime initialization."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0

    require(config.data.type == "RadarGenPPPDatasetWrapper", "data.type must be RadarGenPPPDatasetWrapper")
    require(bool(config.data.dataset_dir) and bool(config.data.ppp_data_dir), "data.dataset_dir and data.ppp_data_dir are required")
    require(not config.data.load_text_feat and not config.data.load_vae_feat, "PPP requires load_text_feat=false and load_vae_feat=false")
    data = config.data.extra
    require(isinstance(data, dict), "data.extra must be a dictionary")
    require(data.get("dataset_name") == "truckscenes", "data.extra.dataset_name must be truckscenes")
    require(data.get("dataset_version") == "v1.2-trainval", "data.extra.dataset_version must be v1.2-trainval")
    require(data.get("eval_split") in ("train", "val"), "data.extra.eval_split must be train or val")
    require(isinstance(data.get("camera_views"), list) and bool(data["camera_views"]), "data.extra.camera_views must be a nonempty list")
    for key in ("coordinate_range", "camera_freq"):
        require(positive(data.get(key)), f"data.extra.{key} must be positive and finite")
    for prefix in ("rcs", "doppler"):
        bounds = [data.get(f"{prefix}_{suffix}") for suffix in ("min", "max")]
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds), f"{prefix} bounds must be finite numbers")
        require(bounds[0] < bounds[1], f"{prefix}_min must be less than {prefix}_max")
    require(config.model.model == "RadarGenPPP_600M_P1_D28", "model.model must be RadarGenPPP_600M_P1_D28")
    require(config.data.image_size == config.model.image_size == 512, "PPP data/model image_size must be 512")
    model = config.model.extra
    require(isinstance(model, dict), "model.extra must be a dictionary")
    require(model.get("num_maps") == model.get("num_conditions") == 3, "PPP requires three modalities and three conditions")
    require(model.get("fixed_timestep") == 0.0, "model.extra.fixed_timestep must be 0.0")
    require(config.model.mixed_precision == "bf16", "PPP mixed_precision must be bf16")
    require(config.vae.vae_type == "dc-ae" and config.vae.weight_dtype == "float32", "PPP requires dc-ae with float32 weights")
    require(config.vae.vae_latent_dim == config.vae.vae_downsample_rate == 32, "PPP requires 32 latent channels and downsample rate 32")
    require(bool(config.vae.vae_pretrained), "vae.vae_pretrained is required")
    losses = config.train.extra
    require(isinstance(losses, dict), "train.extra must be a dictionary")
    for key in ("ppp_weight", "rcs_weight", "doppler_weight"):
        require(positive(losses.get(key)), f"train.extra.{key} must be positive and finite")
    require(losses.get("normalize_ppp_by_gt_count") is True, "normalize_ppp_by_gt_count must be true")
    require(losses.get("velocity_loss_mode") == "signed_l1", "velocity_loss_mode must be signed_l1 until a circular period is established")
    for key in ("train_batch_size", "gradient_accumulation_steps", "num_epochs"):
        require(getattr(config.train, key) > 0, f"train.{key} must be positive")
    require(config.train.num_workers >= 0, "train.num_workers must be nonnegative")
    require(not config.train.use_fsdp, "PPP initially supports the DDP configuration only")
    require(bool(config.work_dir), "root-level work_dir is required")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--validate-only", action="store_true", help="validate configuration and exit without runtime initialization")
    options, config_args = parser.parse_known_args()
    try:
        config = pyrallis.parse(config_class=SanaConfig, args=config_args)
        validate_ppp_config(config)
    except (ValueError, TypeError) as error:
        print(f"PPP configuration error: {error}", file=sys.stderr)
        return 2
    if options.validate_only:
        print("PPP configuration valid (configuration-only; HDF5 contents are not checked).")
        print(config)
        return 0
    print("PPP training is not implemented yet (work package 1 skeleton). No runtime initialization was performed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
