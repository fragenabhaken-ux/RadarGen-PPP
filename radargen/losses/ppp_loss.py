"""Active reference PPP likelihood, with a fixed full-grid measure and FP32 math."""

import torch
from torch import nn

_NO_TARGET = object()


def _require_finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _validate_inputs(prediction, point_mask, target=_NO_TARGET):
    """Shared scalar-grid contract; validate before any precision conversion."""
    if not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point():
        raise ValueError("prediction must be a floating-point tensor")
    if prediction.ndim != 4 or prediction.shape[1] != 1 or any(n == 0 for n in prediction.shape):
        raise ValueError("prediction must have nonempty shape (B,1,H,W)")
    for name, value in (("prediction", prediction), ("point_mask", point_mask), ("target", target)):
        if name == "target" and target is _NO_TARGET:
            continue
        if not isinstance(value, torch.Tensor) or value.is_complex():
            raise ValueError(f"{name} must be a real tensor")
        if value.shape != prediction.shape or value.device != prediction.device:
            raise ValueError(f"{name} must match prediction shape and device")
        _require_finite(value, name)
    if not ((point_mask == 0) | (point_mask == 1)).all():
        raise ValueError("point_mask must be binary")


class PPPLoss(nn.Module):
    """Mean per-sample PPP NLL, optionally divided by max(occupied cells, 1).

    The full reference size defaults to (512,512), independent of the supplied
    tensor's spatial size. Smaller grids are useful for synthetic tests; their
    cell mass still uses the fixed full-grid area. No resizing occurs here.
    """

    def __init__(self, out_img_size: tuple[int, int] = (512, 512), normalize_by_gt_count: bool = True):
        super().__init__()
        if not isinstance(out_img_size, tuple) or len(out_img_size) != 2 or any(
                not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in out_img_size):
            raise ValueError("out_img_size must contain two positive integers")
        if not isinstance(normalize_by_gt_count, bool):
            raise ValueError("normalize_by_gt_count must be boolean")
        self.normalize_by_gt_count = normalize_by_gt_count
        self.register_buffer("pixel_scale", torch.tensor(out_img_size[0] * out_img_size[1], dtype=torch.float32))
        _require_finite(self.pixel_scale, "full-grid area")

    def forward(self, log_intensity: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(log_intensity, point_mask)
        with torch.autocast(device_type=log_intensity.device.type, enabled=False):
            logits, mask = log_intensity.float(), point_mask.float()
            _require_finite(logits, "FP32 log_intensity")
            area = self.pixel_scale.to(device=logits.device, dtype=torch.float32)
            _require_finite(area, "full-grid area")
            if area <= 0:
                raise ValueError("full-grid area must be positive")
            intensity = logits.exp()
            _require_finite(intensity, "exp(log_intensity)")
            cell_mass = intensity / area
            integral = cell_mass.sum(dim=(1, 2, 3))
            observation = (mask * (logits - area.log())).sum(dim=(1, 2, 3))
            _require_finite(cell_mass, "PPP cell masses")
            _require_finite(integral, "PPP integral")
            _require_finite(observation, "PPP observation term")
            per_sample = integral - observation
            if self.normalize_by_gt_count:
                per_sample = per_sample / mask.sum(dim=(1, 2, 3)).clamp_min(1)
            _require_finite(per_sample, "per-sample PPP loss")
            loss = per_sample.mean()
            _require_finite(loss, "PPP loss")
            return loss
