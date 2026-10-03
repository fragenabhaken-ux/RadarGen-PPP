"""Masked physical-mark losses with cubic links and per-sample FP32 reduction."""

from numbers import Real

import torch

from .ppp_loss import _require_finite, _validate_inputs


def _physical_error(raw, target):
    prediction = raw.float().pow(3)
    _require_finite(prediction, "cubed physical prediction")
    error = prediction - target.float()
    _require_finite(error, "physical mark error")
    return error


def _masked_mean(error, point_mask):
    mask = point_mask.float()
    per_sample = (error.abs() * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1)
    _require_finite(per_sample, "per-sample mark loss")
    loss = per_sample.mean()
    _require_finite(loss, "mark loss")
    return loss


def masked_rcs_l1(raw_rcs: torch.Tensor, rcs_target: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
    """Masked |raw_rcs**3 - target_dBsm|; zero targets remain valid marks."""
    _validate_inputs(raw_rcs, point_mask, rcs_target)
    with torch.autocast(device_type=raw_rcs.device.type, enabled=False):
        return _masked_mean(_physical_error(raw_rcs, rcs_target), point_mask)


def _period_tensor(period, prediction):
    if isinstance(period, bool) or not isinstance(period, (Real, torch.Tensor)):
        raise ValueError("signed_circular requires an explicit positive Doppler period")
    if isinstance(period, torch.Tensor) and (period.is_complex() or period.dtype == torch.bool):
        raise ValueError("Doppler period must be real numeric data")
    # Only a scalar or one period per sample is allowed; no spatial broadcasting.
    value = torch.as_tensor(period, device=prediction.device)
    if value.ndim == 0:
        pass
    elif value.shape == (prediction.shape[0],):
        value = value.reshape(-1, 1, 1, 1)
    elif value.shape != (prediction.shape[0], 1, 1, 1):
        raise ValueError("Doppler period must be scalar, (B,), or (B,1,1,1)")
    _require_finite(value, "Doppler period")
    value = value.float()
    _require_finite(value, "FP32 Doppler period")
    if not (value > 0).all() or not (value / 2 > 0).all():
        raise ValueError("Doppler period must be positive and representable in FP32")
    return value


def masked_doppler_loss(raw_doppler: torch.Tensor, doppler_target: torch.Tensor,
                        point_mask: torch.Tensor, mode: str, doppler_period=None) -> torch.Tensor:
    """Masked cubed signed velocity error in m/s; periods must also be in m/s.

    signed_l1 uses absolute signed error. signed_circular uses shortest wrapped
    error with an explicitly supplied scalar or sample-specific period. No
    period is inferred from normalization bounds. A period is unused in l1 mode.
    """
    if mode not in ("signed_l1", "signed_circular"):
        raise ValueError("Doppler mode must be signed_l1 or signed_circular")
    _validate_inputs(raw_doppler, point_mask, doppler_target)
    with torch.autocast(device_type=raw_doppler.device.type, enabled=False):
        error = _physical_error(raw_doppler, doppler_target)
        if mode == "signed_circular":
            period = _period_tensor(doppler_period, raw_doppler)
            shifted = error + period / 2
            _require_finite(shifted, "shifted circular error")
            error = torch.remainder(shifted, period) - period / 2
            _require_finite(error, "circular Doppler error")
        return _masked_mean(error, point_mask)
