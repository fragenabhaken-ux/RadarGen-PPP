"""Discrete PPP sampling at BEV pixel centers, in physical units."""
import math
import numpy as np
import torch


def _grid(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating tensor')
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f'{name} must have shape (H,W) or (1,H,W)')
    # Sampling is intentionally on CPU with a dedicated CPU generator.
    value = value.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')
    return value


def ppp_cell_masses(log_intensity):
    grid = _grid(log_intensity, 'log_intensity')
    masses = grid.exp() / grid.numel()
    if not torch.isfinite(masses).all() or not torch.isfinite(masses.sum()):
        raise ValueError('PPP cell masses or total intensity are nonfinite')
    return masses


@torch.no_grad()
def sample_ppp_grid(log_intensity, raw_rcs, raw_doppler, coordinate_range,
                    generator=None):
    """Return float64 (N,4) [x meters,y meters,RCS dBsm,Doppler m/s].

    Multiple events per cell are retained. A CPU generator is required so
    callers cannot accidentally use the global random stream.
    """
    if generator is None or generator.device.type != 'cpu':
        raise ValueError('Supply a dedicated seeded CPU torch.Generator')
    radius = float(coordinate_range)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('coordinate_range must be positive and finite')
    masses = ppp_cell_masses(log_intensity)
    rcs, doppler = _grid(raw_rcs, 'raw_rcs'), _grid(raw_doppler, 'raw_doppler')
    if rcs.shape != masses.shape or doppler.shape != masses.shape:
        raise ValueError('All scalar fields must have matching shapes')
    rcs, doppler = rcs.pow(3), doppler.pow(3)
    if not torch.isfinite(rcs).all() or not torch.isfinite(doppler).all():
        raise ValueError('Physical marks are nonfinite after cubing')
    counts = torch.poisson(masses, generator=generator)
    if not torch.isfinite(counts).all() or (counts >= 2**63).any() or counts.sum() >= 2**63:
        raise ValueError('Sampled counts exceed representable output size')
    indices = torch.repeat_interleave(torch.arange(masses.numel()), counts.flatten().long())
    if indices.numel() == 0:
        return np.empty((0, 4), dtype=np.float64)
    height, width = masses.shape
    rows, cols = indices // width, indices % width
    x = ((cols.double() + 0.5) / width) * (2 * radius) - radius
    y = ((height - rows.double() - 0.5) / height) * (2 * radius) - radius
    return torch.stack((x, y, rcs.flatten()[indices], doppler.flatten()[indices]), dim=1).numpy()
