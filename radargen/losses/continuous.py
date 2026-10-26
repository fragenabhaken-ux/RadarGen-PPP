"""Detection-wise losses; no target deduplication or mark rasterization.

The caller supplies packed (N,4) [x,y,RCS,Doppler] detections and (N,)
frame indices. B is taken from predictions, including empty frames. Batching
and training integration are deliberately separate from this loss interface.
"""
import torch
from torch import nn

from .ppp_loss import PPPLoss, _require_finite


class ContinuousPPPLoss(nn.Module):
    """Return unweighted per-batch PPP, RCS L1 and signed Doppler L1 losses."""

    def __init__(self, out_img_size=(512, 512), normalize_by_gt_count=True, *, ppp_loss=None):
        super().__init__()
        self.ppp = ppp_loss if ppp_loss is not None else PPPLoss(out_img_size, normalize_by_gt_count)

    def forward(self, fields, detections, frame_indices, *, point_limit):
        if (not isinstance(fields, torch.Tensor) or not fields.is_floating_point()
                or fields.ndim != 4 or fields.shape[1] != 3 or any(n == 0 for n in fields.shape)):
            raise ValueError('fields must be nonempty floating (B,3,H,W)')
        if (not isinstance(detections, torch.Tensor) or not detections.is_floating_point()
                or detections.ndim != 2 or detections.shape[1] != 4 or detections.device != fields.device):
            raise ValueError('detections must be floating (N,4) on prediction device')
        if (not isinstance(frame_indices, torch.Tensor) or frame_indices.dtype != torch.long
                or frame_indices.shape != (len(detections),) or frame_indices.device != fields.device):
            raise ValueError('frame_indices must be int64 (N,) on prediction device')
        b, _, h, w = fields.shape
        if ((frame_indices < 0) | (frame_indices >= b)).any():
            raise ValueError('frame index outside prediction batch')
        for name, value in (('fields', fields), ('detections', detections)):
            _require_finite(value, name)
        with torch.autocast(device_type=fields.device.type, enabled=False):
            pred, target = fields.float(), detections.float()
            _require_finite(pred, 'FP32 fields')
            _require_finite(target, 'FP32 detections')
            limit = torch.as_tensor(point_limit, device=fields.device, dtype=torch.float32)
            if limit.ndim != 0 or not torch.isfinite(limit) or limit <= 0:
                raise ValueError('point_limit must be a positive finite scalar')
            # Match the existing BEV extent: exclude outside, retain both edges.
            inside = (target[:, :2].abs() <= limit).all(dim=1)
            target, frames = target[inside], frame_indices[inside]
            continuous = (target[:, :2] + limit) / (2 * limit)
            cols = (continuous[:, 0] * w).floor().long().clamp(0, w - 1)
            rows = h - 1 - (continuous[:, 1] * h).floor().long().clamp(0, h - 1)
            # Repeated indices intentionally remain repeated in this gather.
            sampled = pred[frames, :, rows, cols]
            counts = torch.bincount(frames, minlength=b).float()
            denominator = counts.clamp_min(1)

            def per_frame_sum(values):
                return values.new_zeros(b).scatter_add(0, frames, values)

            area = self.ppp.pixel_scale.to(fields.device)
            integral = (pred[:, 0].exp() / area).sum(dim=(1, 2))
            observation = per_frame_sum(sampled[:, 0] - area.log())
            ppp = integral - observation
            if self.ppp.normalize_by_gt_count:
                ppp = ppp / denominator
            physical_marks = sampled[:, 1:].pow(3)
            _require_finite(physical_marks, 'physical marks')
            errors = (physical_marks - target[:, 2:]).abs()
            losses = dict(ppp=ppp.mean(),
                          rcs=(per_frame_sum(errors[:, 0]) / denominator).mean(),
                          doppler=(per_frame_sum(errors[:, 1]) / denominator).mean())
            for name, value in losses.items():
                _require_finite(value, name)
            return losses
