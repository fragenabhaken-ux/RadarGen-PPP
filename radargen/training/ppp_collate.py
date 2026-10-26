"""Stack conditioning frames and pack variable-length continuous detections."""
import math

import torch
from torch.utils.data import default_collate


def collate_ppp_samples(samples):
    """Return packed detections with frame indices local to this batch.

    Fixed-size tensors and metadata use default_collate. Batch size is explicit
    because the final frames (or all frames) may have no detections. Distributed
    sampling must happen before collation; never split the packed tensor by N.
    Legacy raster samples remain supported via ordinary default collation.
    """
    if not samples:
        raise ValueError('cannot collate an empty sample list')
    continuous = ['detections' in sample for sample in samples]
    if not any(continuous):
        return default_collate(samples)
    if not all(continuous):
        raise ValueError('cannot mix continuous and raster targets')
    limits = [float(sample['point_limit']) for sample in samples]
    if not math.isfinite(limits[0]) or limits[0] <= 0 or any(v != limits[0] for v in limits):
        raise ValueError('all frames must share a positive finite point_limit')
    points = [sample['detections'] for sample in samples]
    for value in points:
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.ndim != 2 or value.shape[1] != 4 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('sample detections must be finite CPU float32 (N,4)')
    batch = default_collate([{k: v for k, v in sample.items()
                              if k not in ('detections', 'point_limit')} for sample in samples])
    batch.update(detections=torch.cat(points, dim=0),
                 frame_indices=torch.cat([torch.full((len(value),), index, dtype=torch.long)
                                          for index, value in enumerate(points)]),
                 batch_size=len(samples), point_limit=limits[0])
    return batch
