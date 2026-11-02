"""Deterministic global epoch shuffling with rank-local resume offsets."""
from itertools import islice

from torch.utils.data import DistributedSampler


class ResumableDistributedShuffleSampler(DistributedSampler):
    """Shuffle with seed+epoch before rank sharding; pad like DistributedSampler.

    Each rank has equally many samples. At most world_size-1 entries repeat
    to cover a dataset whose length does not divide the number of ranks.
    set_start skips already consumed rank-local samples without changing length.
    """

    def __init__(self, dataset, *, num_replicas, rank, seed):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=seed, drop_last=False)
        self.step_start = 0

    def set_start(self, start):
        if type(start) is not int or not 0 <= start <= self.num_samples:
            raise ValueError('Sampler start must be within the rank-local epoch')
        self.step_start = start

    def __iter__(self):
        return islice(super().__iter__(), self.step_start, None)
