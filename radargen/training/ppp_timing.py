"""Opt-in worker-side timings for the PPP loader diagnostic."""
import time

from torch.utils.data import Dataset

from radargen.training.ppp_collate import collate_ppp_samples


class TimedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        begin = time.perf_counter()
        sample = dict(self.dataset[index])
        sample['_sample_load_s'] = time.perf_counter() - begin
        return sample


def timed_collate_ppp_samples(samples):
    load_s = sum(sample.pop('_sample_load_s') for sample in samples)
    begin = time.perf_counter()
    batch = collate_ppp_samples(samples)
    batch['_worker_load_s'] = load_s
    batch['_worker_collate_s'] = time.perf_counter() - begin
    return batch
