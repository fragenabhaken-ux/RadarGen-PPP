"""Epoch seed, distributed coverage and resume through the prepared loader."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from radargen.training.ppp_sampler import ResumableDistributedShuffleSampler


class ShuffleTests(unittest.TestCase):
    def test_global_seed_epoch_shards_and_resume(self):
        for epoch in (0, 1, 7):
            shards = []
            for rank in range(4):
                sampler = ResumableDistributedShuffleSampler(range(19), num_replicas=4, rank=rank, seed=42)
                sampler.set_epoch(epoch)
                full = list(sampler)
                sampler.set_start(2)
                self.assertEqual(list(sampler), full[2:])
                self.assertEqual(len(sampler), 5)
                shards.append(full)
            expected = torch.randperm(19, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            interleaved = [shards[r][i] for i in range(5) for r in range(4)]
            self.assertEqual(interleaved, expected + expected[:1])
            self.assertEqual(len(set(interleaved)), 19)

    def test_training_resume_matches_across_epoch_boundary(self):
        from test_dino_ppp_training import small_training_config, TinyDirectPPP, TinyBEVDataset
        from scripts.train_ppp import run_training

        class RecordingModel(TinyDirectPPP):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, appearance, semantics, velocity):
                self.seen.extend((appearance[:, 0, 0, 0] * 16 - 1).round().long().tolist())
                return super().forward(appearance, semantics, velocity)

        config = small_training_config()
        config.train.train_batch_size = 4
        config.train.gradient_accumulation_steps = 1
        config.train.extra['shuffle_each_epoch'] = True
        torch.manual_seed(123)
        initial = RecordingModel()
        full, split, resumed = [copy.deepcopy(initial) for _ in range(3)]
        options = SimpleNamespace(smoke_check=False, max_updates=6, resume=None)
        with tempfile.TemporaryDirectory() as directory:
            config.work_dir = str(Path(directory) / 'full')
            run_training(config, options, components=SimpleNamespace(model=full, condition_encoder=None), dataset=TinyBEVDataset())
            expected = []
            for epoch in (0, 1):
                expected.extend(torch.randperm(16, generator=torch.Generator().manual_seed(config.train.seed + epoch)).tolist())
            self.assertEqual(full.seen, expected[:24])
            config.work_dir = str(Path(directory) / 'split')
            options.max_updates = 3
            first = run_training(config, options, components=SimpleNamespace(model=split, condition_encoder=None), dataset=TinyBEVDataset())
            options.resume = first['checkpoint']
            run_training(config, options, components=SimpleNamespace(model=resumed, condition_encoder=None), dataset=TinyBEVDataset())
            self.assertEqual(split.seen + resumed.seen, full.seen)
            for key, value in full.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], value, atol=0, rtol=0)
