"""Optimizer-update timing and endpoints of the unified PPP cosine schedule."""
import unittest
from types import SimpleNamespace

import torch

from diffusion.utils.lr_scheduler import build_lr_scheduler, get_cosine_updates_schedule


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class CosineScheduleTests(unittest.TestCase):
    def optimizer(self):
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1e-3)

    def test_warmup_hold_decay_and_clamped_floor(self):
        optimizer = self.optimizer()
        scheduler = get_cosine_updates_schedule(optimizer, 100, 10, 0.1, 0.3)
        curve = [optimizer.param_groups[0]['lr']]
        for _ in range(110):
            optimizer.step()
            scheduler.step()
            curve.append(optimizer.param_groups[0]['lr'])
        self.assertEqual(curve[0], 0)
        self.assertAlmostEqual(curve[5], 5e-4)
        self.assertAlmostEqual(curve[10], 1e-3)
        self.assertAlmostEqual(curve[30], 1e-3)
        self.assertAlmostEqual(curve[65], 5.5e-4)
        self.assertAlmostEqual(curve[100], 1e-4)
        self.assertAlmostEqual(curve[110], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(curve[30:], curve[31:])))

    def test_duration_counts_accumulated_updates_and_partial_epoch_batch(self):
        optimizer = self.optimizer()
        config = Config(lr_schedule='cosine_updates', num_epochs=4,
                        gradient_accumulation_steps=2,
                        lr_schedule_args=dict(num_warmup_steps=2, min_lr_ratio=0.05))
        scheduler = build_lr_scheduler(config, optimizer, range(5), 1.0)
        # ceil(5 microbatches / 2 accumulation) * 4 epochs = 12 updates.
        for _ in range(12):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5)

    def test_resume_continues_same_curve(self):
        optimizer = self.optimizer()
        scheduler = get_cosine_updates_schedule(optimizer, 100, 10, 0.1, 0.28)
        for _ in range(43):
            optimizer.step()
            scheduler.step()
        resumed_optimizer = self.optimizer()
        resumed = get_cosine_updates_schedule(resumed_optimizer, 100, 10, 0.1, 0.28)
        resumed_optimizer.load_state_dict(optimizer.state_dict())
        resumed.load_state_dict(scheduler.state_dict())
        for _ in range(60):
            optimizer.step()
            scheduler.step()
            resumed_optimizer.step()
            resumed.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], resumed_optimizer.param_groups[0]['lr'])

    def test_early_floor_is_held_for_final_quarter(self):
        optimizer = self.optimizer()
        scheduler = get_cosine_updates_schedule(
            optimizer, 100, 10, 0.05, 0.0, decay_end_fraction=0.75)
        for update in range(1, 111):
            optimizer.step()
            scheduler.step()
            if update >= 75:
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5)

    def test_invalid_schedule(self):
        for kwargs in [dict(num_training_steps=0),
                       dict(num_training_steps=10, num_warmup_steps=10),
                       dict(num_training_steps=10, num_warmup_steps=0, min_lr_ratio=-1),
                       dict(num_training_steps=10, num_warmup_steps=0, decay_start_fraction=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                get_cosine_updates_schedule(self.optimizer(), **kwargs)
