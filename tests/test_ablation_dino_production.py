"""Agreed production settings and 200-update warmup under the real DDP scheduler wrapper."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pyrallis
import torch
from accelerate.scheduler import AcceleratedScheduler
from diffusers import get_constant_schedule_with_warmup
from diffusion.utils.config import SanaConfig
from scripts.train_ppp import validate_ppp_config

ROOT=Path(__file__).resolve().parents[1]


class RandomDinoProductionTests(unittest.TestCase):
    def test_fresh_random_production_config_matches_agreement(self):
        with (ROOT/'configs/ablation_dino_random_init_train_4nodes.yaml').open() as handle:
            cfg=pyrallis.load(SanaConfig,handle)
        validate_ppp_config(cfg)
        self.assertEqual(cfg.train.train_batch_size,16)
        self.assertEqual(cfg.train.gradient_accumulation_steps,1)
        self.assertEqual(cfg.train.training_hours,11.8)
        self.assertEqual(cfg.train.optimizer['lr'],1e-4)
        self.assertEqual(cfg.train.lr_schedule,'constant')
        self.assertEqual(cfg.train.lr_schedule_args,{'num_warmup_steps':200})
        self.assertEqual(cfg.train.save_model_steps,1000)
        self.assertEqual(cfg.train.extra['keep_last_checkpoints'],0)
        self.assertTrue(cfg.train.extra['save_checkpoints'])
        self.assertFalse(cfg.train.extra['lr_range_test'])
        self.assertEqual(cfg.model.extra['backbone_initialization'],'random')
        self.assertFalse(cfg.model.extra['pretrained'])
        self.assertFalse(cfg.model.extra['checkpoint_path'])
        self.assertIsNone(cfg.resume_from)
        self.assertIsNone(cfg.model.resume_from)
        self.assertTrue(cfg.tensorboard)

    def test_constant_warmup_is_200_optimizer_updates_on_sixteen_ranks(self):
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
        optimizer.step_was_skipped=False
        # Mirrors the existing trainer: multiply warmup by world size, while
        # AcceleratedScheduler steps the native scheduler once per rank.
        scheduler=get_constant_schedule_with_warmup(optimizer,num_warmup_steps=200*16)
        wrapped=AcceleratedScheduler(scheduler,optimizer,split_batches=False)
        with patch('accelerate.scheduler.AcceleratorState',return_value=SimpleNamespace(num_processes=16)):
            for update in range(1,202):
                optimizer.step();wrapped.step()
                expected=1e-4*min(update/200,1.)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected,places=12)
        state=wrapped.state_dict()
        self.assertEqual(state['last_epoch'],201*16)


if __name__=='__main__':unittest.main()
