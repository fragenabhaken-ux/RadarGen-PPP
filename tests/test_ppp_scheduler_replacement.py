"""Retain optimizer/progress while starting an explicitly requested no-warmup cosine."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from diffusion.utils.lr_scheduler import get_cosine_updates_schedule
from radargen.training.ppp_checkpoint import save_ppp_checkpoint, load_ppp_checkpoint


class SchedulerReplacementChecks(unittest.TestCase):
    def test_optimizer_progress_preserved_and_new_schedule_resumable(self):
        with TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            for _ in range(3):
                optimizer.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                scheduler.step()
            old_optimizer = copy.deepcopy(optimizer.state_dict())
            contract = dict(lr_schedule='constant', lr_args={'num_warmup_steps': 100}, optimizer={'lr': .01}, loss='original')
            progress = dict(epoch=2, batch_in_epoch=3, global_step=7)
            generator = torch.Generator().manual_seed(42)
            path = save_ppp_checkpoint(Path(directory)/'old', model, optimizer, scheduler,
                                       progress=progress, contract=contract, generator=generator)
            target = dict(contract, lr_schedule='cosine_updates', lr_args={'num_warmup_steps': 0, 'min_lr_ratio': .1}, scheduler_total_updates=10)
            new_model = torch.nn.Linear(2, 1)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=.01)
            new_scheduler = get_cosine_updates_schedule(new_optimizer, 10, 0, .1)
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_ppp_checkpoint(path, new_model, new_optimizer, new_scheduler, contract=target)
            _, _, _, rng = load_ppp_checkpoint(path, new_model, new_optimizer, new_scheduler,
                                               contract=target, replace_scheduler=True)
            self.assertEqual(rng['ppp_training']['progress'], progress)
            self.assertEqual(new_scheduler.last_epoch, 0)
            self.assertEqual(new_optimizer.param_groups[0]['lr'], .01)
            restored = new_optimizer.state_dict()
            for key, state in old_optimizer['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(restored['state'][key][field], value, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(new_model.state_dict()[key], value, rtol=0, atol=0)
            new_optimizer.zero_grad()
            new_model(torch.ones(1, 2)).sum().backward()
            new_optimizer.step()
            new_scheduler.step()
            self.assertLess(new_optimizer.param_groups[0]['lr'], .01)
            self.assertGreater(new_optimizer.param_groups[0]['lr'], .001)
            new_path = save_ppp_checkpoint(Path(directory)/'new', new_model, new_optimizer, new_scheduler,
                                           progress=dict(progress, global_step=8), contract=target, generator=generator)
            exact_model = torch.nn.Linear(2, 1)
            exact_optimizer = torch.optim.AdamW(exact_model.parameters(), lr=.01)
            exact_scheduler = get_cosine_updates_schedule(exact_optimizer, 10, 0, .1)
            load_ppp_checkpoint(new_path, exact_model, exact_optimizer, exact_scheduler, contract=target)
            self.assertEqual(exact_scheduler.state_dict(), new_scheduler.state_dict())
            self.assertEqual(exact_optimizer.param_groups[0]['lr'], new_optimizer.param_groups[0]['lr'])
            with self.assertRaisesRegex(ValueError, 'remaining training/data contract'):
                load_ppp_checkpoint(path, new_model, new_optimizer, get_cosine_updates_schedule(new_optimizer, 10, 0, .1),
                                    contract=dict(target, loss='different'), replace_scheduler=True)
            with self.assertRaisesRegex(ValueError, 'zero warmup'):
                load_ppp_checkpoint(path, new_model, new_optimizer, new_scheduler,
                                    contract=dict(target, lr_args={'num_warmup_steps': 100}), replace_scheduler=True)
