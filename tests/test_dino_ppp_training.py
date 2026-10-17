"""CPU integration checks for direct BEV prediction and the existing PPP losses."""

from contextlib import nullcontext
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyrallis
import torch
from torch import nn
from torch.utils.data import Dataset

from diffusion.utils.config import SanaConfig
from radargen.losses.ppp_loss import PPPLoss
from scripts.train_ppp import compute_ppp_losses, run_training, training_microbatch, validate_ppp_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/PPP_train_dinov3.yaml'


def direct_config():
    with CONFIG_PATH.open() as stream:
        return pyrallis.load(SanaConfig, stream)


def small_training_config():
    config = direct_config()
    config.model.image_size = config.data.image_size = 2
    config.model.mixed_precision = 'no'
    config.train.train_batch_size = 1
    config.train.num_workers = 0
    config.train.num_epochs = 2
    config.train.gradient_accumulation_steps = 2
    config.train.auto_lr = {}
    config.train.lr_schedule_args['num_warmup_steps'] = 0
    config.train.optimizer['lr'] = 0.001
    config.train.extra['prediction_interval'] = 0
    config.train.extra['overfit_diagnostic_interval'] = 0
    config.report_to = 'none'
    return config


class TinyDirectPPP(nn.Module):
    """Small replacement for DINO that preserves the three-map input contract."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(9, 8, kernel_size=1)
        self.output = nn.Conv2d(8, 3, kernel_size=1)
        self.last_conditions = None

    def forward(self, appearance, semantics, velocity):
        self.last_conditions = tuple(value.detach().clone() for value in
                                     (appearance, semantics, velocity))
        features = self.features(torch.cat((appearance, semantics, velocity), dim=1))
        return self.output(features.tanh())


class TinyBEVDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        image = torch.full((3, 2, 2), (index + 1) / 16)
        mask = torch.tensor([[[1., 0.], [0., 1.]]])
        return dict(bev_color_map=image, bev_seg_map=image * 0.5,
                    bev_velocity_map=-image, point_mask=mask,
                    rcs_target=mask * 3, doppler_target=mask * -2, data_info={})


class DirectPPPTrainingChecks(unittest.TestCase):
    def test_three_raw_channels_use_existing_ppp_and_mark_links(self):
        settings = dict(ppp_weight=2., rcs_weight=3., doppler_weight=4.,
                        velocity_loss_mode='signed_l1')
        raw = torch.empty(1, 3, 2, 2)
        raw[:, 0] = math.log(2.)
        raw[:, 1] = -2.
        raw[:, 2] = 0.5
        mask = torch.tensor([[[[1., 0.], [1., 0.]]]])
        batch = dict(point_mask=mask, rcs_target=torch.full_like(mask, -5.),
                     doppler_target=torch.full_like(mask, 0.25))
        direct = raw.clone().requires_grad_()
        legacy = raw.unsqueeze(2).clone().requires_grad_()
        ppp_loss = PPPLoss(out_img_size=(2, 2))

        loss, logs = compute_ppp_losses(direct, batch, ppp_loss, settings)
        legacy_loss, legacy_logs = compute_ppp_losses(legacy, batch, ppp_loss, settings)

        # Intensity exp(log(2)) gives total mass 2; both marks are cubed once.
        torch.testing.assert_close(logs['loss_ppp_raw'], torch.tensor(1. + math.log(2.)))
        torch.testing.assert_close(logs['loss_rcs_raw'], torch.tensor(3.))
        torch.testing.assert_close(logs['loss_doppler_raw'], torch.tensor(0.125))
        expected = 2. * (1. + math.log(2.)) + 3. * 3. + 4. * 0.125
        torch.testing.assert_close(loss, torch.tensor(expected))
        torch.testing.assert_close(loss, legacy_loss)
        for name, value in logs.items():
            torch.testing.assert_close(value, legacy_logs[name])
        loss.backward()
        legacy_loss.backward()
        torch.testing.assert_close(direct.grad, legacy.grad.squeeze(2))
        self.assertTrue(torch.isfinite(direct.grad).all())
        self.assertTrue((direct.grad[:, 0] != 0).all())
        self.assertTrue((direct.grad[:, 1:] * mask).abs().sum() > 0)
        self.assertEqual((direct.grad[:, 1:] * (1. - mask)).abs().sum().item(), 0.)

    def test_microbatch_passes_bev_maps_directly_and_preserves_condition_dropout(self):
        settings = dict(ppp_weight=1., rcs_weight=1., doppler_weight=1.,
                        velocity_loss_mode='signed_l1', conditioning_dropout=0.5)
        # No VAE or text configuration is needed by this training branch.
        config = SimpleNamespace(model=SimpleNamespace(model='DinoPPP'),
                                 train=SimpleNamespace(extra=settings))
        batch = dict(bev_color_map=torch.ones(1, 3, 2, 2),
                     bev_seg_map=torch.full((1, 3, 2, 2), 2.),
                     bev_velocity_map=torch.full((1, 3, 2, 2), 3.),
                     point_mask=torch.ones(1, 1, 2, 2),
                     rcs_target=torch.full((1, 1, 2, 2), 3.),
                     doppler_target=torch.full((1, 1, 2, 2), -2.))
        model = TinyDirectPPP()
        encoder = Mock(side_effect=AssertionError('Direct prediction must not encode latents'))
        text = Mock(side_effect=AssertionError('Direct prediction must not encode text'))
        accelerator = SimpleNamespace(device=torch.device('cpu'), autocast=nullcontext)

        with patch('random.random', side_effect=[0.1, 0.9, 0.1]):
            loss, _ = training_microbatch(model, encoder, batch, text,
                                         PPPLoss(), config, accelerator)
        encoder.assert_not_called()
        text.assert_not_called()
        torch.testing.assert_close(model.last_conditions[0], torch.zeros_like(batch['bev_color_map']))
        torch.testing.assert_close(model.last_conditions[1], batch['bev_seg_map'])
        torch.testing.assert_close(model.last_conditions[2], torch.zeros_like(batch['bev_velocity_map']))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertTrue(model.output.weight.grad.abs().sum() > 0)

    def test_checkpoint_initialization_needs_no_rgb_pretrained_assets(self):
        from radargen.training.ppp_initialization import initialize_dino_ppp_model

        config = SimpleNamespace(model=SimpleNamespace(extra={
            'checkpoint_path': '/missing/dinov3_rgb_checkpoint.pth', 'neck_layers': 1}),
            train=SimpleNamespace(grad_checkpointing=True))
        model = TinyDirectPPP()
        with patch('radargen.models.dino_ppp.DinoPPP', return_value=model) as constructor:
            components = initialize_dino_ppp_model(config, load_pretrained=False)
        self.assertIs(components.model, model)
        self.assertIsNone(components.condition_encoder)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs['checkpoint_path'], '')
        self.assertFalse(kwargs['pretrained'])
        self.assertTrue(kwargs['grad_checkpointing'])
        self.assertEqual(config.model.extra['checkpoint_path'], '/missing/dinov3_rgb_checkpoint.pth')

    def test_direct_config_does_not_require_latent_or_text_assets(self):
        config = direct_config()
        validate_ppp_config(config)
        self.assertEqual(config.model.model, 'DinoPPP')
        config.vae = config.text_encoder = config.scheduler = None
        validate_ppp_config(config)

    def test_direct_training_resumes_exactly_without_encoder_or_text_provider(self):
        config = small_training_config()
        options = SimpleNamespace(smoke_check=False, max_updates=2, resume=None)
        torch.manual_seed(123)
        initial = TinyDirectPPP()
        initial.features.requires_grad_(False)
        continuous = SimpleNamespace(model=copy.deepcopy(initial), condition_encoder=None)
        split = SimpleNamespace(model=copy.deepcopy(initial), condition_encoder=None)
        restored = SimpleNamespace(model=copy.deepcopy(initial), condition_encoder=None)

        with TemporaryDirectory() as directory, patch('scripts.train_ppp.load_local_text_provider',
                side_effect=AssertionError('Direct training must not load a text encoder')):
            config.work_dir = str(Path(directory) / 'continuous')
            full_options = copy.copy(options)
            full_options.max_updates = 4
            full = run_training(config, full_options, components=continuous, dataset=TinyBEVDataset())
            config.work_dir = str(Path(directory) / 'split')
            first = run_training(config, options, components=split, dataset=TinyBEVDataset())
            self.assertEqual(first['progress']['global_step'], 2)
            self.assertTrue(any(not torch.equal(parameter, split.model.state_dict()[name])
                                for name, parameter in initial.state_dict().items()))
            state = torch.load(first['checkpoint'], map_location='cpu', weights_only=False)
            self.assertEqual(len(state['optimizer']['state']),
                             sum(parameter.requires_grad for parameter in split.model.parameters()))
            for name, parameter in initial.features.state_dict().items():
                torch.testing.assert_close(parameter, split.model.features.state_dict()[name], rtol=0, atol=0)
            contract = state['ppp_training']['contract']
            for unused in ('vae', 'text_encoder', 'null_embed_path', 'text_model_dir'):
                self.assertNotIn(unused, contract)
            options.resume = first['checkpoint']
            resumed = run_training(config, options, components=restored, dataset=TinyBEVDataset())
            self.assertEqual(resumed['progress'], full['progress'])
            for name, parameter in continuous.model.state_dict().items():
                torch.testing.assert_close(parameter, restored.model.state_dict()[name], rtol=0, atol=0)
            full_state = torch.load(full['checkpoint'], map_location='cpu', weights_only=False)
            resumed_state = torch.load(resumed['checkpoint'], map_location='cpu', weights_only=False)
            self.assertEqual(full_state['scheduler'], resumed_state['scheduler'])


    def test_direct_smoke_check_restores_trainable_parameters(self):
        config = small_training_config()
        model = TinyDirectPPP()
        model.features.requires_grad_(False)
        components = SimpleNamespace(model=model, condition_encoder=None)
        options = SimpleNamespace(smoke_check=True, max_updates=3, resume=None)
        with TemporaryDirectory() as directory:
            config.work_dir = directory
            result = run_training(config, options, components=components, dataset=TinyBEVDataset())
        self.assertEqual(result['progress']['global_step'], 3)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
