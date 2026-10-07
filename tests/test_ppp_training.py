"""Focused WP5 integration tests with CPU substitute models/data/text only."""
from contextlib import nullcontext
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pyrallis
import torch
from torch import nn
from torch.utils.data import Dataset
from accelerate import Accelerator

from diffusion.utils.config import SanaConfig
from radargen.training.ppp_decoder import FrozenConditionEncoder, PPPDecoder
from radargen.training.radargen_ppp_training_model import RadarGenPPPTrainingModel
from radargen.training.ppp_checkpoint import load_ppp_checkpoint, restore_rng_state
from scripts.train_ppp import run_training, compute_ppp_losses, assert_optimizer_membership, training_microbatch
from radargen.losses.ppp_loss import PPPLoss


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(96,32,1)
        self.modality_pe = nn.Parameter(torch.full((3,32),0.1))
        self.calls = 0
    def forward(self, timestep, y, mask=None, data_info=None, bev_condition_maps=None):
        self.calls += 1
        assert (timestep == 0).all()
        x = self.proj(torch.cat(bev_condition_maps,1))
        return (x[:,None]+self.modality_pe[None,:,:,None,None]).flatten(0,1)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Conv2d(32,8,1)
        self.project_out = nn.Module()
        last = nn.Module()
        last.conv = nn.Conv2d(8,1,1)
        self.project_out.op_list = nn.ModuleList([last])
    def forward(self,x):
        return self.project_out.op_list[-1].conv(self.early(x).tanh())


class TinyDataset(Dataset):
    def __len__(self):
        return 16
    def __getitem__(self,index):
        image = torch.full((3,2,2),(index+1)/16)
        mask = torch.tensor([[[1.,0.],[0.,1.]]])
        return dict(bev_color_map=image,bev_seg_map=image,bev_velocity_map=image,
                    point_mask=mask,rcs_target=mask*3,doppler_target=mask*(-2),data_info={})


def components():
    return SimpleNamespace(model=RadarGenPPPTrainingModel(TinyDiT(),PPPDecoder(TinyDecoder(),0.41407)),
                           condition_encoder=FrozenConditionEncoder(nn.Conv2d(3,32,1),0.41407))


def text(batch_size,device):
    return torch.ones(batch_size*3,1,3,8,device=device),torch.ones(batch_size*3,1,1,3,device=device)


class TrainingChecks(unittest.TestCase):
    def test_weighting_and_zero_marks(self):
        settings=dict(ppp_weight=2.,rcs_weight=3.,doppler_weight=4.,velocity_loss_mode='signed_l1')
        raw=torch.full((2,3,1,2,2),0.5,requires_grad=True)
        batch=dict(point_mask=torch.ones(2,1,2,2),rcs_target=torch.zeros(2,1,2,2),
                   doppler_target=torch.zeros(2,1,2,2))
        loss,logs=compute_ppp_losses(raw,batch,PPPLoss(),settings)
        torch.testing.assert_close(logs['loss_rcs_raw'],torch.tensor(0.125))
        torch.testing.assert_close(loss,2*logs['loss_ppp_raw']+3*logs['loss_rcs_raw']+4*logs['loss_doppler_raw'])
        loss.backward()
        self.assertTrue(raw.grad[:,1:].abs().sum()>0)

    def test_accumulated_step_matches_manual_reference(self):
        from accelerate.utils import set_seed
        from diffusion.utils.optimizer import build_optimizer
        from torch.utils.data import DataLoader
        with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
            config=pyrallis.load(SanaConfig,stream)
        config.model.image_size=config.data.image_size=2
        config.vae.vae_downsample_rate=1
        config.model.mixed_precision='no'
        config.train.train_batch_size=1
        config.train.num_workers=0
        config.train.gradient_accumulation_steps=2
        config.train.auto_lr={}
        config.train.lr_schedule_args={'num_warmup_steps':0}
        config.report_to='none'
        config.train.optimizer['lr']=0.001
        torch.manual_seed(123)
        manual=components()
        integrated=copy.deepcopy(manual)
        optimizer=build_optimizer(manual.model,copy.deepcopy(config.train.optimizer))
        assert_optimizer_membership(manual.model,manual.condition_encoder,optimizer)
        invalid=torch.optim.SGD(list(manual.model.parameters())+list(manual.condition_encoder.parameters()),lr=0.001)
        with self.assertRaisesRegex(ValueError,'exactly once'):
            assert_optimizer_membership(manual.model,manual.condition_encoder,invalid)
        set_seed(config.train.seed)
        fake=SimpleNamespace(device=torch.device('cpu'),autocast=nullcontext)
        manual.model.train()
        for index,batch in enumerate(DataLoader(TinyDataset(),batch_size=1)):
            loss,_=training_microbatch(manual.model,manual.condition_encoder,batch,text,PPPLoss(),config,fake)
            (loss/2).backward()
            if index==1:
                break
        torch.nn.utils.clip_grad_norm_(manual.model.parameters(),config.train.gradient_clip)
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            config.work_dir=directory
            opts=SimpleNamespace(smoke_check=False,max_updates=1,resume=None,
                                 null_embed_path='substitute-test-only',text_model_dir='substitute-test-only')
            result=run_training(config,opts,components=integrated,dataset=TinyDataset(),text_provider=text)
            self.assertEqual(result['progress']['batch_in_epoch'],2)
            self.assertEqual(integrated.model.dit.calls,2)
            for name,parameter in manual.model.state_dict().items():
                torch.testing.assert_close(parameter,integrated.model.state_dict()[name],rtol=0,atol=0)

    def test_manifest_identity_survives_subset_and_resume(self):
        with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
            config = pyrallis.load(SanaConfig, stream)
        config.model.image_size = config.data.image_size = 2
        config.vae.vae_downsample_rate = 1
        config.model.mixed_precision = 'no'
        config.train.train_batch_size = 1
        config.train.num_workers = 0
        config.train.gradient_accumulation_steps = 1
        config.train.auto_lr = {}
        config.train.lr_schedule_args = {'num_warmup_steps': 0}
        config.report_to = 'none'
        config.data.extra['ppp_subset_indices'] = [0, 1]
        data = TinyDataset()
        data.manifest_identity = dict(format_version=1, validation_version=1, sha256='a'*64)
        options = SimpleNamespace(smoke_check=False, max_updates=1, resume=None,
                                  null_embed_path='substitute-test-only', text_model_dir='substitute-test-only')
        with tempfile.TemporaryDirectory() as directory:
            config.work_dir = directory
            first = run_training(config, options, components=components(), dataset=data, text_provider=text)
            state = torch.load(first['checkpoint'], map_location='cpu', weights_only=False)
            contract = state['ppp_training']['contract']
            self.assertEqual(contract['manifest_identity'], data.manifest_identity)
            self.assertNotIn('ppp_manifest_path', contract['data'])
            options.resume = first['checkpoint']
            config.data.ppp_manifest_path = 'relocated-identical-manifest.json'
            second = run_training(config, options, components=components(), dataset=data, text_provider=text)
            self.assertEqual(second['progress']['global_step'], 2)
            data.manifest_identity = dict(data.manifest_identity, sha256='b'*64)
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                run_training(config, options, components=components(), dataset=data, text_provider=text)

    def test_integrated_updates_accumulation_and_resume(self):
        with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
            config=pyrallis.load(SanaConfig,stream)
        config.model.image_size=config.data.image_size=2
        config.vae.vae_downsample_rate=1
        config.model.mixed_precision='no'
        config.train.train_batch_size=1
        config.train.num_workers=0
        config.train.num_epochs=2
        config.train.gradient_accumulation_steps=2
        config.train.auto_lr={}
        config.train.lr_schedule_args={'num_warmup_steps':1}
        config.train.log_interval=1
        config.report_to='none'
        config.train.optimizer['lr']=0.01
        options=SimpleNamespace(smoke_check=True,max_updates=3,resume=None,
                                null_embed_path='substitute-test-only',text_model_dir='substitute-test-only')
        torch.manual_seed(123)
        reference_components=components()
        initial=copy.deepcopy(reference_components)
        with tempfile.TemporaryDirectory() as directory:
            config.work_dir=str(Path(directory)/'continuous')
            continuous=copy.deepcopy(reference_components)
            full_options=copy.copy(options);full_options.max_updates=6
            full=run_training(config,full_options,components=continuous,dataset=TinyDataset(),text_provider=text)
            config.work_dir=str(Path(directory)/'resume')
            split=copy.deepcopy(initial)
            before_encoder=copy.deepcopy(split.condition_encoder.state_dict())
            first=run_training(config,options,components=split,dataset=TinyDataset(),text_provider=text)
            self.assertEqual(first['progress'],dict(epoch=0,batch_in_epoch=6,global_step=3))
            self.assertEqual(split.model.dit.calls,6)
            for key in ('dit.proj.weight','dit.modality_pe','ppp_decoder.decoder.early.weight',
                        'ppp_decoder.decoder.project_out.op_list.0.conv.weight'):
                self.assertFalse(torch.equal(initial.model.state_dict()[key],split.model.state_dict()[key]))
            state=torch.load(first['checkpoint'],map_location='cpu',weights_only=False)
            self.assertTrue(any(k.startswith('dit.') for k in state['state_dict']))
            self.assertTrue(any(k.startswith('ppp_decoder.') for k in state['state_dict']))
            self.assertFalse(any('condition_encoder' in k for k in state['state_dict']))
            self.assertTrue(state['optimizer']['state'])
            resumed=copy.deepcopy(initial)
            options.resume=first['checkpoint']
            second=run_training(config,options,components=resumed,dataset=TinyDataset(),text_provider=text)
            self.assertEqual(second['progress'],full['progress'])
            for key,tensor in continuous.model.state_dict().items():
                torch.testing.assert_close(tensor,resumed.model.state_dict()[key],rtol=0,atol=0)
            final=torch.load(second['checkpoint'],map_location='cpu',weights_only=False)
            full_state=torch.load(full['checkpoint'],map_location='cpu',weights_only=False)
            self.assertEqual(final['scheduler'],full_state['scheduler'])
            for key,value in full_state['optimizer']['state'].items():
                for name,tensor in value.items():
                    if isinstance(tensor,torch.Tensor):
                        torch.testing.assert_close(tensor,final['optimizer']['state'][key][name],rtol=0,atol=0)
                    else:
                        self.assertEqual(tensor,final['optimizer']['state'][key][name])
            for key,tensor in before_encoder.items():
                torch.testing.assert_close(tensor,resumed.condition_encoder.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in resumed.condition_encoder.parameters()))
            generator=torch.Generator()
            rng=final['rank_rng_states'][0]
            restore_rng_state(rng,generator)
            numbers=(torch.rand(1),random.random(),np.random.rand())
            restore_rng_state(rng,generator)
            torch.testing.assert_close(numbers[0],torch.rand(1))
            self.assertEqual(numbers[1],random.random());self.assertEqual(numbers[2],np.random.rand())
            # Resume mismatch fails without injecting an unprefixed text parameter.
            broken=dict(final);broken['state_dict']=dict(final['state_dict'])
            broken['state_dict']['y_embedder.y_embedding']=torch.zeros(1)
            bad=Path(directory)/'bad.pth';torch.save(broken,bad)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_ppp_checkpoint(bad,resumed.model)
            with self.assertRaisesRegex(ValueError,'contract'):
                load_ppp_checkpoint(second['checkpoint'],resumed.model,contract={'wrong':True})

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
