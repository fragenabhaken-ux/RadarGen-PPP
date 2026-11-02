"""Epoch permutations, rank shards and mid-epoch training resume are reproducible."""
from unittest.mock import patch
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import pyrallis
from diffusion.data.wids.wids import DistributedRangedSampler
from radargen.training.ppp_sampler import ResumableDistributedShuffleSampler
from diffusion.utils.config import SanaConfig
from test_ppp_training import components, TinyDataset, text
from scripts.train_ppp import run_training


def test_seed_plus_epoch_and_resume_offset_preserve_global_rng():
    sampler = ResumableDistributedShuffleSampler(range(20), num_replicas=1, rank=0, seed=42)
    sampler.set_epoch(7)
    rng = torch.get_rng_state().clone()
    order = list(sampler)
    expected = torch.randperm(20, generator=torch.Generator().manual_seed(49)).tolist()
    assert order == expected
    assert torch.equal(torch.get_rng_state(), rng)
    sampler.set_epoch(7)
    sampler.set_start(6)
    assert list(sampler) == order[6:]
    sampler.set_epoch(8)
    sampler.set_start(0)
    assert list(sampler) != order
    sequential = DistributedRangedSampler(range(20))
    assert list(sequential) == list(range(20))


def test_distributed_shards_use_same_global_permutation_without_overlap():
    with patch('diffusion.data.wids.wids.dist.is_initialized', return_value=True), \
         patch('diffusion.data.wids.wids.dist.get_rank', return_value=0):
        samplers = [ResumableDistributedShuffleSampler(range(20), num_replicas=4, rank=r,
                    seed=42) for r in range(4)]
    for s in samplers:
        s.set_epoch(3)
    parts = [list(s) for s in samplers]
    expected = torch.randperm(20, generator=torch.Generator().manual_seed(45)).tolist()
    assert [parts[r][i] for i in range(5) for r in range(4)] == expected
    assert len(set(sum(parts, []))) == 20


def test_shuffled_training_resume_matches_across_epoch_boundary():
    torch.set_num_threads(2)
    with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as f:
        config = pyrallis.load(SanaConfig, f)
    config.model.image_size = config.data.image_size = 2
    config.vae.vae_downsample_rate = 1
    config.model.mixed_precision = 'no'
    config.train.train_batch_size = 1
    config.train.num_workers = 0
    config.train.num_epochs = 2
    config.train.gradient_accumulation_steps = 2
    config.train.auto_lr = {}
    config.train.lr_schedule_args = {'num_warmup_steps':1}
    config.train.extra['shuffle_each_epoch'] = True
    config.report_to = 'none'
    torch.manual_seed(123)
    initial = components()
    options = SimpleNamespace(smoke_check=False,max_updates=10,resume=None,
                              null_embed_path='substitute-test-only',text_model_dir='substitute-test-only')
    with tempfile.TemporaryDirectory() as directory:
        config.work_dir = str(Path(directory)/'full')
        continuous = copy.deepcopy(initial)
        full = run_training(config,options,components=continuous,dataset=TinyDataset(),text_provider=text)
        config.work_dir = str(Path(directory)/'split')
        options.max_updates = 3
        first = run_training(config,options,components=copy.deepcopy(initial),dataset=TinyDataset(),text_provider=text)
        options.max_updates = 7
        options.resume = first['checkpoint']
        resumed = copy.deepcopy(initial)
        second = run_training(config,options,components=resumed,dataset=TinyDataset(),text_provider=text)
        assert full['progress'] == second['progress']
        assert full['progress']['epoch'] == 1
        for key, parameter in continuous.model.state_dict().items():
            torch.testing.assert_close(parameter,resumed.model.state_dict()[key],rtol=0,atol=0)
