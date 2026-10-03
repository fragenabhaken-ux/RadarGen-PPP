"""PPP wrapper checkpoints: strict resume without SANA null-embedding surgery."""
import logging
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng_state(generator):
    return dict(torch=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                numpy=np.random.get_state(), python=random.getstate(), generator=generator.get_state())


def restore_rng_state(state, generator):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    generator.set_state(state['generator'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_ppp_checkpoint(directory, training_model, optimizer, lr_scheduler, *,
                        progress, contract, generator, rank_rng_states=None):
    """Reuse generic saving with explicit progress/RNG and atomic publication.

    Call only after synchronized optimizer steps, with an unwrapped model.
    """
    from diffusion.utils.checkpoint import save_checkpoint
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = capture_rng_state(generator)
    extra = dict(ppp_training=dict(version=1, progress=progress, contract=contract),
                 rank_rng_states=rank_rng_states or [rng])
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        path = save_checkpoint(temporary, progress['epoch'], training_model,
                               optimizer=optimizer, lr_scheduler=lr_scheduler,
                               generator=generator, step=progress['global_step'],
                               extra_state=extra, rng_state=rng)
        target = directory / Path(path).name
        os.replace(path, target)
    latest = directory / 'latest.pth'
    pending = directory / '.latest.tmp'
    pending.unlink(missing_ok=True)
    pending.symlink_to(target.name)
    os.replace(pending, latest)
    return str(target)


def load_ppp_checkpoint(checkpoint_path, training_model, optimizer=None, lr_scheduler=None,
                        resume_optimizer=True, resume_lr_scheduler=True, *, contract=None, rank=0):
    """Restore exact wrapper keys and requested training state; return baseline tuple.

    Returned RNG dictionary also includes ppp_training progress/contract metadata.
    No positional keys are deleted and no null embedding is injected.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f'Local PPP checkpoint unavailable: {path}')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    metadata = state.get('ppp_training', {})
    if metadata.get('version') != 1:
        raise ValueError('Expected a version-1 PPP training checkpoint')
    if contract is not None and metadata['contract'] != contract:
        raise ValueError('PPP resume training/data contract differs from saved checkpoint')
    saved, expected = state['state_dict'], training_model.state_dict()
    missing, unexpected = sorted(set(expected)-set(saved)), sorted(set(saved)-set(expected))
    logging.getLogger(__name__).info('PPP missing keys: %s; unexpected keys: %s', missing, unexpected)
    if missing or unexpected:
        raise ValueError(f'PPP checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    if optimizer is not None and resume_optimizer and 'optimizer' not in state:
        raise ValueError('PPP checkpoint missing optimizer state')
    if lr_scheduler is not None and resume_lr_scheduler and 'scheduler' not in state:
        raise ValueError('PPP checkpoint missing scheduler state')
    ranks = state['rank_rng_states']
    if rank >= len(ranks):
        raise ValueError('PPP checkpoint has no RNG state for this rank')
    training_model.load_state_dict(saved, strict=True)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(state['scheduler'])
    rng = dict(ranks[rank], ppp_training=metadata)
    return state['epoch'], missing, unexpected, rng
