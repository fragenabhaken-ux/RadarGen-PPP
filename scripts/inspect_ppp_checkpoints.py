#!/usr/bin/env python3
"""CPU checkpoint integrity/key inspection; no pretrained forward or null replacement.

SANA uses mmap and meta construction with assign=True to inspect real weights
without allocating a second 600M model. DC-AE uses the baseline local loader.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch, pyrallis
from safetensors import safe_open
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.model.nets.radargen_ppp import RadarGenPPP_600M_P1_D28
from diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
from radargen.training.ppp_initialization import validate_local_ppp_assets
with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
    config=pyrallis.load(SanaConfig,stream)
sana=config.model.load_from; ae=config.vae.vae_pretrained
print('INTEGRITY',validate_local_ppp_assets(sana,ae),flush=True)
state=torch.load(sana,map_location='cpu',mmap=True,weights_only=False)['state_dict']
kwargs=model_init_config(config,16)
kwargs['config']=None
from unittest.mock import patch
linspace = torch.linspace
with patch('torch.linspace', lambda *a, **k: linspace(*a, **dict(k,device='cpu'))), torch.device('meta'):
    dit=RadarGenPPP_600M_P1_D28(**kwargs)
expected=dit.state_dict()
state.pop('pos_embed',None)
print('SANA missing',sorted(set(expected)-set(state)), 'unexpected',sorted(set(state)-set(expected)),flush=True)
print('SANA shape mismatches',[(k,tuple(v.shape),tuple(expected[k].shape)) for k,v in state.items() if k in expected and v.shape!=expected[k].shape],flush=True)
result=dit.load_state_dict(state,strict=False,assign=True)
print('SANA assigned real weights BEFORE inflation',result,tuple(dit.x_embedder.proj.weight.shape),flush=True)
from diffusion.model_modification import inflate_sana_input_channels_for_ppp
import logging
original = dit.x_embedder.proj.weight.detach().clone()
inflate_sana_input_channels_for_ppp(dit, logging.getLogger(__name__))
torch.testing.assert_close(dit.x_embedder.proj.weight[:, :96], original.repeat(1,3,1,1)/3)
print('SANA ADAPTED',tuple(dit.x_embedder.proj.weight.shape),flush=True)
with torch.device('meta'):
    dcae=DCAE_HF('dc-ae-f32c32-sana-1.1')
with safe_open(ae+'/model.safetensors',framework='pt',device='cpu') as f:
    keys=set(f.keys()); expected=dcae.state_dict()
    print('DC-AE missing',sorted(set(expected)-keys),'unexpected',sorted(keys-set(expected)),flush=True)
    print('DC-AE shape mismatches',[(k,tuple(f.get_slice(k).get_shape()),tuple(expected[k].shape)) for k in keys & set(expected) if tuple(f.get_slice(k).get_shape())!=tuple(expected[k].shape)],flush=True)
from diffusion.model.builder import get_vae
from radargen.training.ppp_decoder import adapt_ppp_decoder_output
import logging
del dit,state
loaded=get_vae('dc-ae',ae,'cpu')
print('DC-AE get_vae LOCAL LOADED',tuple(loaded.decoder.project_out.op_list[-1].conv.weight.shape),loaded.cfg.scaling_factor,flush=True)
original_rgb=loaded.decoder.project_out.op_list[-1].conv.weight.detach().clone()
adapt_ppp_decoder_output(loaded.decoder,logging.getLogger(__name__))
torch.testing.assert_close(loaded.decoder.project_out.op_list[-1].conv.weight,original_rgb.mean(0,keepdim=True))
print('DC-AE ADAPTED',tuple(loaded.decoder.project_out.op_list[-1].conv.weight.shape),flush=True)
