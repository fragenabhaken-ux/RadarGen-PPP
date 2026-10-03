#!/usr/bin/env python3
"""Explicit pretrained WP4 GPU probe; no optimizer, job submission, or training."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyrallis
import torch
from diffusion.utils.config import SanaConfig
from radargen.training.ppp_initialization import initialize_ppp_model
from radargen.losses.mark_losses import masked_rcs_l1, masked_doppler_loss
from radargen.losses.ppp_loss import PPPLoss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/RadarGen_600M_512px_TS_PPP_training.yaml')
    parser.add_argument('--null-embed-path',required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('Run this explicit pretrained runtime probe on an allocated GPU')
    logging.basicConfig(level=logging.INFO)
    with open(args.config) as stream:
        config=pyrallis.load(SanaConfig,stream)
    components=initialize_ppp_model(config,null_embed_path=args.null_embed_path,device='cuda')
    print('SANA missing/unexpected:',components.missing_keys,components.unexpected_keys)
    model=components.model.train()
    encoder=components.condition_encoder
    torch.manual_seed(7)
    images=[torch.randn(1,3,512,512,device='cuda') for _ in range(3)]
    null=torch.load(args.null_embed_path,map_location='cuda',weights_only=True)
    y=null['uncond_prompt_embeds'][:,None].repeat(3,1,1,1)
    mask=null['uncond_prompt_embeds_mask'][:,None,None].repeat(3,1,1,1)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        conditions=encoder(images)
        output=model(conditions,y,mask)
    assert output.shape==(1,3,1,512,512)
    for index,name in ((1,'RCS'),(2,'Doppler')):
        raw=output[:,index]
        assert torch.isfinite(raw).all(), f'{name} raw output is nonfinite'
        assert raw.count_nonzero()>0, f'{name} raw output is identically zero'
    del raw
    print('condition shapes:',[tuple(c.shape) for c in conditions])
    print('raw output min/max/mean/std:',*[v.item() for v in (output.min(),output.max(),output.mean(),output.std())])
    point_mask=torch.ones_like(output[:,0]); target=torch.ones_like(point_mask)*10
    del output
    decoder=model.ppp_decoder.decoder
    early=next(p for p in decoder.parameters() if p.ndim>=2)
    checked_parameters=(early,decoder.project_out.op_list[-1].conv.weight,
                        model.dit.modality_pe,model.dit.blocks[0].attn.qkv.weight)
    for index,name in ((0,'PPP'),(1,'RCS'),(2,'Doppler')):
        model.zero_grad(set_to_none=True)
        # Recompute one graph per loss; never retain a previous backward graph.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(conditions,y,mask)
        if index==0:
            loss=PPPLoss()(output[:,0],point_mask)
        elif index==1:
            loss=masked_rcs_l1(output[:,1],target,point_mask)
        else:
            loss=masked_doppler_loss(output[:,2],target,point_mask,'signed_l1')
        loss.backward()
        for p in checked_parameters:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0, f'{name}-only gradient check failed'
        print(f'{name}-only decoder/DiT gradient checks passed')
        del loss,output
    model.zero_grad(set_to_none=True)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert not encoder.training and not encoder.encoder.training
    assert all(p.dtype==torch.float32 and p.requires_grad for p in decoder.parameters())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        latent=torch.randn(1,32,16,16,device='cuda')
        model.ppp_decoder.train(); train_output=model.ppp_decoder(latent)
        model.ppp_decoder.eval(); eval_output=model.ppp_decoder(latent)
        torch.testing.assert_close(train_output,eval_output)
    print('Pretrained WP4 forward/backward and decoder train/eval checks passed')

if __name__=='__main__':
    main()
