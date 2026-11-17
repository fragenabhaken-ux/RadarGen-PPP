"""Original DC-AE stages, full-width DiT access, and unchanged legacy decoding."""
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pyrallis
import torch
from torch import nn
from diffusion.utils.config import SanaConfig
from diffusion.model.nets.radargen_ppp import RadarGenPPP
from diffusion.model_modification import inflate_sana_input_channels_for_ppp
from diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import Decoder, DecoderConfig
from diffusion.model.dc_ae.efficientvit.models.nn.ops import ConvLayer, ResidualBlock
from radargen.training.ppp_decoder import (
    PPPDecoder, FullWidthDCAEDecoder, adapt_ppp_decoder_output, adapt_ppp_decoder_input,
)
from radargen.training.radargen_ppp_training_model import RadarGenPPPTrainingModel
from radargen.training.ppp_initialization import initialize_ppp_model
from scripts.train_ppp import validate_ppp_config

LOG=logging.getLogger(__name__)
ROOT=Path(__file__).resolve().parents[1]


def tiny_decoder():
    cfg=DecoderConfig(in_channels=3,latent_channels=32,width_list=(16,32,64),
                      depth_list=(1,1,1),block_type='ResBlock',norm='rms2d',act='silu',
                      upsample_block_type='InterpolateConv')
    decoder=Decoder(cfg)
    # Native DC-AE RMS affine tensors use empty storage until pretrained weights
    # load. The tiny CPU fixture has no external checkpoint, so initialize them.
    from diffusion.model.dc_ae.efficientvit.models.nn.norm import RMSNorm2d
    for module in decoder.modules():
        if isinstance(module,RMSNorm2d):
            if module.weight is not None:nn.init.ones_(module.weight)
            if module.bias is not None:nn.init.zeros_(module.bias)
    adapt_ppp_decoder_output(decoder,LOG)
    return decoder


def tiny_dit():
    dit=RadarGenPPP(input_size=2,hidden_size=64,depth=1,num_heads=4,
                   caption_channels=8,model_max_length=4,attn_type='linear',
                   ffn_type='mlp',use_pe=False,class_dropout_prob=0.)
    inflate_sana_input_channels_for_ppp(dit,LOG)
    dit.final_layer=nn.Identity();dit.return_features=True
    return dit


class FullWidthDCAEChecks(unittest.TestCase):
    def test_only_input_block_changes_and_other_weights_survive_exactly(self):
        decoder=tiny_decoder()
        self.assertIsInstance(decoder.project_in,ResidualBlock)
        saved={k:v.clone() for k,v in decoder.state_dict().items() if not k.startswith('project_in.')}
        wrapped=FullWidthDCAEDecoder(decoder,64,LOG)
        self.assertIsInstance(decoder.project_in,ConvLayer)
        self.assertEqual(decoder.project_in.conv.weight.shape,(64,64,3,3))
        self.assertEqual(decoder.cfg.latent_channels,64)
        self.assertIsNone(decoder.cfg.in_shortcut)
        for k,v in saved.items():torch.testing.assert_close(decoder.state_dict()[k],v,rtol=0,atol=0)
        seen=[]
        hook=decoder.project_in.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
        features=torch.randn(3,64,2,2,requires_grad=True)
        out=wrapped(features)
        hook.remove()
        torch.testing.assert_close(seen[0],features.detach(),rtol=0,atol=0)
        self.assertEqual(out.shape,(3,1,8,8))
        out.square().mean().backward()
        self.assertTrue((features.grad.abs().sum(dim=(0,2,3))>0).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapped.parameters()))
        with self.assertRaises(ValueError):wrapped(torch.zeros(3,32,2,2))
        with self.assertRaises(ValueError):adapt_ppp_decoder_input(decoder,64,LOG)

    def test_full_model_shape_gradient_checkpointing_and_strict_roundtrip(self):
        dit=tiny_dit();decoder=tiny_decoder();decoder.gradient_checkpointing=True
        model=RadarGenPPPTrainingModel(dit,FullWidthDCAEDecoder(decoder,64,LOG))
        maps=[torch.randn(1,32,2,2) for _ in range(3)]
        y=torch.randn(3,1,4,8);mask=torch.ones(3,1,1,4)
        out=model(maps,y,mask=mask)
        self.assertEqual(out.shape,(1,3,1,8,8))
        out.square().mean().backward()
        self.assertGreater(dit.x_embedder.proj.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        clone=RadarGenPPPTrainingModel(tiny_dit(),FullWidthDCAEDecoder(tiny_decoder(),64,LOG))
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(maps,y,mask=mask),out)

    def test_real_decoder_budget_and_512_output_without_allocating_weights(self):
        from diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
        with torch.device('meta'):
            ae=DCAE_HF('dc-ae-f32c32-sana-1.1')
            adapt_ppp_decoder_output(ae.decoder,LOG)
            before=sum(p.numel() for p in ae.decoder.parameters())
            decoder=FullWidthDCAEDecoder(ae.decoder,1152,LOG)
            self.assertEqual(before,159220865)
            self.assertEqual(decoder.parameter_count,169542785)
            self.assertEqual(decoder(torch.empty(3,1152,16,16)).shape,(3,1,512,512))

    def test_legacy_decoder_keeps_scaling_and_32_channel_shortcut(self):
        decoder=tiny_decoder();legacy=PPPDecoder(decoder,0.5)
        seen=[]
        hook=decoder.project_in.register_forward_pre_hook(lambda module,args:seen.append(args[0]))
        z=torch.randn(3,32,2,2)
        legacy(z);hook.remove()
        torch.testing.assert_close(seen[0],2*z)
        self.assertIsInstance(decoder.project_in,ResidualBlock)

    def test_initializer_retains_frozen_encoder_but_removes_final_prediction_layer(self):
        with (ROOT/'configs/ablation_dit_dcae_no_bottleneck.yaml').open() as handle:
            config=pyrallis.load(SanaConfig,handle)
        validate_ppp_config(config)
        config.text_encoder.model_max_length=4
        config.model.qk_norm=False
        calls=[]
        def builder(*args,**kwargs):
            dit=tiny_dit()
            # Constructor expects the bare 32-channel input before PPP inflation.
            dit.x_embedder.proj=nn.Conv2d(32,64,1)
            return dit
        def weights(*args,**kwargs):
            calls.append('sana');return 0,['modality_pe','pos_embed'],[],None
        class AE(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.Conv2d(3,32,1)
                self.decoder=Decoder(DecoderConfig(in_channels=3,latent_channels=32,
                    width_list=(16,32,64),depth_list=(1,1,1),block_type='ResBlock',
                    norm='rms2d',act='silu',upsample_block_type='InterpolateConv'))
                self.cfg=SimpleNamespace(scaling_factor=0.41407)
        with TemporaryDirectory() as directory:
            config.work_dir=directory
            path=Path(directory)/'null.pth'
            torch.save({'uncond_prompt_embeds':torch.ones(1,4,8),
                        'uncond_prompt_embeds_mask':torch.ones(1,4,dtype=torch.long)},path)
            result=initialize_ppp_model(config,null_embed_path=path,caption_channels=8,
                                       model_builder=builder,checkpoint_loader=weights,
                                       vae_loader=lambda *args:AE())
        self.assertEqual(calls,['sana'])
        self.assertIsInstance(result.model.ppp_decoder,FullWidthDCAEDecoder)
        self.assertIsInstance(result.model.dit.final_layer,nn.Identity)
        self.assertTrue(result.model.dit.return_features)
        self.assertTrue(all(p.requires_grad for p in result.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in result.condition_encoder.parameters()))
        self.assertEqual(result.condition_encoder.scaling_factor,0.41407)


if __name__=='__main__':unittest.main()
