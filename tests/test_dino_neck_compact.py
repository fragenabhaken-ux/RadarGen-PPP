"""Compact internal attention/MLP widths and legacy neck compatibility."""
from pathlib import Path
import unittest

import pyrallis
import torch

from diffusion.utils.config import SanaConfig
from radargen.models.dino_ppp import RoPETransformerNeck
from scripts.train_ppp import validate_ppp_config
import test_dino_ppp_model as baseline

ROOT=Path(__file__).resolve().parents[1]


def compact_neck(checkpointing=False):
    return RoPETransformerNeck(36,2,2,4.,0.,checkpointing,head_dim=8,mlp_hidden_dim=4)


class CompactNeckTests(unittest.TestCase):
    def test_exact_projections_parameter_budget_and_legacy_keys(self):
        with torch.device('meta'):
            compact=RoPETransformerNeck(1152,12,4,4.,0.,False,head_dim=128,mlp_hidden_dim=128)
            legacy=RoPETransformerNeck(1152,12,9,4.,0.,False)
        self.assertEqual(sum(p.numel() for p in compact.parameters()),31869696)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()),191122176)
        self.assertEqual(len(compact.layers),12)
        for block in compact.layers:
            self.assertEqual((block.linear1.in_features,block.linear1.out_features),(1152,1664))
            self.assertEqual((block.linear2.in_features,block.linear2.out_features),(640,1152))
            self.assertEqual((block.num_heads,block.head_dim,block.attention_dim,block.mlp_hidden),(4,128,512,128))
        self.assertEqual(compact.state_dict().keys(),legacy.state_dict().keys())
        changed=[k for k,v in compact.state_dict().items() if v.shape!=legacy.state_dict()[k].shape]
        self.assertEqual(len(changed),24)  # Only two learned projections per block change shape.
        self.assertTrue(all(k.endswith(('linear1.weight','linear2.weight')) for k in changed))

    def test_checkpointed_compact_neck_keeps_gradients_and_strict_restore(self):
        plain=compact_neck(False)
        checked=compact_neck(True)
        checked.load_state_dict(plain.state_dict(),strict=True)
        x=torch.randn(2,36,2,3,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        a,b=plain(x),checked(y)
        self.assertEqual(a.shape,x.shape)
        torch.testing.assert_close(a,b)
        a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad)
        for p,q in zip(plain.parameters(),checked.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            torch.testing.assert_close(p.grad,q.grad)

    def test_explicit_legacy_widths_restore_existing_defaults_exactly(self):
        old=RoPETransformerNeck(24,2,3,4.,0.,False)
        explicit=RoPETransformerNeck(24,2,3,4.,0.,False,head_dim=8,mlp_hidden_dim=96)
        explicit.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(2,24,2,3)
        torch.testing.assert_close(old(x),explicit(x),rtol=0,atol=0)
        self.assertEqual(old.layers[0].linear1.weight.shape,(168,24))
        self.assertEqual(old.layers[0].linear2.weight.shape,(24,120))

    def test_dino_full_output_and_backbone_gradient_with_compact_neck(self):
        model=baseline.DinoPPPTests().build(decoder='attention',frozen_base=False,
            feature_fusion='last_concat',out_indices=(1,),embed_dim=36,neck_layers=2,
            neck_heads=2,neck_head_dim=8,neck_mlp_hidden_dim=4,output_scale=2,
            attention_channels=(8,4,4),attention_blocks=2,attention_heads=2,
            attention_window_size=4,attention_projection_kernel_size=3,
            attention_shifted_windows=True,grad_checkpointing=True)
        maps=[torch.randn(1,3,17,19) for _ in range(3)]
        result=model(*maps)
        self.assertEqual(result.shape,(1,3,34,38))
        result.square().mean().backward()
        self.assertGreater(model.backbone.patch_embed.proj.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad))
        self.assertEqual(model.neck.layers[0].attention_dim,16)
        self.assertEqual(model.neck.layers[0].mlp_hidden,4)

    def test_compact_configs_are_pretrained_and_invalid_widths_are_rejected(self):
        for filename in ['ablation_dino_neck_compact.yaml','ablation_dino_neck_compact_lr_range.yaml']:
            with (ROOT/'configs'/filename).open() as handle:cfg=pyrallis.load(SanaConfig,handle)
            validate_ppp_config(cfg)
            self.assertEqual(cfg.model.extra['backbone_initialization'],'pretrained')
            self.assertEqual(cfg.model.extra['neck_mlp_hidden_dim'],128)
            self.assertFalse(cfg.model.extra['frozen_base'])
            self.assertIsNone(cfg.resume_from)
        for key,value in [('neck_head_dim',0),('neck_head_dim',7),('neck_head_dim',True),
                          ('neck_mlp_hidden_dim',0),('neck_mlp_hidden_dim',1.5)]:
            with (ROOT/'configs/ablation_dino_neck_compact.yaml').open() as handle:
                cfg=pyrallis.load(SanaConfig,handle)
            cfg.model.extra[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_ppp_config(cfg)
        with self.assertRaises(ValueError):
            RoPETransformerNeck(36,1,2,4.,0.,False,head_dim=7,mlp_hidden_dim=4)
        with self.assertRaises(ValueError):
            RoPETransformerNeck(36,1,2,4.,0.,False,head_dim=8,mlp_hidden_dim=0)


if __name__=='__main__':unittest.main()
