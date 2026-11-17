"""ResNet backbone substitution: geometry, pretrained loading, gradients and restore."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyrallis
import torch
from torch import nn

from diffusion.utils.config import SanaConfig
from radargen.models.resnet_ppp import ResNetPPP
from radargen.models.dino_ppp import AttentionPixelShuffleHead, RoPETransformerNeck
from radargen.training.ppp_initialization import initialize_dino_ppp_model
from radargen.losses.continuous import ContinuousPPPLoss
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline
from scripts.train_ppp import validate_ppp_config

ROOT=Path(__file__).resolve().parents[1]


def config():
    with (ROOT/'configs/ablation_resnet50_backbone.yaml').open() as handle:
        return pyrallis.load(SanaConfig,handle)


def tiny_options():
    return dict(projection_channels=12,embed_dim=36,neck_layers=1,neck_heads=3,
                attention_channels=(8,4,4),attention_heads=2,attention_window_size=4,
                attention_blocks=2,output_scale=2)


class ResNetPPPTests(unittest.TestCase):
    def test_default_neck_and_head_match_reference_architecture(self):
        with torch.device('meta'):
            model=ResNetPPP()
        self.assertIsInstance(model.neck,RoPETransformerNeck)
        self.assertIsInstance(model.head,AttentionPixelShuffleHead)
        self.assertEqual(len(model.neck.layers),12)
        self.assertEqual(model.neck.out_norm.normalized_shape,(1152,))
        self.assertEqual(model.projection.weight.shape,(384,2048,1,1))
        self.assertEqual(model.backbone.conv1.kernel_size,(7,7))
        self.assertEqual(model.backbone.layer4[0].conv2.stride,(1,1))
        self.assertEqual(model.backbone.layer4[1].conv2.dilation,(2,2))
        self.assertIsInstance(model.backbone.fc,nn.Identity)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),239031907)
        self.assertEqual(sum(p.numel() for p in model.neck.parameters()),191122176)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()),23615267)

    def test_stride16_rectangular_prediction_gradients_and_single_bn_update(self):
        model=ResNetPPP(**tiny_options(),grad_checkpointing=True).train()
        maps=[torch.randn(1,3,17,19) for _ in range(3)]
        seen=[]
        hook=model.neck.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
        before=model.backbone.bn1.num_batches_tracked.clone()
        out=model(*maps)
        hook.remove()
        self.assertEqual(seen,[torch.Size([1,36,2,2])])
        self.assertEqual(out.shape,(1,3,34,38))
        losses=ContinuousPPPLoss((34,38))(out,torch.tensor([[0.,0.,2.,-1.]]),
                                        torch.zeros(1,dtype=torch.long),point_limit=50.)
        sum(losses.values()).backward()
        for p in [model.backbone.conv1.weight,model.projection.weight,model.head.readout.weight]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum(),0)
        self.assertEqual(model.backbone.bn1.num_batches_tracked-before,1)

    def test_shared_projection_preserves_modality_channel_order(self):
        model=ResNetPPP(**tiny_options()).eval()
        model.projection.weight.data.fill_(1/2048)
        seen=[]
        model.neck=nn.Identity()
        model.head=nn.Conv2d(36,3,1)
        hook=model.neck.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
        maps=[torch.full((2,3,16,16),float(i)) for i in [1,2,3]]
        with patch.object(model,'forward_features',side_effect=lambda x:x[:,:1,:1,:1].expand(-1,2048,1,1)):
            model(*maps)
        hook.remove()
        for i,chunk in enumerate(seen[0].chunk(3,dim=1),1):
            torch.testing.assert_close(chunk,torch.full_like(chunk,float(i)))

    def test_local_pretrained_weights_are_strictly_loaded_before_classifier_removal(self):
        cfg=config();validate_ppp_config(cfg)
        opts=tiny_options()
        model=ResNetPPP(checkpoint_path=str(ROOT/cfg.model.extra['checkpoint_path']),**opts)
        weights=torch.load(ROOT/cfg.model.extra['checkpoint_path'],map_location='cpu',weights_only=True)
        torch.testing.assert_close(model.backbone.conv1.weight,weights['conv1.weight'])
        torch.testing.assert_close(model.backbone.layer4[2].conv3.weight,weights['layer4.2.conv3.weight'])
        with self.assertRaises(FileNotFoundError):
            ResNetPPP(checkpoint_path='/tmp/nonexistent-resnet-weights.pth',**opts)
        with TemporaryDirectory() as directory:
            bad=Path(directory)/'bad.pth';torch.save({'conv1.weight':weights['conv1.weight']},bad)
            with self.assertRaises(RuntimeError):
                ResNetPPP(checkpoint_path=str(bad),**opts)

    def test_ppp_inference_restores_without_external_imagenet_asset(self):
        cfg=config();cfg.model.extra.update(tiny_options());cfg.train.grad_checkpointing=False
        cfg.model.extra['checkpoint_path']='/tmp/missing-original-resnet-imagenet.pth'
        source=ResNetPPP(**{k:v for k,v in cfg.model.extra.items() if k!='checkpoint_path'}).eval()
        maps=[torch.randn(3,16,16) for _ in range(3)]
        with torch.no_grad(): expected=source(*[x[None] for x in maps])
        with TemporaryDirectory() as directory:
            path=Path(directory)/'ppp.pth'
            torch.save(dict(state_dict=source.state_dict(),epoch=0,rank_rng_states=[{}],
                            ppp_training={'version':1,'progress':{'global_step':0}}),path)
            pipeline=RadarGenPPPPipeline.from_config(cfg,checkpoint_path=str(path),device='cpu')
            with torch.no_grad(): actual=pipeline.model(*[x[None] for x in maps])
            torch.testing.assert_close(actual,expected)
        self.assertIsNone(pipeline.condition_encoder)
        self.assertIsNone(pipeline.text_provider)
        with self.assertRaises(FileNotFoundError):
            initialize_dino_ppp_model(cfg,device='cpu',load_pretrained=True)

    def test_input_and_configuration_mismatches_are_rejected(self):
        cfg=config();cfg.model.extra['embed_dim']=384
        with self.assertRaises(ValueError): validate_ppp_config(cfg)
        with self.assertRaises(ValueError): ResNetPPP(model_name='resnet50_v1c')
        model=ResNetPPP(**tiny_options())
        with self.assertRaises(ValueError):
            model(torch.zeros(1,3,16,16),torch.zeros(1,3,32,16),torch.zeros(1,3,16,16))


if __name__=='__main__':
    unittest.main()
