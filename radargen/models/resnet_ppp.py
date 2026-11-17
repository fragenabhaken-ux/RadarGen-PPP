"""Standard ResNet-50 BEV encoder with the unchanged DINO PPP neck and head."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .dino_ppp import AttentionPixelShuffleHead, RoPETransformerNeck


class ResNetPPP(nn.Module):
    """Shared RGB encoder, 2048→384 per-view projection, concatenated PPP readout.

    Last-stage stride is replaced by dilation, giving stride-16 features while
    preserving ImageNet weight shapes. Backbone BatchNorm runs normally and is
    not checkpoint-recomputed; only the existing neck/head use checkpointing.
    Inputs use the same BEV normalization as DinoPPP, without an extra resize.
    """

    def __init__(self, model_name='resnet50', checkpoint_path='', frozen_base=False,
                 projection_channels=384, embed_dim=1152, neck_layers=12,
                 neck_heads=9, mlp_ratio=4., attn_dropout=0., decoder='attention',
                 output_scale=2, attention_channels=(128,64,32), attention_blocks=2,
                 attention_heads=4, attention_window_size=8,
                 attention_projection_kernel_size=3, attention_shifted_windows=True,
                 grad_checkpointing=False):
        super().__init__()
        if model_name != 'resnet50':
            raise ValueError('ResNetPPP requires standard torchvision ResNet-50')
        if frozen_base:
            raise ValueError('This backbone ablation requires fully trainable ResNet-50')
        if embed_dim != 3*projection_channels or decoder != 'attention':
            raise ValueError('ResNetPPP requires concatenated modality features and the standard head')
        from torchvision.models import resnet50
        # No implicit downloads: external RGB weights must be supplied locally.
        self.backbone = resnet50(weights=None, replace_stride_with_dilation=[False,False,True])
        if checkpoint_path:
            path=Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f'Local ResNet-50 checkpoint unavailable: {path}')
            state=torch.load(path,map_location='cpu',weights_only=True)
            # Strict matching includes the original classification layer before removal.
            self.backbone.load_state_dict(state,strict=True)
        self.backbone.fc=nn.Identity()
        self.backbone.avgpool=nn.Identity()
        self.backbone.requires_grad_(True)
        self.projection=nn.Conv2d(2048,projection_channels,1,bias=False)
        self.neck=RoPETransformerNeck(embed_dim,neck_layers,neck_heads,mlp_ratio,
                                      attn_dropout,grad_checkpointing)
        self.head=AttentionPixelShuffleHead(embed_dim,attention_channels,attention_blocks,
                attention_heads,attention_window_size,output_scale,grad_checkpointing,
                attention_projection_kernel_size,attention_shifted_windows)
        self.output_scale=output_scale
        self.output_stride=16

    def forward_features(self,x):
        net=self.backbone
        x=net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x=net.layer1(x)
        x=net.layer2(x)
        x=net.layer3(x)
        return net.layer4(x)

    def forward(self,bev_color_map,bev_seg_map,bev_velocity_map):
        maps=(bev_color_map,bev_seg_map,bev_velocity_map)
        if any(x.ndim != 4 or x.shape != maps[0].shape or x.shape[1] != 3 for x in maps):
            raise ValueError('Three equally shaped BCHW RGB BEV maps are required')
        batch,_,height,width=maps[0].shape
        x=F.pad(torch.cat(maps,dim=0),(0,-width%16,0,-height%16))
        x=self.projection(self.forward_features(x))
        x=torch.cat(x.split(batch,dim=0),dim=1)
        x=self.head(self.neck(x))
        return x[...,:height*self.output_scale,:width*self.output_scale]
