"""WP4 CPU checks: tiny real DiT architecture and substitute DC-AE modules."""
import logging
import unittest
import tempfile
from pathlib import Path
import pyrallis
from types import SimpleNamespace
import torch
from torch import nn
from diffusion.model.nets.radargen_ppp import RadarGenPPP
from diffusion.model_modification import inflate_sana_input_channels_for_ppp
from radargen.training.ppp_decoder import PPPDecoder, FrozenConditionEncoder, adapt_ppp_decoder_output
from radargen.training.radargen_ppp_training_model import RadarGenPPPTrainingModel, PPP_MODALITIES
from radargen.losses.mark_losses import masked_rcs_l1, masked_doppler_loss

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Conv2d(32, 8, 1)
        self.project_out = nn.Module()
        last = nn.Module()
        last.conv = nn.Conv2d(8, 3, 1)
        self.project_out.op_list = nn.ModuleList([last])
    def forward(self, x):
        return self.project_out.op_list[-1].conv(self.early(x).tanh())

class PPPModelChecks(unittest.TestCase):
    def test_initialization_order_with_substitute_loaders(self):
        from diffusion.utils.config import SanaConfig
        from radargen.training.ppp_initialization import initialize_ppp_model
        with open('configs/RadarGen_600M_512px_TS_PPP_training.yaml') as stream:
            config = pyrallis.load(SanaConfig, stream)
        calls = []
        def builder(*args, **kwargs):
            calls.append('build')
            return RadarGenPPP(input_size=2, hidden_size=32, depth=1, num_heads=4,
                               caption_channels=8, model_max_length=300, attn_type='vanilla',
                               ffn_type='mlp', use_pe=False, class_dropout_prob=0.0)
        def checkpoint(source, dit, **kwargs):
            self.assertEqual(dit.x_embedder.proj.in_channels,32)
            calls.append('sana')
            with torch.no_grad():
                dit.x_embedder.proj.weight.fill_(0.3)
            return 0, ['modality_pe','pos_embed'], [], None
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Conv2d(3,32,1)
                self.decoder = Decoder()
                self.cfg = SimpleNamespace(scaling_factor=0.41407)
        def vae(*args):
            calls.append('ae')
            ae = AE()
            self.assertEqual(ae.decoder.project_out.op_list[-1].conv.out_channels,3)
            return ae
        with tempfile.TemporaryDirectory() as directory:
            config.work_dir = directory
            path = Path(directory)/'substitute-test-null.pth'
            # Synthetic text only for explicitly injected substitute-loader test.
            torch.save({'uncond_prompt_embeds':torch.ones(1,300,8),
                        'uncond_prompt_embeds_mask':torch.ones(1,300,dtype=torch.long)},path)
            result = initialize_ppp_model(config,null_embed_path=path,caption_channels=8,
                                          model_builder=builder,checkpoint_loader=checkpoint,vae_loader=vae)
            self.assertEqual(calls,['build','sana','ae'])
            torch.testing.assert_close(result.model.dit.x_embedder.proj.weight[:,:96],
                                       torch.full_like(result.model.dit.x_embedder.proj.weight[:,:96],0.1))
            self.assertEqual(result.model.ppp_decoder.decoder.project_out.op_list[-1].conv.out_channels,1)
            self.assertFalse(result.condition_encoder.training)
            for invalid_mask in (None, torch.ones(300), torch.zeros(1,300),
                                 torch.full((1,300),2), torch.full((1,300),float('nan'))):
                with self.subTest(invalid_mask=invalid_mask):
                    null = {'uncond_prompt_embeds':torch.ones(1,300,8)}
                    if invalid_mask is not None:
                        null['uncond_prompt_embeds_mask'] = invalid_mask
                    torch.save(null,path)
                    calls.clear()
                    with self.assertRaisesRegex(ValueError,'attention mask'):
                        initialize_ppp_model(config,null_embed_path=path,caption_channels=8,
                                             model_builder=builder,checkpoint_loader=checkpoint,vae_loader=vae)
                    self.assertEqual(calls,[])


    def test_fixed_endpoint(self):
        from diffusion import Scheduler
        scheduler = Scheduler('', noise_schedule='linear_flow', predict_v=True,
                              pred_sigma=False, flow_shift=3.0)
        self.assertEqual(float(scheduler.timestep_map[0]),0.0)

    def test_contract_adaptation_and_gradients(self):
        torch.manual_seed(7)
        logger = logging.getLogger(__name__)
        dit = RadarGenPPP(input_size=2, hidden_size=32, depth=1, num_heads=4,
                         caption_channels=8, model_max_length=3, attn_type='vanilla',
                         ffn_type='mlp', use_pe=False, class_dropout_prob=0.0)
        old = dit.x_embedder.proj.weight.detach().clone()
        bias = dit.x_embedder.proj.bias.detach().clone()
        inflate_sana_input_channels_for_ppp(dit, logger)
        torch.testing.assert_close(dit.x_embedder.proj.weight[:, :96], old.repeat(1,3,1,1)/3)
        torch.testing.assert_close(dit.x_embedder.proj.bias, bias)
        self.assertTrue(dit.x_embedder.proj.weight[:,96:].count_nonzero())
        decoder = Decoder()
        rgb = decoder.project_out.op_list[-1].conv
        weight, bias = rgb.weight.detach().mean(0,keepdim=True), rgb.bias.detach().mean().view(1)
        adapt_ppp_decoder_output(decoder,logger)
        torch.testing.assert_close(decoder.project_out.op_list[-1].conv.weight,weight)
        torch.testing.assert_close(decoder.project_out.op_list[-1].conv.bias,bias)
        encoder = FrozenConditionEncoder(nn.Conv2d(3,32,1),0.41407)
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertFalse(encoder.encoder.training)
        images = [torch.full((2,3,2,2),float(i+1)) for i in range(3)]
        conditions = encoder(images)
        torch.testing.assert_close(conditions[0],encoder.encoder(images[0])*0.41407)
        model = RadarGenPPPTrainingModel(dit,PPPDecoder(decoder,0.41407)).train()
        y, mask = torch.randn(6,1,3,8), torch.ones(6,1,1,3)
        captured = []
        hook = dit.x_embedder.proj.register_forward_pre_hook(lambda m,args: captured.append(args[0].detach()))
        output = model(conditions,y,mask)
        hook.remove()
        self.assertEqual(PPP_MODALITIES,('density','rcs','doppler'))
        self.assertEqual(output.shape,(2,3,1,2,2))
        spatial = captured[0].reshape(2,3,99,2,2)
        for b in range(2):
            for v in range(3):
                torch.testing.assert_close(spatial[b,v,:96],torch.cat(conditions,1)[b])
                torch.testing.assert_close(spatial[b,v,96:],dit.modality_pe[v,:,None,None].expand(3,2,2))
        self.assertTrue(output[:,1:].count_nonzero())
        for index,loss_fn in [(1,lambda q: masked_rcs_l1(q,torch.ones_like(q)*10,torch.ones_like(q))),
                              (2,lambda q: masked_doppler_loss(q,torch.ones_like(q)*10,torch.ones_like(q),'signed_l1'))]:
            model.zero_grad(set_to_none=True)
            loss_fn(model(conditions,y,mask)[:,index]).backward()
            for parameter in (decoder.early.weight,decoder.project_out.op_list[-1].conv.weight,
                              dit.modality_pe,dit.blocks[0].attn.qkv.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in decoder.parameters()))
        with self.assertRaises(ValueError):
            dit(torch.ones(6),y,mask,bev_condition_maps=conditions)

if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
