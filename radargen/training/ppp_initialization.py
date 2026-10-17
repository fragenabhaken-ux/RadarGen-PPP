"""Fresh PPP model construction only; no optimizer, training, or resume support."""

from dataclasses import dataclass
import logging
from pathlib import Path
import json
import zipfile

import torch

from diffusion.utils.config import model_init_config
from diffusion.model_modification import inflate_sana_input_channels_for_ppp
from .ppp_decoder import FrozenConditionEncoder, PPPDecoder, adapt_ppp_decoder_output
from .radargen_ppp_training_model import RadarGenPPPTrainingModel


@dataclass
class PPPModelComponents:
    model: torch.nn.Module
    condition_encoder: torch.nn.Module | None
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def initialize_dino_ppp_model(config, *, device="cpu", load_pretrained=True):
    """Build the direct BEV predictor; a full PPP checkpoint needs no RGB weights.

    The shared RGB DINO processes each BEV condition with its own LoRA before
    feature fusion and prediction of three raw radar fields. There is no
    conditioning autoencoder or text encoder.
    """
    from radargen.models.dino_ppp import DinoPPP

    kwargs = dict(config.model.extra or {})
    if load_pretrained:
        checkpoint = kwargs.get("checkpoint_path", "")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Local DINOv3 checkpoint unavailable: {checkpoint}")
    else:
        # Resume/inference restores every parameter from the trained PPP model.
        kwargs["checkpoint_path"] = ""
        kwargs["pretrained"] = False
    kwargs.setdefault("grad_checkpointing", config.train.grad_checkpointing)
    model = DinoPPP(**kwargs).to(device)
    return PPPModelComponents(model, None, (), ())


def initialize_ppp_model(config, *, null_embed_path, device="cpu", checkpoint_path=None,
                         caption_channels=None, logger=None, model_builder=None,
                         checkpoint_loader=None, vae_loader=None):
    """Optionally load SANA -> adapt input; load DC-AE -> adapt -> wrap.

    Default loaders require local DC-AE assets and, when enabled, a local SANA file;
    they do not download checkpoints. Callback injection supports small CPU
    substitute-model tests. This initializes fresh weights, not PPP checkpoints.
    """
    logger = logger or logging.getLogger(__name__)
    extra = config.model.extra
    if (config.model.model != 'RadarGenPPP_600M_P1_D28' or extra.get('fixed_timestep') != 0.0
            or extra.get('num_maps') != 3 or extra.get('num_conditions') != 3):
        raise ValueError('Expected the configured three-modality PPP model with fixed timestep 0.0')
    if config.resume_from or config.model.resume_from:
        raise ValueError('This helper implements fresh initialization only; PPP resume is not implemented')
    if config.vae.vae_type != 'dc-ae' or config.vae.weight_dtype != 'float32' or config.vae.vae_latent_dim != 32:
        raise ValueError('PPP requires the FP32 DC-AE with 32 latent channels')
    if not Path(null_embed_path).is_file():
        raise ValueError('An existing baseline-format empty-prompt embedding file is required')
    load_sana = extra.get('load_sana_weights', True)
    if not isinstance(load_sana, bool):
        raise ValueError('model.extra.load_sana_weights must be boolean')
    source = checkpoint_path or config.load_from or config.model.load_from
    if load_sana and not source:
        raise ValueError('Pretrained SANA weights are required')
    if load_sana and checkpoint_loader is None:
        if not Path(source).is_file():
            raise ValueError('Supply checkpoint_path for the local pretrained SANA file; downloads are disabled')
        from diffusion.utils.checkpoint import load_checkpoint
        checkpoint_loader = load_checkpoint
    if model_builder is None:
        from diffusion.model import nets  # Register factories only when initialization is requested.
        from diffusion.model.builder import build_model
        model_builder = build_model
    if vae_loader is None:
        validate_local_ppp_assets(source if load_sana else None, config.vae.vae_pretrained)
        def vae_loader(name, model_path, target_device):
            from diffusion.model.builder import get_vae
            return get_vae(name, model_path, target_device)

    kwargs = model_init_config(config, latent_size=config.model.image_size // config.vae.vae_downsample_rate)
    if caption_channels is not None:
        kwargs['caption_channels'] = caption_channels
    kwargs.update(num_maps=3, num_conditions=3, fixed_timestep=0.0,
                  pred_sigma=False, learn_sigma=False)
    null = torch.load(null_embed_path, map_location='cpu', weights_only=True)
    embedding = null.get('uncond_prompt_embeds')
    expected = (1, config.text_encoder.model_max_length, kwargs['caption_channels'])
    if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != expected or not torch.isfinite(embedding).all():
        raise ValueError(f'Empty-prompt embedding must be finite with shape {expected}')
    attention_mask = null.get('uncond_prompt_embeds_mask')
    mask_shape = (1, config.text_encoder.model_max_length)
    if (not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != mask_shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
            or not (attention_mask == 1).any()):
        raise ValueError(f'Empty-prompt attention mask must be binary with shape {mask_shape} and at least one active token')
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    dit = model_builder(config.model.model, config.train.grad_checkpointing,
                        config.model.fp32_attention, **kwargs).to(device)
    if dit.x_embedder.proj.in_channels != 32:
        raise ValueError('PPP initialization requires a bare 32-channel input projection')
    missing, unexpected = (), ()
    if load_sana:
        _, missing, unexpected, _ = checkpoint_loader(str(source), dit, load_ema=False,
                                                     null_embed_path=str(null_embed_path))
        logger.info('SANA missing keys: %s; unexpected keys: %s', missing, unexpected)
        allowed_missing = {'modality_pe', 'pos_embed'}
        if config.model.qk_norm:
            # Older SANA checkpoints lack QK norms; retain constructor defaults.
            allowed_missing.update(
                f'blocks.{index}.attn.{norm}.{name}'
                for index, block in enumerate(dit.blocks)
                for norm in ('q_norm', 'k_norm')
                for name, _ in getattr(block.attn, norm).named_parameters()
            )
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(f'Unexplained SANA checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    else:
        logger.info('SANA weight loading disabled; using random DiT initialization')
    inflate_sana_input_channels_for_ppp(dit, logger, replicate_weights=load_sana)
    ae = vae_loader(config.vae.vae_type, config.vae.vae_pretrained, device).float()
    ae.requires_grad_(False).eval()
    adapt_ppp_decoder_output(ae.decoder, logger)
    # Match baseline vae_encode/vae_decode: checkpoint cfg overrides fallback.
    scale = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
    condition_encoder = FrozenConditionEncoder(ae.encoder, scale)
    decoder_checkpointing = config.train.extra.get('decoder_grad_checkpointing', config.train.grad_checkpointing)
    if not isinstance(decoder_checkpointing, bool):
        raise ValueError('train.extra.decoder_grad_checkpointing must be boolean')
    ae.decoder.gradient_checkpointing = decoder_checkpointing
    logger.info('PPP decoder activation checkpointing: %s', decoder_checkpointing)
    decoder = PPPDecoder(ae.decoder, scale)
    model = RadarGenPPPTrainingModel(dit, decoder)
    logger.info('PPP adapted shapes: input=%s modality=%s output=%s',
                tuple(dit.x_embedder.proj.weight.shape), tuple(dit.modality_pe.shape),
                tuple(ae.decoder.project_out.op_list[-1].conv.weight.shape))
    return PPPModelComponents(model, condition_encoder, tuple(missing), tuple(unexpected))


@torch.no_grad()
def encode_ppp_empty_text(tokenizer, text_encoder, *, batch_size, model_max_length, device):
    """Baseline Gemma/Qwen empty-prompt branch, expanded to 3B entries."""
    if batch_size <= 0 or model_max_length <= 0:
        raise ValueError('Batch size and text length must be positive')
    tokens = tokenizer([''] * (3 * batch_size), padding='max_length', max_length=model_max_length,
                       truncation=True, return_tensors='pt').to(device)
    indices = [0] + list(range(-model_max_length + 1, 0))
    embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0][:, None]
    return embeddings[:, :, indices], tokens.attention_mask[:, None, None][:, :, :, indices]


def validate_local_ppp_assets(sana_path, ae_path):
    """Reject absent/truncated containers before allocating pretrained models.

    Structural completeness does not establish an upstream checksum match.
    """
    ae = Path(ae_path)
    sana_bytes = 0
    if sana_path is not None:
        sana = Path(sana_path)
        if not sana.is_file() or not zipfile.is_zipfile(sana):
            raise ValueError(f'Missing or incomplete SANA checkpoint: {sana}')
        with zipfile.ZipFile(sana) as archive:
            if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                raise ValueError('SANA checkpoint has no PyTorch state metadata')
            bad = archive.testzip()
            if bad:
                raise ValueError(f'SANA checkpoint CRC failure: {bad}')
        sana_bytes = sana.stat().st_size
    config_path, weights = ae / 'config.json', ae / 'model.safetensors'
    if not config_path.is_file() or not weights.is_file():
        raise ValueError(f'DC-AE requires config.json and model.safetensors in {ae}')
    if json.loads(config_path.read_text()).get('model_name') != 'dc-ae-f32c32-sana-1.1':
        raise ValueError('Unexpected local DC-AE architecture')
    with weights.open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little')
        if length <= 0 or length > weights.stat().st_size - 8:
            raise ValueError('Incomplete safetensors header')
        header = json.loads(stream.read(length))
    offsets = sorted(v['data_offsets'] for k, v in header.items() if k != '__metadata__')
    cursor = 0
    for begin, end in offsets:
        if begin != cursor or end < begin:
            raise ValueError('Invalid safetensors offsets')
        cursor = end
    if 8 + length + cursor != weights.stat().st_size:
        raise ValueError('Incomplete safetensors data')
    return {'sana_bytes': sana_bytes, 'dc_ae_bytes': weights.stat().st_size}
