import torch.nn as nn
import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter


def inflate_sana_input_channels_for_radargen(model, logger, num_conditions: int = 3):
    weight_replication_factor = num_conditions + 1 # +1 for the noised latent
    _weight = model.x_embedder.proj.weight.clone()
    _bias = model.x_embedder.proj.bias.clone()
    _weight = _weight.repeat((1, weight_replication_factor, 1, 1))  # Keep selected channel(s)
    
    # Scale the weights to maintain activation magnitude
    _weight = _weight / float(weight_replication_factor)

    # New conv_in channel
    _weight_channels = _weight.shape[1] + model.modality_pe.shape[1] # Add C_pe for the modality PE
    _n_convin_out_channel = model.x_embedder.proj.out_channels
    _kernel_size = model.x_embedder.proj.kernel_size
    _stride = model.x_embedder.proj.stride
    _padding = model.x_embedder.proj.padding
    _new_conv_in = Conv2d(
        _weight_channels, _n_convin_out_channel, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=True
    )
    # Initialize the weight like in the original SANA model
    w = _new_conv_in.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    # Set the value of the weight and bias as the original, divided by the number of duplications
    _new_conv_in.weight.data[:, :_weight.shape[1]] = _weight
    _new_conv_in.bias = Parameter(_bias)
    _new_conv_in.to(model.x_embedder.proj.weight.device, model.x_embedder.proj.weight.dtype)

    # Finally, replace the original layer with the new one
    model.x_embedder.proj = _new_conv_in
    logger.info("SANA PatchEmbedMS.proj layer was replaced")


def inflate_sana_input_channels_for_ppp(model, logger, num_conditions: int = 3):
    """Adapt an already initialized 32-channel SANA projection to 96+3 channels."""
    old = model.x_embedder.proj
    if (not isinstance(old, nn.Conv2d) or old.in_channels != 32 or old.groups != 1
            or num_conditions != 3 or tuple(model.modality_pe.shape) != (3, 3)):
        raise ValueError("PPP inflation requires the bare 32-channel projection and three modalities/conditions")
    new = nn.Conv2d(99, old.out_channels, old.kernel_size, stride=old.stride,
                    padding=old.padding, dilation=old.dilation, groups=old.groups,
                    bias=old.bias is not None, padding_mode=old.padding_mode,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        nn.init.xavier_uniform_(new.weight.flatten(1))
        new.weight[:, :96].copy_(old.weight.repeat(1, 3, 1, 1) / 3)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.x_embedder.proj = new
    logger.info("PPP PatchEmbedMS.proj initialized: %s", tuple(new.weight.shape))
