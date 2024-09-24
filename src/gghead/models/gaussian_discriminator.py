from dataclasses import dataclass, field, asdict
from typing import Literal, List, Dict, Optional

import numpy as np
import torch
from eg3d.torch_utils.ops import upfirdn2d
from eg3d.training.dual_discriminator import DualDiscriminator, filtered_resizing
from eg3d.training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue
from elias.config import Config, implicit
from torch import nn

from gghead.models.gghead_model import MappingNetworkConfig


@dataclass
class DiscriminatorBlockConfig(Config):
    activation: Literal['relu', 'lrelu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.
    resample_filter: List[int] = field(default_factory=lambda: [1, 3, 3, 1])  # Low-pass filter to apply when resampling activations.
    fp16_channels_last: bool = False  # Use channels-last memory format with FP16?
    freeze_layers: int = 0  # Freeze-D: Number of layers to freeze.


@dataclass
class DiscriminatorEpilogueConfig(Config):
    mbstd_group_size: int = 4  # Group size for the minibatch standard deviation layer, None = entire minibatch.
    mbstd_num_channels: int = 1  # Number of features for the minibatch standard deviation layer, 0 = disable.
    activation: Literal['lrelu', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.


@dataclass
class GaussianDiscriminatorConfig(Config):
    mapping_network_config: MappingNetworkConfig
    block_config: DiscriminatorBlockConfig = DiscriminatorBlockConfig()
    epilogue_config: DiscriminatorEpilogueConfig = DiscriminatorEpilogueConfig()

    architecture: Literal['orig', 'skip', 'resnet'] = 'resnet'  # Architecture: 'orig', 'skip', 'resnet'.
    channel_base: int = 32768  # Overall multiplier for the number of channels.
    channel_max: int = 512  # Maximum number of channels in any layer.
    num_fp16_res: int = 4  # Use FP16 for the N highest resolutions.
    conv_clamp: int = 256  # Clamp the output of convolution layers to +-X, None = disable clamping.
    cmap_dim: Optional[int] = None  # Dimensionality of mapped conditioning label, None = default.
    disc_c_noise: float = 0  # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
    use_dual_discrimination: bool = False

    c_dim: int = implicit()  # Conditioning label (C) dimensionality.
    img_resolution: int = implicit()  # Input resolution.
    img_channels: int = implicit()  # Number of input color channels.
    pretrained_resolution: Optional[int] = implicit()

    @staticmethod
    def from_eg3d_config(
            c_dim,  # Conditioning label (C) dimensionality.
            img_resolution,  # Input resolution.
            img_channels,  # Number of input color channels.
            architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base=32768,  # Overall multiplier for the number of channels.
            channel_max=512,  # Maximum number of channels in any layer.
            num_fp16_res=4,  # Use FP16 for the N highest resolutions.
            conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
            disc_c_noise=0,  # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
            block_kwargs={},  # Arguments for DiscriminatorBlock.
            mapping_kwargs={},  # Arguments for MappingNetwork.
            epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
    ):
        return GaussianDiscriminatorConfig(
            c_dim,
            img_resolution,
            img_channels,
            mapping_network_config=MappingNetworkConfig(**mapping_kwargs),
            block_config=DiscriminatorBlockConfig(**block_kwargs),
            epilogue_config=DiscriminatorEpilogueConfig(**epilogue_kwargs),
            architecture=architecture,
            channel_base=channel_base,
            channel_max=channel_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            cmap_dim=cmap_dim,
            disc_c_noise=disc_c_noise)


class GaussianDiscriminator(nn.Module):
    def __init__(self, config: GaussianDiscriminatorConfig):
        super().__init__()

        img_channels = config.img_channels
        if config.use_dual_discrimination:
            img_channels *= 2

        self.c_dim = config.c_dim
        self.img_resolution = config.img_resolution
        self.pretrained_resolution = config.pretrained_resolution
        self.img_resolution_log2 = int(np.log2(config.img_resolution))
        self.img_channels = img_channels
        self.use_dual_discrimination = config.use_dual_discrimination
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(config.channel_base // res, config.channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - config.num_fp16_res), 8)

        if config.cmap_dim is None:
            cmap_dim = channels_dict[4]
        if config.c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, conv_clamp=config.conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            is_new_layer = config.pretrained_resolution is not None and res > config.pretrained_resolution
            in_channels = channels_dict[res] if res < config.img_resolution and (
                        config.pretrained_resolution is None or res != config.pretrained_resolution) else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block_architecture = 'skip' if is_new_layer else config.architecture
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, architecture=block_architecture,
                                       **asdict(config.block_config), **common_kwargs)
            if is_new_layer:
                block.conv1.weight.data.zero_()
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if config.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=config.c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **asdict(config.mapping_network_config))
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **asdict(config.epilogue_config), **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))
        self.disc_c_noise = config.disc_c_noise

    def forward(self, img: Dict, c, update_emas=False, alpha_new_layers: float = 1, **block_kwargs):
        if self.use_dual_discrimination:
            image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
            img = torch.cat([img['image'], image_raw], 1)
        else:
            img = img['image']

        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if self.pretrained_resolution is not None and res > self.pretrained_resolution:
                # Smoothly blend in outputs of newly added layers (that are not trained yet)
                x, img = block(x, img, **block_kwargs)  # TODO: force x closer to 0 in the beginning? Then discriminator should be more similar to pre-trained
                x = alpha_new_layers * x
            else:
                x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def forward_ggh(self, img: torch.Tensor, c, update_emas=False, **block_kwargs) -> torch.Tensor:
        return self.forward({"image": img}, c, update_emas=update_emas, **block_kwargs)

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
