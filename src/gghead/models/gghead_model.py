import os
from copy import copy
from dataclasses import dataclass, field, asdict
from typing import Literal, List, Union, Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector.vector_torch import to_homogeneous
from eg3d.datamanager.nersemble import decode_camera_params

from elias.config import Config, implicit

from torch import nn
from torch.nn import init
from torch.nn.functional import grid_sample

from gghead.constants import DEFAULT_INTRINSICS
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.sh_utils import C0, eval_sh
from gghead.config.gaussian_attribute import GaussianAttribute, GaussianAttributeConfig
from gghead.env import REPO_ROOT_DIR
from gghead.models.stylegan2 import GGHGenerator as GGHStyleGAN2Backbone, GGHSynthesisNetwork, GGHSynthesisBlock
from gghead.util.activation import mip_tanh, mip_sigmoid, mip_tanh2
from gghead.util.logging import LoggerBundle
from gghead.util.mesh import gaussians_to_mesh
from gghead.util.rotation import axis_angle_to_quaternion
from gghead.util.uv import gen_tritex


@dataclass
class MappingNetworkConfig(Config):
    num_layers: int = 8  # Number of mapping layers.
    embed_features: Optional[int] = None  # Label embedding dimensionality, None = same as w_dim.
    layer_features: Optional[int] = None  # Number of intermediate features in the mapping layers, None = same as w_dim.
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: float = 0.01  # Learning rate multiplier for the mapping layers.
    # w_avg_beta: float = 0.998  # Decay for tracking the moving average of W during training, None = do not track.


@dataclass
class SynthesisNetworkConfig(Config):
    channel_base: int = 32768  # Overall multiplier for the number of channels.
    channel_max: int = 512  # Maximum number of channels in any layer.
    num_fp16_res: int = 4  # Use FP16 for the N highest resolutions.

    # Block Config
    architecture: Literal['orig', 'skip', 'resnet'] = 'skip'  # Architecture: 'orig', 'skip', 'resnet'.
    resample_filter: List[int] = field(
        default_factory=lambda: [1, 3, 3, 1])  # Low-pass filter to apply when resampling activations.
    conv_clamp: Optional[int] = 256  # Clamp the output of convolution layers to +-X, None = disable clamping.
    fp16_channels_last: bool = False  # Use channels-last memory format with FP16?
    fused_modconv_default: Union[
        bool, str] = True  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.

    # Layer config
    kernel_size: int = 3  # Convolution kernel size.
    use_noise: bool = True  # Enable noise input?
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.

    def get_block_kwargs(self) -> dict:
        block_kwargs = {k: v for k, v in asdict(self).items() if
                        k not in ['channel_base', 'channel_max', 'num_fp16_res']}
        return block_kwargs


@dataclass
class RenderingConfig(Config):
    c_gen_conditioning_zero: bool = True  # if true, fill generator pose conditioning label with dummy zero vector
    c_scale: float = 1  # Scale factor for generator pose conditioning
    box_warp: float = 1  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].


@dataclass
class SuperResolutionConfig(Config):
    use_superresolution: bool = False
    superresolution_version: int = 1
    n_channels: int = 3
    n_downsampling_layers: int = 1
    use_skip: bool = True
    cbase: int = 32768  # Capacity multiplier
    cmax: int = 512  # Max. feature maps
    fused_modconv_default: str = 'inference_only'
    sr_num_fp16_res: int = 4  # Number of fp16 layers in superresolution
    sr_antialias: bool = True
    noise_mode: Literal['random', 'none'] = 'none'


@dataclass
class GGHeadConfig(Config):
    z_dim: int = 512
    w_dim: int = 512
    c_dim: int = implicit(default=25)
    # img_resolution: int
    mapping_network_config: MappingNetworkConfig = MappingNetworkConfig()
    synthesis_network_config: SynthesisNetworkConfig = SynthesisNetworkConfig()
    rendering_config: RenderingConfig = RenderingConfig()
    super_resolution_config: SuperResolutionConfig = SuperResolutionConfig()

    uv_attributes: List[GaussianAttribute] = field(default_factory=lambda: [
        GaussianAttribute.POSITION])  # Which attributes should be predicted in UV space
    n_triplane_channels: int = 16  # number of channels for each TriPlane
    disable_position_offsets: bool = False  # If set, no position offsets will be predicted and Gaussians will always be fixed to template vertices
    use_align_corners: bool = False  # For grid_sample()
    interpolation_mode: str = 'bilinear'

    # FLAME template
    n_flame_subdivisions: int = 0  # How often the FLAME template mesh should be subdivided (increases number of predicted Gaussians)
    use_uniform_flame_vertices: bool = False  # If true, will not use predefined FLAME vertices, but instead uniformly distribute points on mesh surface using UV
    n_uniform_flame_vertices: int = 64  # How many points (squared) should be sampled in FLAME's UV space. Final number of Gaussians will be slightly smaller due to holes in UV map
    n_shells: int = 1
    shell_distance: float = 0.05
    use_learnable_template_offsets: bool = False  # If true, position of flame vertices can be adapted during training
    use_learnable_template_offset_plane: bool = False
    learnable_template_offset_plane_size: int = 64
    use_gsm_flame_template: bool = False  # Use template with back removed and more efficient UV layout
    use_flame_template_v2: bool = False
    use_sphere_template: bool = False
    use_plane_template: bool = False
    use_auxiliary_sphere: bool = False  # Predict additional set of Gaussians in front of face to models microphones, hands, other stuff that occludes the face
    auxiliary_sphere_radius: float = 0.1
    auxiliary_sphere_position: Tuple[float, float, float] = (0, -0.1, 0.4)
    uv_grid_threshold: Optional[
        float] = None  # If set, template positions with uv coordinates closer to the boundary than threshold will be dropped

    plane_resolution: int = 256
    effective_plane_resolution: Optional[int] = None
    pretrained_plane_resolution: Optional[int] = implicit()
    pretrained_resolution: Optional[int] = implicit()
    # Gaussian Attribute decoding
    use_position_activation: bool = True
    use_color_activation: bool = True
    use_scale_activation: bool = False
    center_scale_activation: bool = False  # If true, the max_scale option will be properly applied inside the softplus
    use_initial_scales: bool = False
    use_rotation_activation: bool = False
    use_periodic_rotation_activation: bool = False  # If true, will use sine() activation instead of tanh()
    normalize_quaternions: bool = True
    position_attenuation: float = 1
    position_range: float = 1  # Maximum range that predicted positions can have. 1 means [-1, 1]
    color_attenuation: float = 1
    scale_attenuation: float = 1
    rotation_attenuation: float = 1
    scale_offset: float = -5
    additional_scale_offset: float = 0
    max_scale: float = 1
    use_softplus_scale_activation: bool = False
    no_exp_scale_activation: bool = False  # Disable 3DGS default exp() scale activation
    scale_overshoot: float = 0.001
    color_overshoot: float = 0  # Allows prediction of colors slightly outside of the range to prevent tanh saturation. EG3D uses 0.001
    opacity_overshoot: float = 0  # Avoid having to predict ridiculously large opacities to saturate sigmoid
    clamp_opacity: bool = False
    use_optimizable_gaussian_attributes: bool = False  # For debugging: Gaussians are directly learnable instead of building them from predicted UV / TriPlanes
    gaussian_attribute_config: GaussianAttributeConfig = GaussianAttributeConfig()
    use_zero_conv_position: bool = False
    use_zero_conv_scale: bool = False
    use_density_map: bool = False

    # Gaussian Attribute MLP
    mlp_layers: int = 1
    mlp_hidden_dim: int = 256

    # Gaussian Hierarchy MLP
    use_gaussian_hierarchy: bool = False
    exclude_position_from_hierarchy: bool = False  # If true, positions will be directly sampled in uv map while all other attributes will be decoded with MLP
    use_uv_position_and_hierarchy: bool = False  # If true, positions will be directly sampled in uv map in addition to decoded offset
    n_gaussians_per_texel: int = 1
    gaussian_hierarchy_feature_dim: int = 16  # number of features in uv map that will be decoded into actual Gaussian UV attributes
    use_separate_hierarchy_mlps: bool = False  # If true, use one MLP per attribute

    # Gradient Multipliers
    grad_multiplier_position: Optional[float] = None
    grad_multiplier_scale: Optional[float] = None
    grad_multiplier_rotation: Optional[float] = None
    grad_multiplier_color: Optional[float] = None
    grad_multiplier_opacity: Optional[float] = None

    # Background modeling
    use_background_plane: bool = False  # If True, will additionally generate Gaussians behind the FLAME template
    curve_background_plane: bool = False
    background_cylinder_angle: float = torch.pi / 2  # Angle of the cylinder patch if curve_background_plane=True. Larger angle = larger background plane
    background_plane_distance: float = 1  # Distance of background plane to FLAME template
    background_plane_width: float = 1
    background_plane_height: float = 1
    n_background_gaussians: int = 64  # Number of background gaussians PER DIMENSION that will be distributed on background plane. E.g., 128 -> 128x128
    use_background_cnn: bool = False  # If True, will use 3 additional RGB channels from StyleGAN2 to models background
    use_background_upsampler: bool = False  # If use_background_cnn=True and the rendering resolution is larger than the backbone synthesis resolution
    use_separate_background_cnn: bool = False  # If True, will use an additional StyleGAN network to models background
    n_background_channels: int = 3  # Relevant if bg upsampler is used. Will be number of channels for intermediate upsampling layers
    use_masks: bool = False
    fix_alpha_blending: bool = False
    use_cnn_adaptor: bool = False

    # Maintenance
    maintenance_interval: Optional[int] = None  # How often Gaussians should be densified / pruned
    maintenance_grad_threshold: float = 0.01
    use_pruning: bool = False
    use_densification: bool = True
    use_template_update: bool = False
    template_update_attributes: List[GaussianAttribute] = field(default_factory=list)
    position_map_update_factor: float = 1  # How much of the average position map should be baked into the template at each maintenance step
    prune_opacity_threshold: float = 0.005

    use_autodecoder: bool = False  # Whether to assign one learnable latent code to each person
    use_flame_to_bfm_registration: bool = False
    load_average_offset_map: bool = False
    img_resolution: int = 512
    neural_rendering_resolution: int = 512

    n_persons: Optional[int] = implicit()
    random_background: Optional[bool] = implicit(default=False)
    return_background: Optional[bool] = implicit(default=False)
    background_color: Tuple[int, int, int] = implicit(
        default=(255, 255,
                 255))  # Background color to use during training. Should match the background color used in the dataset

    @staticmethod
    def from_eg3d_config(z_dim,  # Input latent (Z) dimensionality.
                         c_dim,  # Conditioning label (C) dimensionality.
                         w_dim,  # Intermediate latent (W) dimensionality.
                         img_resolution,  # Output resolution.
                         img_channels,  # Number of output color channels.
                         sr_num_fp16_res=0,
                         mapping_kwargs={},  # Arguments for MappingNetwork.
                         rendering_kwargs={},
                         sr_kwargs={},
                         **synthesis_kwargs,  # Arguments for SynthesisNetwork
                         ) -> 'GGHeadConfig':
        config = GGHeadConfig(z_dim, w_dim,
                                                 mapping_network_config=MappingNetworkConfig(**mapping_kwargs),
                                                 synthesis_network_config=SynthesisNetworkConfig(**synthesis_kwargs),
                                                 rendering_config=RenderingConfig(**rendering_kwargs),
                                                 use_flame_to_bfm_registration=True,
                                                 img_resolution=img_resolution)
        config.c_dim = c_dim
        return config

    @staticmethod
    def default() -> 'GGHeadConfig':
        config = GGHeadConfig(512, 512,
                                                 mapping_network_config=MappingNetworkConfig(),
                                                 synthesis_network_config=SynthesisNetworkConfig(),
                                                 rendering_config=RenderingConfig(),
                                                 use_flame_to_bfm_registration=True)
        config.c_dim = 25
        return config


@dataclass
class GaussianAttributeOutput:
    gaussian_attributes: Dict[GaussianAttribute, torch.Tensor]

    # Needed for regularization
    raw_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None

    # Diagnostics
    uv_map: Optional[torch.Tensor] = None  # [B, S, UV, H_f, W_f]
    background_uv_map: Optional[torch.Tensor] = None
    auxiliary_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None
    raw_auxiliary_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None


@dataclass
class GGHeadOutput(dict):
    images: torch.Tensor  # [B, 3, H, W] in [-1, 1]
    images_raw: torch.Tensor  # [B, 3, H_raw, W_raw] in [-1, 1]
    images_depth: torch.Tensor  # [B, H, W]
    gaussian_attribute_output: GaussianAttributeOutput
    masks: Optional[torch.Tensor] = None  # [B, 1, H, W] in [-1, 1]
    backgrounds: Optional[torch.Tensor] = None  # [B, 3, H, W] in [-1, 1]
    images_features: Optional[torch.Tensor] = None

    # # Used for Gaussian maintenance
    # viewspace_points: Optional[List[torch.Tensor]] = None
    # visibility_filters: Optional[torch.Tensor] = None
    # radii: Optional[torch.Tensor] = None

    def __getitem__(self, key) -> torch.Tensor:
        # Legacy support for EG3D code. Behave like a dictionary
        if key == 'image':
            images = self.images
            if self.masks is not None:
                images = torch.cat([images, self.masks], dim=1)
            if self.backgrounds is not None:
                images = torch.cat([images, self.backgrounds], dim=1)
            return images
        elif key == 'image_raw':
            return self.images_raw
        elif key == 'image_depth':
            return self.images_depth
        else:
            raise ValueError(f"Unknown key: {key}")

    def __setitem__(self, key, value):
        # Legacy support for EG3D code. Behave like a dictionary
        if key == 'image':
            if self.masks is None:
                if self.backgrounds is None:
                    self.images = value
                else:
                    self.images = value[:, :-3]
                    self.backgrounds = value[:, -3:]
            else:
                if self.backgrounds is None:
                    self.images = value[:, :-1]
                    self.masks = value[:, [-1]]
                else:
                    self.images = value[:, :-4]
                    self.masks = value[:, [3]]
                    self.backgrounds = value[:, -3:]
        elif key == 'image_raw':
            self.images_raw = value
        elif key == 'image_depth':
            self.images_depth = value
        else:
            raise ValueError(f"Unknown key: {key}")

    def keys(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return iter(['image', 'image_raw', 'image_depth'])

    def values(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return iter([self[key] for key in self.keys()])

    def items(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return zip(self.keys(), self.values())

    def __len__(self) -> int:
        return self.images.shape[0]


class GGHeadModel(nn.Module):
    z_dim: int

    def __init__(self, config: GGHeadConfig, logger_bundle: Optional[LoggerBundle] = None,
                 post_init: bool = True):
        super().__init__()

        self._config = config

        # Gaussians have the following attributes:
        #  - Position: 3
        #  - Scale: 3
        #  - Rotation: 4
        #  - Color: 3
        #  - Opacity: 1

        self._all_gaussian_attribute_names = [GaussianAttribute.POSITION, GaussianAttribute.SCALE,
                                              GaussianAttribute.ROTATION, GaussianAttribute.OPACITY,
                                              GaussianAttribute.COLOR]
        self._uv_attribute_names = [attribute_name for attribute_name in config.uv_attributes
                                    if (
                                            attribute_name != GaussianAttribute.POSITION or not config.disable_position_offsets)]

        self._n_uv_channels = sum(
            [gaussian_attribute.get_n_channels(config.gaussian_attribute_config) for gaussian_attribute in
             self._uv_attribute_names])

        n_gaussian_attributes = self._n_uv_channels

        # Setup StyleGAN2
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.w_dim = config.w_dim

        n_backbone_channels = self._n_uv_channels
        if self._config.use_background_cnn:
            n_backbone_channels += self._config.n_background_channels

        self.backbone = GGHStyleGAN2Backbone(self.z_dim, self.c_dim, self.w_dim,
                                             img_resolution=self._config.plane_resolution,
                                             pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                             img_channels=n_backbone_channels,
                                             mapping_kwargs=asdict(config.mapping_network_config),
                                             **asdict(config.synthesis_network_config))

        if self._config.use_background_cnn and self._config.use_background_upsampler and self._config.img_resolution > self._config.plane_resolution:
            img_resolution_log2 = int(np.log2(self._config.img_resolution))
            plane_resolution_log2 = int(np.log2(self._config.plane_resolution))
            n_upsampling_layers = img_resolution_log2 - plane_resolution_log2
            background_channels = [self._config.n_background_channels] * n_upsampling_layers + [3]
            self._background_upsampling_blocks = []
            for i in range(n_upsampling_layers):
                in_channels = background_channels[i]
                out_channels = background_channels[i + 1]
                use_fp16 = False
                is_last = i == (n_upsampling_layers - 1)

                block = GGHSynthesisBlock(in_channels, out_channels, w_dim=self.w_dim,
                                          resolution=2 ** (plane_resolution_log2 + i + 1),
                                          img_channels=3, is_last=is_last, use_fp16=use_fp16,
                                          **config.synthesis_network_config.get_block_kwargs())
                self._background_upsampling_blocks.append(block)  # TODO: Set torgb() to zeros?

            self._background_upsampling_blocks = nn.ModuleList(self._background_upsampling_blocks)

        self._uv_attribute_start_channel = dict()
        self._uv_attribute_n_channels = dict()
        c = 0
        for attribute_name in self._uv_attribute_names:
            n_channels = attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            self._uv_attribute_start_channel[attribute_name] = c
            self._uv_attribute_n_channels[attribute_name] = n_channels
            c += n_channels

        if config.use_zero_conv_position:
            n_channels_position = GaussianAttribute.POSITION.get_n_channels(config.gaussian_attribute_config)

            c = 0
            for attribute_name in self._uv_attribute_names:
                if attribute_name == GaussianAttribute.POSITION:
                    break
                c += attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            self._position_start_channel = c
            self._n_position_channels = n_channels_position

            zero_conv_position = nn.Conv2d(n_channels_position, n_channels_position, 1)
            init.zeros_(zero_conv_position.weight)
            init.zeros_(zero_conv_position.bias)
            self._zero_conv_position = zero_conv_position

        self.neural_rendering_resolution = self._config.neural_rendering_resolution
        self.rendering_config = config.rendering_config

        if config.use_gsm_flame_template:
            flame_template_mesh = trimesh.load(
                f"{REPO_ROOT_DIR}/assets/flame_uv_no_back_close_mouth_no_subdivision.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_flame_template_v2:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/flame_template_v2.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_sphere_template:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/sphere_template.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_plane_template:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/plane_template.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        else:
            raise ValueError("No mesh template specified!")

        vertices = flame_template_mesh.vertices
        faces = flame_template_mesh.faces
        vertex_attributes = {"uvs_per_flame_vertex": uvs_per_flame_vertex}

        idxim, tidxim, barim = gen_tritex(uv_coords, faces, uv_faces, config.plane_resolution)

        v0_map = vertices[idxim[..., 0]]
        v1_map = vertices[idxim[..., 1]]
        v2_map = vertices[idxim[..., 2]]
        flame_position_map = barim[..., [0]] * v0_map + barim[..., [1]] * v1_map + barim[
            ..., [2]] * v2_map  # Maps texels to 3D positions

        xs = torch.linspace(-1, 1, steps=self._config.n_uniform_flame_vertices)
        ys = torch.linspace(-1, 1, steps=self._config.n_uniform_flame_vertices)

        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        sampled_uv_coords = torch.stack([ys, xs], dim=-1)

        torch_position_map = torch.from_numpy(flame_position_map).float().permute(2, 0, 1)  # [3, H_map, W_map]
        torch_face_index_map = torch.from_numpy(idxim).permute(2, 0, 1)
        valid_uv_map = (torch_face_index_map > 0).any(dim=0).float()[None]  # [1, H_map, W_map]

        valid_samples = torch.nn.functional.grid_sample(valid_uv_map.unsqueeze(0), sampled_uv_coords.unsqueeze(0),
                                                        align_corners=config.use_align_corners,
                                                        mode=config.interpolation_mode)[0].permute(1, 2, 0)
        valid_samples = valid_samples[:, :, 0] > 0.99
        valid_uv_coords = sampled_uv_coords[valid_samples]  # [G, 2]
        uv_grid = valid_uv_coords.unsqueeze(0).unsqueeze(2)  # [1, G, 1, 2]
        self.register_buffer("_uv_grid", uv_grid.contiguous())
        self.register_buffer("_uv_position_map", torch_position_map.contiguous())

        sampled_positions = torch.nn.functional.grid_sample(torch_position_map.unsqueeze(0), uv_grid,
                                                            align_corners=config.use_align_corners,
                                                            mode=config.interpolation_mode)[0, :, :, 0].T  # [G, 3]
        flame_vertices = sampled_positions

        if self._config.use_flame_to_bfm_registration:
            # Register FLAME template to (scaled) BFM template
            flame_to_bfm_neutral_37_face_only = torch.tensor([
                [2.682716929557429, 0.010446918125791843, 0.04746649927210553, 0.0030014961233934233],
                [-0.009421598630294275, 2.682515580606053, -0.05790466856909523, 0.04740944787628047],
                [-0.047680604263683285, 0.05772857372357329, 2.682112266656848, -0.0024605516344398115],
                [0.0, 0.0, 0.0, 1.0]
            ])
            flame_vertices = (to_homogeneous(flame_vertices) @ flame_to_bfm_neutral_37_face_only.T)[..., :3]

            if hasattr(self, "_uv_position_map"):
                self._uv_position_map = (to_homogeneous(
                    self._uv_position_map.permute(1, 2, 0)) @ flame_to_bfm_neutral_37_face_only.T)[..., :3].permute(2,
                                                                                                                    0,
                                                                                                                    1).contiguous()

        flame_vertices = flame_vertices.unsqueeze(0)  # [1, G, 3]
        self.register_buffer("_flame_vertices", flame_vertices.contiguous())

        # Setup Gaussian Model for rendering
        self._gaussian_model = GaussianModel(sh_degree=self._config.gaussian_attribute_config.sh_degree)
        self._gaussian_model.active_sh_degree = self._config.gaussian_attribute_config.sh_degree
        self._gaussian_model.opacity_activation = self._apply_opacity_activation
        if config.no_exp_scale_activation:
            # Note: inverse_scaling_activation is not changed
            self._gaussian_model.scaling_activation = self._apply_scale_activation

        gaussian_bg = torch.Tensor([1 for _ in range(config.gaussian_attribute_config.n_color_channels)])
        gaussian_bg_train = torch.Tensor(self._config.background_color) / 255
        self.register_buffer("_gaussian_bg", gaussian_bg, persistent=False)
        self.register_buffer("_gaussian_bg_train", gaussian_bg_train, persistent=False)

        self._last_planes = None

        # Needed for EG3D visualizer
        self.img_resolution = config.img_resolution
        self.rendering_kwargs = {
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
        }

        # Logging
        self._logger_bundle = logger_bundle

    #     if post_init:
    #         self.post_init()
    #
    # def post_init(self):
    #     """
    #     Regular __init__() is called before checkpoint is loaded. The post_init() method is called after a potential checkpoint is loaded.
    #     Its purpose is to incorporate any additional parameters/buffers loaded from the checkpoint.
    #     """
    #
    #     if self._config.uv_grid_threshold is not None:
    #         # In case, the uv grid filtering scheme is changed after loading a checkpoint, the filtering has to be performed here, not in __init__().
    #         # Maybe, this is obsolete now, since we do not load _flame_vertices and _uv_grid anymore from the checkpoint (which would override the effect
    #         # of _filter_uv_grid() if done already in __init__())
    #         self._filter_uv_grid(self._config.uv_grid_threshold)
    #
    #     if self._config.use_template_update:
    #         # Checkpoint contains _maintenance_average_position_map, but it has not yet been applied to change the positions of the flame vertices
    #         # (which are not persisted)
    #         self._update_template_vertices()

    def sample_z(self, person_ids: torch.Tensor) -> torch.Tensor:
        if self._config.use_autodecoder:
            z = self._identity_codes(person_ids)
        else:
            z = torch.randn((len(person_ids), self._config.z_dim)).cuda()

        return z

    # ==========================================================
    # Forward Helpers
    # ==========================================================

    def sample_uv_map(self, planes: torch.Tensor, uv_grid: torch.Tensor, n_shells: int = 1):
        B, C, H_f, W_f = planes.shape
        S = n_shells
        C_uv = self._n_uv_channels

        uv_map = planes
        if self._config.use_zero_conv_position and GaussianAttribute.POSITION in self._uv_attribute_names:
            position_start_channel = self._position_start_channel

            zeroed_positions = self._zero_conv_position(
                uv_map[:, position_start_channel: position_start_channel + self._zero_conv_position.in_channels])
            uv_map = torch.cat([uv_map[:, :position_start_channel],
                                zeroed_positions,
                                uv_map[:, position_start_channel + self._zero_conv_position.in_channels:]], dim=1)

        uv_map = uv_map.reshape(B * S, C_uv, H_f, W_f)  # [B*S, UV, H_f, W_f]

        uv_attributes = grid_sample(uv_map, uv_grid.repeat(B * S, 1, 1, 1),
                                    align_corners=self._config.use_align_corners,
                                    mode=self._config.interpolation_mode)  # [B*S, C_uv, G, 1]
        uv_attributes = uv_attributes.squeeze(3).permute(0, 2, 1)  # [B*Shells, G, C_uv]
        G = uv_attributes.shape[1]

        uv_attributes = uv_attributes.reshape(B, S * G, C_uv)

        return uv_attributes, uv_map

    def predict_planes(self, ws: torch.Tensor, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                       alpha_plane_resolution: Optional[float] = None, **synthesis_kwargs):
        # Predict 2D planes
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        return planes

    def predict_gaussian_attributes(self,
                                    planes: torch.Tensor,
                                    return_raw_attributes: bool = False,
                                    return_uv_map: bool = False) -> GaussianAttributeOutput:

        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()

        # Predict UV textures and collect gaussian attributes
        C_uv = self._n_uv_channels
        planes_uv = planes[:, -self._n_uv_channels:]
        planes_main = planes_uv[:, : C_uv]
        uv_attributes, uv_map_main = self.sample_uv_map(planes_main, self._uv_grid)
        uv_map = uv_map_main

        collected_uv_attributes, raw_uv_attributes = self._collect_gaussian_attributes(
            self._uv_attribute_names,
            uv_attributes,
            return_raw_attributes=return_raw_attributes)

        gaussian_attributes.update(collected_uv_attributes)
        raw_gaussian_attributes.update(raw_uv_attributes)

        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]

        if self.training and self._logger_bundle is not None:
            self._logger_bundle.log_metrics({
                "Analyze/norm_gaussian_positions": gaussian_positions.norm(dim=-1).mean(),
                "Analyze/gaussian_scales": gaussian_scales.mean(),
                "Analyze/angle_gaussian_rotations": 2 * torch.acos(gaussian_rotations[..., 0]),
                "Analyze/gaussian_opacities": gaussian_opacities.mean(),
                "Analyze/gaussian_colors": gaussian_colors.norm(dim=-1).mean()
            })

        gaussian_attribute_output = GaussianAttributeOutput(gaussian_attributes,
                                                            raw_gaussian_attributes,
                                                            uv_map=uv_map if return_uv_map else None)
        return gaussian_attribute_output

    def _log_gradients(self, name: str, tensor: torch.Tensor):
        if tensor.requires_grad and self._logger_bundle is not None:
            tensor.register_hook(lambda grad, n=name: self._logger_bundle.log_metrics(
                {f"Analyze/Gradients/grad_gaussian_{n}": grad.norm(dim=1).mean()}))

    # TODO: Do we need this?
    def get_uv_rendering(self, c: torch.Tensor, output: GGHeadOutput,
                         include_transparent_gaussians: bool = False) -> torch.Tensor:
        B = len(c)
        device = c.device
        resolution = output['image'].shape[2]
        debug_gaussian_attributes = copy(output.gaussian_attribute_output.gaussian_attributes)
        uv_colors = self._uv_grid.squeeze(2)  # [1, G, 2]
        uv_colors = torch.concatenate([uv_colors, -torch.ones((1, uv_colors.shape[1], 1), device=device)],
                                      dim=-1)  # [1, G, 3]
        debug_gaussian_attributes[GaussianAttribute.COLOR] = uv_colors.repeat(B, 1, 1)
        if include_transparent_gaussians:
            debug_gaussian_attributes[GaussianAttribute.OPACITY] = torch.ones_like(
                debug_gaussian_attributes[GaussianAttribute.OPACITY])

        all_uv_renders = []
        for i, single_c in enumerate(c.cpu()):
            gaussian_model = self._setup_gaussian_model(debug_gaussian_attributes, i)
            pose, intrinsics = decode_camera_params(single_c)
            intrinsics = intrinsics.rescale(resolution)
            gs_cam = pose_to_rendercam(pose, intrinsics, resolution, resolution, device=device)

            with torch.cuda.device(device):
                uv_render = render(gs_cam, gaussian_model, PipelineParams2(), torch.tensor([1., 1., 1.], device=device),
                                   override_color=uv_colors[0])
            all_uv_renders.append(uv_render['render'])

        all_uv_renders = torch.stack(all_uv_renders)
        return all_uv_renders

    # TODO: Do we need this?
    def get_gaussian_mesh(self,
                          gaussian_attributes: Dict[GaussianAttribute, torch.Tensor],
                          idx: int = 0,
                          use_spheres: bool = True, random_colors: bool = True, ellipsoid_res: int = 5,
                          scale_factor: float = 1.5,
                          overwrite_colors: Optional[torch.Tensor] = None,
                          opacity_threshold: float = 0.01,
                          max_n_gaussians: Optional[int] = None,
                          min_scale: Optional[float] = None,
                          max_scale: Optional[float] = None,
                          include_alphas: bool = False) -> trimesh.Trimesh:
        if overwrite_colors is None:
            device = gaussian_attributes[GaussianAttribute.POSITION].device
            pose_front = Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 2.7), pose_type=PoseType.CAM_2_WORLD,
                              camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)

            gaussian_sh_ref_cam = pose_to_rendercam(pose_front, DEFAULT_INTRINSICS, 512, 512, device=device)
            sh_degree = self._config.gaussian_attribute_config.sh_degree
            shs_view = gaussian_attributes[GaussianAttribute.COLOR][idx].view(-1, (sh_degree + 1) ** 2, 3).permute(0, 2,
                                                                                                                   1)
            dir_pp = (gaussian_attributes[GaussianAttribute.POSITION][idx] - gaussian_sh_ref_cam.camera_center.repeat(1,
                                                                                                                      1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)

            gaussian_colors = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)
        else:
            gaussian_colors = overwrite_colors

        gaussian_opacities = self._gaussian_model.opacity_activation(
            gaussian_attributes[GaussianAttribute.OPACITY][idx])
        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION][idx]
        gaussian_scales = self._gaussian_model.scaling_activation(gaussian_attributes[GaussianAttribute.SCALE][idx])
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION][idx]

        if min_scale is not None:
            gaussian_scales = gaussian_scales.clamp(min=min_scale)
        if max_scale is not None:
            gaussian_scales = gaussian_scales.clamp(max=max_scale)

        combined_mesh = gaussians_to_mesh(gaussian_positions, gaussian_scales, gaussian_rotations, gaussian_colors,
                                          gaussian_opacities,
                                          use_spheres=use_spheres, random_colors=random_colors,
                                          ellipsoid_res=ellipsoid_res, scale_factor=scale_factor,
                                          opacity_threshold=opacity_threshold, max_n_gaussians=max_n_gaussians,
                                          include_alphas=include_alphas)

        return combined_mesh

    # ==========================================================
    # Main forward
    # ==========================================================

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False,
                use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone,
                              use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_config.c_gen_conditioning_zero:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_config.c_scale, truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff,
                                     update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False,
                  return_raw_attributes: bool = False, return_uv_map: bool = False,
                  alpha_plane_resolution: Optional[float] = None,
                  return_masks: bool = False,
                  sh_ref_cam: Optional[Pose] = None,
                  **synthesis_kwargs) -> GGHeadOutput:

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics_matrix = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        planes = self.predict_planes(ws, update_emas=update_emas, cache_backbone=cache_backbone,
                                     use_cached_backbone=use_cached_backbone,
                                     alpha_plane_resolution=alpha_plane_resolution, **synthesis_kwargs)
        if self._config.use_background_cnn:
            background_rgb = planes[:, -self._config.n_background_channels:]

            if self._config.use_background_upsampler:
                x = background_rgb
                bg_img = background_rgb[:,
                         :3]  # First 3 channels of background plane tensor have special meaning: they are already the low-res background
                for block in self._background_upsampling_blocks:
                    x, bg_img = block(x, bg_img, ws[:, :block.num_conv + block.num_torgb])
                background_rgb = bg_img

            if neural_rendering_resolution > background_rgb.shape[-1]:
                background_rgb = torch.nn.functional.interpolate(background_rgb, (
                    neural_rendering_resolution, neural_rendering_resolution), mode='bilinear')

            background_rgb = mip_tanh(background_rgb)
            planes = planes[:, :-self._config.n_background_channels]
        else:
            background_rgb = None

        gaussian_attribute_output = self.predict_gaussian_attributes(
            planes,
            return_raw_attributes=return_raw_attributes,
            return_uv_map=return_uv_map)
        gaussian_attributes = gaussian_attribute_output.gaussian_attributes

        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]  # [B, G, SH*3]
        B = len(c)
        G = gaussian_colors.shape[1]
        C = 3
        gaussian_colors = gaussian_colors.view(B, G, -1, C)

        # Gradient logging
        for attribute_name, attribute_value in gaussian_attributes.items():
            self._log_gradients(attribute_name, attribute_value)

        device = self._gaussian_bg.device

        # Rasterization
        rgb_images = []
        masks = []
        backgrounds = []
        for i in range(B):
            cam_2_world_pose = Pose(cam2world_matrix[i].cpu().numpy(), pose_type=PoseType.CAM_2_WORLD,
                                    disable_rotation_check=True)
            intrinsics = Intrinsics(intrinsics_matrix[i].cpu().numpy())
            intrinsics = intrinsics.rescale(neural_rendering_resolution,
                                            inplace=False)  # EG3D intrinsics are given in normalized format wrt to [0-1] image
            gaussian_camera = pose_to_rendercam(cam_2_world_pose, intrinsics, neural_rendering_resolution,
                                                neural_rendering_resolution, device=device)

            self._gaussian_model._xyz = gaussian_positions[i]
            self._gaussian_model._features_dc = gaussian_colors[i][:, [0]]
            self._gaussian_model._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
            self._gaussian_model._scaling = gaussian_scales[i]
            self._gaussian_model._rotation = gaussian_rotations[i].contiguous()  # Rotation needs to be contiguous!
            self._gaussian_model._opacity = gaussian_opacities[i]

            if sh_ref_cam is not None:
                gaussian_sh_ref_cam = pose_to_rendercam(sh_ref_cam, intrinsics, neural_rendering_resolution, neural_rendering_resolution, device=device)

                sh_degree = self._config.gaussian_attribute_config.sh_degree
                n_feature_channels = self._config.gaussian_attribute_config.n_color_channels
                shs_view = self._gaussian_model.get_features.view(-1, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 2, 1)
                dir_pp = (self._gaussian_model.get_xyz - gaussian_sh_ref_cam.camera_center.repeat(1, 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
                sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
                colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
                override_color = colors
            else:
                override_color = None

            gaussian_bg = self._gaussian_bg_train

            # The with statement is necessary, since otherwise the rasterizer internally may move something to the wrong GPU
            with torch.cuda.device(device):
                rendered_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(), gaussian_bg,
                                        override_color=override_color)

            rendered_image = rendered_image['render']  # [3, H, W]
            rendered_image = rendered_image * 2 - 1  # [0, 1] -> [-1, 1]

            if self._config.use_background_cnn or return_masks:
                # Obtain alpha image by rendering a second time with all Gaussians set to black
                black_colors = torch.ones_like(gaussian_colors[i][:, 0]) * 0

                with torch.cuda.device(device):
                    rendered_alpha_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(),
                                                  self._gaussian_bg, override_color=black_colors)
                rendered_alpha_image = 1 - rendered_alpha_image['render']  # 0 is background, 1 is foreground

                if self._config.use_background_cnn:
                    rendered_image = (rendered_image + 1) / 2  # Blending has to be done in [0, 1] range
                    bg_img = (background_rgb[i] + 1) / 2
                    # Alpha blending of Gaussian rendering with CNN background
                    if self._config.fix_alpha_blending:
                        # rendered_image contains blended white colors. Remove them here
                        rendered_image = rendered_image - (1 - rendered_alpha_image) * self._gaussian_bg[:, None, None]
                        rendered_image = (1 - rendered_alpha_image) * bg_img + rendered_image
                    else:
                        rendered_image = (1 - rendered_alpha_image) * bg_img + rendered_alpha_image * rendered_image
                    rendered_image = rendered_image * 2 - 1  # [0, 1] -> [-1, 1]

                if return_masks:
                    # Rendered alpha image has 3 channels, but they are all the same
                    rendered_alpha_image = rendered_alpha_image * 2 - 1  # [0, 1] -> [-1, 1]
                    masks.append(rendered_alpha_image[[0]])  # [1, H, W]

            rgb_images.append(rendered_image)

        rgb_images_direct = torch.stack(rgb_images)
        masks = torch.stack(masks) if self._config.use_masks or return_masks else None

        rgb_images = rgb_images_direct
        rgb_images_raw = rgb_images_direct

        output = GGHeadOutput(rgb_images, rgb_images_raw, rgb_images,
                              gaussian_attribute_output=gaussian_attribute_output,
                              masks=masks)

        return output

    def _setup_gaussian_model(self, gaussian_attributes: Dict[GaussianAttribute, torch.Tensor],
                              i: int) -> GaussianModel:
        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]  # [B, G, SH*3]
        B, G, _ = gaussian_colors.shape
        gaussian_colors = gaussian_colors.view(B, G, -1, 3)

        self._gaussian_model._xyz = gaussian_positions[i]
        self._gaussian_model._features_dc = gaussian_colors[i][:, [0]]
        self._gaussian_model._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
        self._gaussian_model._scaling = gaussian_scales[i]
        self._gaussian_model._rotation = gaussian_rotations[
            i].contiguous()  # Important: Rotation needs to be contiguous!
        self._gaussian_model._opacity = gaussian_opacities[i]

        return self._gaussian_model

    def _collect_gaussian_attributes(self, attribute_names: List[GaussianAttribute], predictions: torch.Tensor,
                                     return_raw_attributes: bool = False) \
            -> Tuple[Dict[GaussianAttribute, torch.Tensor], Dict[GaussianAttribute, torch.Tensor]]:
        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()
        c = 0
        for attribute_name in attribute_names:
            n_channels = attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            attribute_map = predictions[..., c: c + n_channels]  # Slice corresponding channels from sampled plane
            if return_raw_attributes:
                raw_gaussian_attributes[attribute_name] = attribute_map

            if self.training and self._logger_bundle is not None:
                self._logger_bundle.log_metrics({
                    f"Analyze/norm_raw_{attribute_name}": attribute_map.norm(dim=-1).mean(),
                    f"Analyze/max_raw_{attribute_name}": attribute_map.max(),
                    f"Analyze/min_raw_{attribute_name}": attribute_map.min(),
                })

            attribute_map = self._apply_gaussian_attribute_activation(attribute_name, attribute_map)
            gaussian_attributes[attribute_name] = attribute_map
            c += n_channels

        return gaussian_attributes, raw_gaussian_attributes

    # ==========================================================
    # Activations
    # ==========================================================

    def _apply_gaussian_attribute_activation(self, attribute_name: GaussianAttribute, value: torch.Tensor) -> torch.Tensor:

        B = value.shape[0]

        # POSITION
        if attribute_name == GaussianAttribute.POSITION:
            value = self._apply_position_activation(value)
            value = self._flame_vertices.repeat(B, 1, 1) + value

        # SCALE
        elif attribute_name == GaussianAttribute.SCALE:
            # These scale activations are working around 3DGS's default exp() activation for scaling
            scale_offset = self._config.scale_offset
            value = -(value + scale_offset)
            if self._config.center_scale_activation:
                value = self._config.max_scale - torch.nn.functional.softplus(value + self._config.max_scale)
            else:
                value = self._config.max_scale - torch.nn.functional.softplus(value)

        # ROTATION
        elif attribute_name == GaussianAttribute.ROTATION:
            if self._config.gaussian_attribute_config.use_rodriguez_rotation:
                # Rodriguez -> Quaternion
                if self._config.use_rotation_activation:
                    value = torch.tanh(self._config.rotation_attenuation * value) * (2 * torch.pi)
                value = axis_angle_to_quaternion(value)
                value = torch.cat([value[..., [3]], value[..., :3]], dim=-1)  # xyzr -> rxyz
            elif self._config.normalize_quaternions:
                value = value / value.norm(dim=2).unsqueeze(2)  # Normalize quaternions

            value = value.contiguous()  # Important: Rotation needs to be contiguous!

        # COLOR
        elif attribute_name == GaussianAttribute.COLOR:
            value = self._apply_color_activation(value)

        return value

    def _apply_position_activation(self, value: torch.Tensor) -> torch.Tensor:
        # Dividing by some value before tanh() is super important. Otherwise, tanh() saturates super quickly at -1 or 1
        if self._config.use_position_activation:
            value = self._config.position_range * torch.tanh(self._config.position_attenuation * value)
        else:
            value = self._config.position_attenuation * value
        return value

    def _apply_opacity_activation(self, value: torch.Tensor) -> torch.Tensor:
        return mip_sigmoid(value, overshoot=self._config.opacity_overshoot, clamp=self._config.clamp_opacity)

    def _apply_color_activation(self, value: torch.Tensor) -> torch.Tensor:
        if self._config.use_color_activation:
            color_value = value[..., :3]  # First 3 channels are always color values
            color_value = mip_tanh(color_value, overshoot=self._config.color_overshoot)
            color_value = color_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

            # TODO: SH bands have the same scaling as color bands
            sh_value = value[..., 3:]
            sh_value = mip_tanh(sh_value, overshoot=self._config.color_overshoot)
            sh_value = sh_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

            value = torch.cat([color_value, sh_value], dim=-1)

        return value

    def _apply_cnn_color_activation(self, value: torch.Tensor) -> torch.Tensor:
        return mip_tanh2(value, clamp=True)

    def _apply_scale_activation(self, value: torch.Tensor) -> torch.Tensor:
        # Scales should be between [0, 0.05]
        value = value + np.exp(self._config.scale_offset)
        return mip_sigmoid(value, overshoot=self._config.scale_overshoot) * np.exp(self._config.max_scale)