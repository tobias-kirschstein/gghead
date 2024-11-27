import os
from typing import Optional, Tuple, Literal

import torch
import tyro
from eg3d import dnnlib
from eg3d.metrics import metric_main
from eg3d.metrics.metric_main import register_metric, fid100, fid1k, fid50k_full

from gghead.config.eg3d_train_arguments import EG3DTrainArguments
from gghead.config.gaussian_attribute import GaussianAttribute, GaussianAttributeConfig
from gghead.dataset.image_folder_dataset import MaskMethod, GGHeadImageFolderDatasetConfig, GGHeadMaskImageFolderDataset
from gghead.eg3d.loss import GGHeadStyleGAN2LossConfig
from gghead.eg3d.train_loop import launch_training
from gghead.env import GGHEAD_DATA_PATH
from gghead.model_manager.base_model_manager import GGHGeneratorType, GGHeadGANConfig, GGHeadGANOptimizerConfig, OptimizerConfig, GGHeadTrainSetup
from gghead.model_manager.finder import find_model_manager
from gghead.model_manager.gghead_model_manager import GGHeadExperimentConfig
from gghead.models.gaussian_discriminator import GaussianDiscriminatorConfig, DiscriminatorBlockConfig, DiscriminatorEpilogueConfig
from gghead.models.gghead_model import GGHeadConfig, MappingNetworkConfig, SynthesisNetworkConfig, RenderingConfig, SuperResolutionConfig


def main(
        cfg: str,  # Base configuration
        data: str,  # Training data [ZIP|DIR]
        gpus: int,  # Number of GPUs to use
        batch: int,  # Total batch size
        /,
        name: Optional[str] = None,
        resume_run: Optional[str] = None,  # Resume training of another model
        resume_checkpoint: int = -1,
        reset_cur_nimg: bool = False,

        # Architecture
        generator_type: GGHGeneratorType = 'gaussians',
        gen_pose_cond: bool = True,
        disc_pose_cond: bool = True,
        gpc_reg_prob: float = 0.5,
        use_superresolution: bool = False,
        superresolution_version: int = 1,
        use_dual_discrimination: bool = True,
        plane_resolution: int = 256,
        effective_plane_resolution: Optional[int] = None,
        plane_resolution_blend_kimg: int = 1000,
        plane_resolution_start_kimg: Optional[int] = None,
        n_triplane_channels: int = 16,
        mlp_layers: int = 1,  # Gaussian Attribute MLP
        use_align_corners: bool = False,  # For grid_sample()

        # Resolutions
        resolution: int = 256,
        neural_rendering_resolution: Optional[int] = None,
        neural_rendering_resolution_final: Optional[int] = None,  # for second stage of EG3D. Gradually increase neural rendering resolution

        # Optimization
        lr_gen: float = 0.0025,
        lr_disc: float = 0.002,
        n_phases_gen: int = 1,
        effective_res_disc: float = 1,
        blur_fade_kimg: int = 200,
        separate_lr_template_offsets: Optional[float] = None,
        template_offsets_gradient_accumulation: int = 1,
        aug: Literal['noaug', 'ada', 'fixed'] = 'noaug',
        ada_target: float = 0.6,
        kimg: int = 25000,

        # FLAME
        n_subdivisions: int = 0,
        use_uniform_flame_vertices: bool = True,
        n_uniform_flame_vertices: int = 256,
        use_gsm_flame_template: bool = True,
        use_flame_template_v2: bool = False,
        use_sphere_template: bool = False,
        use_plane_template: bool = False,

        # Gaussian Decoding
        uv_attributes: str = 'position,scale,rotation,color,opacity',
        disable_position_offsets: bool = False,
        use_rodriguez: bool = True,
        use_position_activation: bool = True,
        use_scale_activation: bool = True,
        use_rotation_activation: bool = True,
        no_exp_scale_activation: bool = False,
        use_softplus_scale_activation: bool = False,
        position_range: float = 0.25,
        use_initial_scales: bool = False,
        sh_degree: int = 1,
        opacity_overshoot: float = 0.001,
        clamp_opacity: bool = False,
        use_zero_conv_position: bool = True,

        # Background
        use_background_cnn: bool = False,
        use_separate_background_cnn: bool = False,
        use_background_upsampler: bool = False,
        n_background_channels: int = 3,
        fix_alpha_blending: bool = False,
        n_background_gaussians: int = 64,
        use_masks: bool = True,
        blur_masks: bool = True,
        apply_masks: bool = True,
        return_masks: bool = False,
        mask_method: MaskMethod = 'modnet',
        random_background: bool = False,
        return_background: bool = False,
        background_color: Tuple[int, int, int] = (255, 255, 255),

        # Regularizations
        gamma: float = 1,  # R1 regularization weight for discriminator
        lambda_gaussian_position: float = 0,
        lambda_gaussian_scale: float = 0,
        lambda_raw_gaussian_position: float = 0.1,
        lambda_raw_gaussian_scale: float = 0.05,
        lambda_raw_gaussian_rotation: float = 0,
        lambda_raw_gaussian_color: float = 0,
        lambda_raw_gaussian_opacity: float = 0,
        lambda_learnable_template_offsets: float = 0,
        lambda_tv_learnable_template_offsets: float = 0,
        lambda_tv_uv_rendering: float = 0,
        tv_uv_include_transparent_gaussians: bool = False,
        lambda_beta_loss: float = 0,
        use_l1_scale_reg: bool = False,
        lambda_raw_scale_std: float = 0,

        # Resuming
        overwrite_n_subdivisions: Optional[int] = None,
        overwrite_n_uniform_flame_vertices: Optional[int] = None,
        overwrite_grad_multiplier_position: Optional[float] = None,
        overwrite_gamma: Optional[float] = None,
        overwrite_lambda_raw_gaussian_position: Optional[float] = None,
        overwrite_lambda_raw_gaussian_scale: Optional[float] = None,
        overwrite_lambda_raw_gaussian_rotation: Optional[float] = None,
        overwrite_lambda_raw_gaussian_color: Optional[float] = None,
        overwrite_lambda_raw_gaussian_opacity: Optional[float] = None,
        overwrite_lambda_raw_scale_std: Optional[float] = None,
        overwrite_lambda_beta_loss: Optional[float] = None,
        overwrite_lambda_tv_uv_rendering: Optional[float] = None,
        overwrite_tv_uv_include_transparent_gaussians: Optional[bool] = None,
        overwrite_resolution: Optional[int] = None,
        overwrite_plane_resolution: Optional[int] = None,
        overwrite_new_layers_gen_blend_kimg: Optional[int] = None,
        overwrite_maintenance_interval: Optional[int] = None,
        overwrite_maintenance_grad_threshold: Optional[float] = None,
        overwrite_use_pruning: Optional[bool] = None,
        overwrite_separate_lr_template_offsets: Optional[float] = None,
        overwrite_align_corners: bool = False,
        overwrite_interpolation_mode: Optional[str] = None,
        overwrite_uv_grid_threshold: Optional[float] = None,
        overwrite_use_superresolution: Optional[bool] = None,
        overwrite_use_background_cnn: Optional[bool] = None,
        overwrite_use_separate_background_cnn: Optional[bool] = None,
        overwrite_use_background_upsampler: Optional[bool] = None,
        overwrite_n_background_channels: Optional[int] = None,
        overwrite_apply_masks: Optional[bool] = None,
        overwrite_use_masks: Optional[bool] = None,
        overwrite_fix_alpha_blending: Optional[bool] = None,
        n_feature_channels: int = 32,
        n_downsampling_layers: int = 1,
        freeze_generator: bool = False,
        smooth_D_intro: bool = True,
        smooth_res_intro: bool = False,
        smooth_G_intro: bool = True,
        smooth_G_blend: bool = True,

        # Logging
        metrics: str = 'fid100,fid1k,fid10k',
        image_snapshot_ticks: int = 50,
        use_vis_window: bool = False,
):
    use_gaussians = generator_type in {'gaussians', 'GSM'}
    data = f"{GGHEAD_DATA_PATH}/{data}"

    # Initialize config.
    if use_gaussians:
        neural_rendering_resolution_initial = neural_rendering_resolution
    elif neural_rendering_resolution == resolution:
        neural_rendering_resolution_initial = neural_rendering_resolution
    else:
        neural_rendering_resolution_initial = 64
    opts = EG3DTrainArguments(None, cfg, data, gpus, batch, gamma,
                              gen_pose_cond=gen_pose_cond,
                              gpc_reg_prob=gpc_reg_prob,
                              metrics=metrics.split(',') if metrics != '' else [],

                              # Gaussian Splatting specific
                              neural_rendering_resolution_initial=neural_rendering_resolution_initial,
                              neural_rendering_resolution_final=neural_rendering_resolution_final,
                              density_reg=0 if use_gaussians else 0.25,
                              aug=aug,
                              target=ada_target,
                              kimg=kimg
                              )

    c = dnnlib.EasyDict()  # Main config dict.
    c.use_gaussians = use_gaussians
    c.use_vis_window = use_vis_window
    c.G_kwargs = dnnlib.EasyDict()
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    use_dual_discrimination = use_dual_discrimination and (generator_type == 'triplanes' and resolution != neural_rendering_resolution or use_superresolution)

    if resume_run is not None:
        model_manager = find_model_manager(resume_run)
        dataset_config: GGHeadImageFolderDatasetConfig = model_manager.load_dataset_config()
        model_config: GGHeadGANConfig = model_manager.load_model_config()
        optimizer_config: GGHeadGANOptimizerConfig = model_manager.load_optimization_config()
        resume_checkpoint = model_manager._resolve_checkpoint_id(resume_checkpoint)

        if overwrite_n_subdivisions is not None:
            model_config.generator_config.n_flame_subdivisions = overwrite_n_subdivisions

        if overwrite_n_uniform_flame_vertices is not None:
            model_config.generator_config.n_uniform_flame_vertices = overwrite_n_uniform_flame_vertices

        if overwrite_grad_multiplier_position is not None:
            model_config.generator_config.grad_multiplier_position = overwrite_grad_multiplier_position

        if overwrite_gamma is not None:
            optimizer_config.loss_config.r1_gamma = overwrite_gamma

        if overwrite_lambda_raw_gaussian_position is not None:
            optimizer_config.loss_config.lambda_raw_gaussian_position = overwrite_lambda_raw_gaussian_position

        if overwrite_lambda_raw_gaussian_scale is not None:
            optimizer_config.loss_config.lambda_raw_gaussian_scale = overwrite_lambda_raw_gaussian_scale

        if overwrite_lambda_raw_gaussian_rotation is not None:
            optimizer_config.loss_config.lambda_raw_gaussian_rotation = overwrite_lambda_raw_gaussian_rotation

        if overwrite_lambda_raw_gaussian_color is not None:
            optimizer_config.loss_config.lambda_raw_gaussian_color = overwrite_lambda_raw_gaussian_color

        if overwrite_lambda_raw_gaussian_opacity is not None:
            optimizer_config.loss_config.lambda_raw_gaussian_opacity = overwrite_lambda_raw_gaussian_opacity

        if overwrite_lambda_beta_loss is not None:
            optimizer_config.loss_config.lambda_beta_loss = overwrite_lambda_beta_loss

        if overwrite_lambda_raw_scale_std is not None:
            optimizer_config.loss_config.lambda_raw_scale_std = overwrite_lambda_raw_scale_std

        if overwrite_lambda_tv_uv_rendering is not None:
            optimizer_config.loss_config.lambda_tv_uv_rendering = overwrite_lambda_tv_uv_rendering

        if overwrite_tv_uv_include_transparent_gaussians is not None:
            optimizer_config.loss_config.tv_uv_include_transparent_gaussians = overwrite_tv_uv_include_transparent_gaussians

        if overwrite_resolution is not None:
            pretrained_resolution = model_config.discriminator_config.img_resolution if model_config.discriminator_config.pretrained_resolution is None else model_config.discriminator_config.pretrained_resolution
            model_config.generator_config.img_resolution = overwrite_resolution
            model_config.generator_config.neural_rendering_resolution = overwrite_resolution
            model_config.generator_config.pretrained_resolution = pretrained_resolution
            model_config.discriminator_config.img_resolution = overwrite_resolution
            model_config.discriminator_config.pretrained_resolution = pretrained_resolution
            optimizer_config.loss_config.neural_rendering_resolution_initial = overwrite_resolution
            if smooth_D_intro:
                optimizer_config.loss_config.new_layers_disc_start_kimg = resume_checkpoint  # For a very smooth introduction of the new high-res discriminator block
            if smooth_res_intro:
                optimizer_config.loss_config.effective_res_disc_start_kimg = resume_checkpoint
                optimizer_config.loss_config.pretrained_resolution = pretrained_resolution
            dataset_config.resolution = overwrite_resolution

        if overwrite_plane_resolution is not None:
            pretrained_plane_resolution = model_config.generator_config.plane_resolution
            model_config.generator_config.pretrained_plane_resolution = pretrained_plane_resolution
            model_config.generator_config.plane_resolution = overwrite_plane_resolution

            if smooth_G_intro:
                optimizer_config.loss_config.new_layers_gen_start_kimg = resume_checkpoint

            if smooth_G_blend:
                optimizer_config.loss_config.plane_resolution_start_kimg = resume_checkpoint
                model_config.generator_config.effective_plane_resolution = pretrained_plane_resolution

        if overwrite_new_layers_gen_blend_kimg is not None:
            optimizer_config.loss_config.new_layers_gen_blend_kimg = overwrite_new_layers_gen_blend_kimg

        if overwrite_maintenance_interval is not None:
            model_config.generator_config.maintenance_interval = overwrite_maintenance_interval
            optimizer_config.loss_config.use_gaussian_maintenance = True

        if overwrite_maintenance_grad_threshold is not None:
            model_config.generator_config.maintenance_grad_threshold = overwrite_maintenance_grad_threshold

        if overwrite_use_pruning is not None:
            model_config.generator_config.use_pruning = overwrite_use_pruning

        if overwrite_separate_lr_template_offsets is not None:
            optimizer_config.separate_lr_template_offsets = overwrite_separate_lr_template_offsets

        if overwrite_align_corners:
            model_config.generator_config.use_align_corners = overwrite_align_corners

        if overwrite_interpolation_mode is not None:
            model_config.generator_config.interpolation_mode = overwrite_interpolation_mode

        if overwrite_uv_grid_threshold is not None:
            model_config.generator_config.uv_grid_threshold = overwrite_uv_grid_threshold

        if neural_rendering_resolution_final:
            optimizer_config.loss_config.neural_rendering_resolution_final = neural_rendering_resolution_final

        if overwrite_use_background_cnn is not None:
            model_config.generator_config.use_background_cnn = overwrite_use_background_cnn

        if overwrite_use_separate_background_cnn is not None:
            model_config.generator_config.use_separate_background_cnn = overwrite_use_separate_background_cnn

        if overwrite_use_background_upsampler is not None:
            model_config.generator_config.use_background_upsampler = overwrite_use_background_upsampler

        if overwrite_n_background_channels is not None:
            model_config.generator_config.n_background_channels = overwrite_n_background_channels

        if overwrite_apply_masks is not None:
            dataset_config.apply_masks = overwrite_apply_masks

        if overwrite_use_masks is not None:
            dataset_config.use_masks = overwrite_use_masks

        if overwrite_fix_alpha_blending is not None:
            model_config.generator_config.fix_alpha_blending = overwrite_fix_alpha_blending

        if overwrite_use_superresolution is not None:
            model_config.generator_config.super_resolution_config.use_superresolution = overwrite_use_superresolution
            model_config.generator_config.super_resolution_config.n_channels = n_feature_channels
            model_config.generator_config.gaussian_attribute_config.n_color_channels = n_feature_channels
            model_config.generator_config.super_resolution_config.superresolution_version = superresolution_version

            if superresolution_version == 1:
                optimizer_config.loss_config.new_layers_gen_start_kimg = resume_checkpoint
            elif superresolution_version == 2:
                model_config.generator_config.super_resolution_config.n_downsampling_layers = n_downsampling_layers

        dataset_config.path = data
        optimizer_config.loss_config.aug = aug
        optimizer_config.loss_config.ada_target = ada_target

        model_config.use_masks = dataset_config.use_masks and (not dataset_config.apply_masks or dataset_config.return_masks)

        optimizer_config.loss_config.blur_init_sigma = 0  # Disable blur rampup.
        optimizer_config.loss_config.gpc_reg_fade_kimg = 0  # Disable swapping rampup
        optimizer_config.freeze_generator = freeze_generator

        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.
    else:

        # ----------------------------------------------------------
        # Dataset config
        # ----------------------------------------------------------

        dataset_config = GGHeadImageFolderDatasetConfig(
            opts.data,
            resolution=resolution,
            use_labels=opts.cond,
            max_size=None,
            xflip=opts.mirror,
            random_seed=opts.seed,
            use_masks=use_masks,
            return_masks=return_masks,
            mask_method=mask_method,
            apply_masks=apply_masks,
            random_background=random_background,
            background_color=background_color,
            return_background=return_background)
        proxy_dataset = GGHeadMaskImageFolderDataset(dataset_config)
        dataset_config.max_size = len(proxy_dataset)

        if opts.cond and not dataset_config.use_labels:
            raise ValueError('--cond=True requires labels specified in dataset.json')

        # ----------------------------------------------------------
        # Model Config
        # ----------------------------------------------------------

        c_dim = proxy_dataset.label_dim
        img_channels = proxy_dataset.num_channels
        neural_rendering_resolution = resolution if neural_rendering_resolution is None else neural_rendering_resolution

        if generator_type in {'gaussians', 'triplanes'}:
            uv_attributes = [GaussianAttribute.from_name(uv_attribute) for uv_attribute in uv_attributes.split(',') if uv_attribute != '']
            generator_config = GGHeadConfig(
                z_dim=512,
                w_dim=512,
                neural_rendering_resolution=neural_rendering_resolution,
                img_resolution=resolution,
                plane_resolution=plane_resolution,
                effective_plane_resolution=effective_plane_resolution,
                mapping_network_config=MappingNetworkConfig(num_layers=opts.map_depth),
                synthesis_network_config=SynthesisNetworkConfig(
                    channel_base=opts.cbase,
                    channel_max=opts.cmax,
                    num_fp16_res=opts.g_num_fp16_res,
                    conv_clamp=256 if opts.g_num_fp16_res > 0 else None,
                    fused_modconv_default='inference_only'),
                rendering_config=RenderingConfig(
                    c_gen_conditioning_zero=not gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
                    c_scale=opts.c_scale,  # mutliplier for generator pose conditioning label
                    box_warp=1,
                ),
                super_resolution_config=SuperResolutionConfig(
                    use_superresolution=use_superresolution,
                    superresolution_version=superresolution_version,
                    n_channels=32,
                ),
                gaussian_attribute_config=GaussianAttributeConfig(
                    use_rodriguez_rotation=use_rodriguez,
                    sh_degree=sh_degree,
                ),

                # FLAME
                use_flame_to_bfm_registration=True,
                n_flame_subdivisions=n_subdivisions,
                use_uniform_flame_vertices=use_uniform_flame_vertices,
                n_uniform_flame_vertices=n_uniform_flame_vertices,
                use_align_corners=use_align_corners,
                use_gsm_flame_template=use_gsm_flame_template,
                use_flame_template_v2=use_flame_template_v2,
                use_sphere_template=use_sphere_template,
                use_plane_template=use_plane_template,

                uv_attributes=uv_attributes,
                disable_position_offsets=disable_position_offsets,

                # Limit Gaussian representation to reasonable positions/scales
                position_range=position_range,
                use_scale_activation=use_scale_activation,
                use_rotation_activation=use_rotation_activation,
                use_position_activation=use_position_activation,
                max_scale=-3,
                scale_offset=-5,
                use_initial_scales=use_initial_scales,
                center_scale_activation=True,
                color_overshoot=0.001,
                opacity_overshoot=opacity_overshoot,
                clamp_opacity=clamp_opacity,
                no_exp_scale_activation=no_exp_scale_activation,
                use_softplus_scale_activation=use_softplus_scale_activation,
                use_zero_conv_position=use_zero_conv_position,

                # Gaussian Attribute MLP
                n_triplane_channels=n_triplane_channels,
                mlp_layers=mlp_layers,
                mlp_hidden_dim=256,

                # Background modeling
                background_plane_distance=0.5,
                background_plane_width=1 / 2,
                background_plane_height=1,
                background_cylinder_angle=torch.pi,
                curve_background_plane=True,
                use_background_cnn=use_background_cnn,
                use_separate_background_cnn=use_separate_background_cnn,
                use_background_upsampler=use_background_upsampler,
                n_background_channels=n_background_channels,
                fix_alpha_blending=fix_alpha_blending,
                n_background_gaussians=n_background_gaussians,
                use_masks=use_masks and (not apply_masks or return_masks),
            )
            generator_config.gaussian_attribute_config.n_color_channels = generator_config.super_resolution_config.n_channels if use_superresolution else 3
            generator_config.random_background = random_background
            generator_config.return_background = return_background
            generator_config.background_color = background_color

        discriminator_config = GaussianDiscriminatorConfig(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            disc_c_noise=opts.disc_c_noise,
            num_fp16_res=opts.d_num_fp16_res,
            conv_clamp=256 if opts.d_num_fp16_res > 0 else None,
            use_dual_discrimination=use_dual_discrimination,
            mapping_network_config=MappingNetworkConfig(),
            block_config=DiscriminatorBlockConfig(
                freeze_layers=opts.freezed,
            ),
            epilogue_config=DiscriminatorEpilogueConfig(
                mbstd_group_size=opts.mbstd_group
            )
        )
        # Communicate dataset -> model config
        generator_config.c_dim = c_dim
        if disc_pose_cond:
            discriminator_config.c_dim = c_dim
        else:
            discriminator_config.c_dim = 0
        discriminator_config.img_resolution = resolution
        discriminator_config.img_channels = img_channels
        model_config = GGHeadGANConfig(generator_config, discriminator_config, generator_type=generator_type)

        # ----------------------------------------------------------
        # Optimizer config
        # ----------------------------------------------------------

        generator_optimizer_config = OptimizerConfig(
            lr=lr_gen,
            beta1=0,
            beta2=0.99,
            eps=1e-8,
            n_phases=n_phases_gen)
        discriminator_optimizer_config = OptimizerConfig(
            lr=lr_disc,
            beta1=0,
            beta2=0.99,
            eps=1e-8)
        optimizer_config = GGHeadGANOptimizerConfig(
            generator_optimizer_config=generator_optimizer_config,
            discriminator_optimizer_config=discriminator_optimizer_config,
            separate_lr_template_offsets=separate_lr_template_offsets,
            template_offsets_gradient_accumulation=template_offsets_gradient_accumulation,
            loss_config=GGHeadStyleGAN2LossConfig(
                r1_gamma=gamma,
                style_mixing_prob=opts.style_mixing_prob,
                neural_rendering_resolution_initial=opts.neural_rendering_resolution_initial,
                neural_rendering_resolution_final=opts.neural_rendering_resolution_final,
                neural_rendering_resolution_fade_kimg=opts.neural_rendering_resolution_fade_kimg,
                plane_resolution_start_kimg=plane_resolution_start_kimg,
                plane_resolution_blend_kimg=plane_resolution_blend_kimg,
                dual_discrimination=use_dual_discrimination,
                gpc_reg_fade_kimg=opts.gpc_reg_fade_kimg,
                gpc_reg_prob=gpc_reg_prob if gen_pose_cond else None,
                blur_init_sigma=10,  # Blur the images seen by the discriminator.
                blur_fade_kimg=int(batch * blur_fade_kimg / 32),  # Fade out the blur during the first N kimg.
                filter_mode='antialiased',  # Filter mode for raw images ['antialiased', 'none', float [0-1]]

                effective_res_disc=effective_res_disc,

                lambda_gaussian_position=lambda_gaussian_position,
                lambda_gaussian_scale=lambda_gaussian_scale,
                lambda_raw_gaussian_position=lambda_raw_gaussian_position,
                lambda_raw_gaussian_scale=lambda_raw_gaussian_scale,
                lambda_raw_gaussian_rotation=lambda_raw_gaussian_rotation,
                lambda_raw_gaussian_color=lambda_raw_gaussian_color,
                lambda_raw_gaussian_opacity=lambda_raw_gaussian_opacity,
                lambda_learnable_template_offsets=lambda_learnable_template_offsets,
                lambda_tv_learnable_template_offsets=lambda_tv_learnable_template_offsets,
                lambda_tv_uv_rendering=lambda_tv_uv_rendering,
                tv_uv_include_transparent_gaussians=tv_uv_include_transparent_gaussians,
                lambda_beta_loss=lambda_beta_loss,

                reg_gaussian_position_above=0.1,
                reg_gaussian_position_below=-0.1,
                use_l1_scale_reg=use_l1_scale_reg,
                lambda_raw_scale_std=lambda_raw_scale_std,

                blur_masks=blur_masks,
                aug=aug,
            ),
            batch_size=batch
        )

    # ----------------------------------------------------------
    # Train Setup
    # ----------------------------------------------------------

    if generator_type == 'gaussians':
        group_name = 'gghead'
    elif generator_type == 'triplanes':
        group_name = 'eg3d'
    else:
        raise ValueError()

    train_setup = GGHeadTrainSetup(
        group_name=group_name,
        accumulate_metrics=64,
        metrics=metrics.split(',') if metrics != '' else [],
        gpus=opts.gpus,
        slurm_jobid=int(os.environ['SLURM_JOBID']) if 'SLURM_JOBID' in os.environ else None,
        slurm_nodelist=os.environ['SLURM_NODELIST'] if 'SLURM_NODELIST' in os.environ else None,
        resume_run=resume_run,
        resume_checkpoint=resume_checkpoint,
        reset_cur_nimg=reset_cur_nimg,
        total_kimg=kimg
    )

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.image_snapshot_ticks = image_snapshot_ticks
    c.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # ----------------------------------------------------------
    # Register evaluation metrics
    # ----------------------------------------------------------
    register_metric(fid100)
    register_metric(fid1k)
    register_metric(fid50k_full)

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < model_config.discriminator_config.epilogue_config.mbstd_group_size:
        raise ValueError('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in train_setup.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32

    if generator_type == 'triplanes' and resolution == neural_rendering_resolution:
        print("[Warning] Training EG3D with superresolution turned off!")
        sr_module = 'ggh.models.superresolution.SuperresolutionDummy'
    elif dataset_config.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif dataset_config.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif dataset_config.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert use_gaussians or False, f"Unsupported resolution {dataset_config.resolution}; make a new superresolution module"

    if opts.sr_module != None:
        sr_module = opts.sr_module

    if use_gaussians:
        rendering_options = {
            'c_gen_conditioning_zero': not gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
            'c_scale': opts.c_scale,  # mutliplier for generator pose conditioning label
        }
    else:
        rendering_options = {
            'image_resolution': dataset_config.resolution,
            'disparity_space_sampling': False,
            'clamp_mode': 'softplus',
            'superresolution_module': sr_module,
            'c_gen_conditioning_zero': not gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
            'gpc_reg_prob': gpc_reg_prob if gen_pose_cond else None,
            'c_scale': opts.c_scale,  # mutliplier for generator pose conditioning label
            'superresolution_noise_mode': opts.sr_noise_mode,  # [random or none], whether to inject pixel noise into super-resolution layers
            'density_reg': opts.density_reg,  # strength of density regularization
            'density_reg_p_dist': opts.density_reg_p_dist,  # distance at which to sample perturbed points for density regularization
            'reg_type': opts.reg_type,  # for experimenting with variations on density regularization
            'decoder_lr_mul': opts.decoder_lr_mul,  # learning rate multiplier for decoder
            'sr_antialias': True,
            'white_back': use_masks
        }

    if opts.cfg == 'ffhq':
        if use_gaussians:
            rendering_options.update({
                'box_warp': 1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            })
        else:
            rendering_options.update({
                'depth_resolution': 48,  # number of uniform samples to take per ray.
                'depth_resolution_importance': 48,  # number of importance samples to take per ray.
                'ray_start': 2.25,  # near point along each ray to start taking samples.
                'ray_end': 3.3,  # far point along each ray to stop taking samples.
                'box_warp': 1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
                'avg_camera_radius': 2.7,  # used only in the visualizer to specify camera orbit radius.
                'avg_camera_pivot': [0, 0, 0.2],  # used only in the visualizer to control center of camera rotation.
            })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    else:
        assert False, "Need to specify config"

    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only')

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
                                           brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    # NB: This is NOT the description that will be used by our model managers!
    # dataset_name = dataset_config.get_eg3d_name()
    # desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{optimizer_config.loss_config.r1_gamma:g}-res{resolution}'
    # if opts.desc is not None:
    #     desc += f'-{opts.desc}'

    experiment_config = GGHeadExperimentConfig(
        model_config=model_config,
        dataset_config=dataset_config,
        optimizer_config=optimizer_config,
        train_setup=train_setup)

    # Launch.
    launch_training(experiment_config, c=c, dry_run=opts.dry_run, name=name)


if __name__ == '__main__':
    tyro.cli(main)
