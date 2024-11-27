import copy
import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from datetime import timedelta
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import psutil
import torch
from dreifus.util.visualizer import ImageWindow
from eg3d import legacy, dnnlib
from eg3d.dnnlib import EasyDict
from eg3d.dnnlib.util import format_time, Logger
from eg3d.metrics.metric_main import calc_metric, report_metric, register_metric, fid100, fid1k, fid50k_full
from eg3d.torch_utils import training_stats, custom_ops
from eg3d.torch_utils.misc import InfiniteSampler, copy_params_and_buffers, print_module_summary, params_and_buffers, nan_to_num, constant, \
    check_ddp_consistency
from eg3d.torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from eg3d.training.augment import AugmentPipe
from eg3d.training.dual_discriminator import DualDiscriminator
from eg3d.training.training_loop import setup_snapshot_image_grid, save_image_grid
from eg3d.training.triplane import TriPlaneGenerator
from elias.util import ensure_directory_exists_for_file, ensure_directory_exists
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader

from gghead.config.gaussian_attribute import GaussianAttribute
from gghead.dataset.image_folder_dataset import GGHeadMaskImageFolderDataset
from gghead.eg3d.loss import GGHeadStyleGAN2Loss
from gghead.model_manager.base_model_manager import GGHeadEvaluationConfig, GGHeadEvaluationResult
from gghead.model_manager.finder import find_model_manager
from gghead.model_manager.gghead_model_manager import GGHeadExperimentConfig, GGHeadModelFolder
from gghead.models.gaussian_discriminator import GaussianDiscriminator
from gghead.models.gghead_model import GGHeadModel
from gghead.util.logging import LoggerBundle


# ----------------------------------------------------------------------------


def subprocess_fn(rank: int, experiment_config: GGHeadExperimentConfig, c, temp_dir, name: Optional[str] = None):
    num_gpus = experiment_config.train_setup.gpus

    # Init torch.distributed.
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus, timeout=timedelta(minutes=30))

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop(experiment_config, rank=rank, name=name, **c)


# ----------------------------------------------------------------------------

def launch_training(experiment_config: GGHeadExperimentConfig, c, dry_run, name: Optional[str] = None):
    Logger(should_flush=True)

    num_gpus = experiment_config.train_setup.gpus

    # Print options.
    print()
    print('Training options:')

    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o):
                return asdict(o)
            return super().default(o)

    print(json.dumps(c, indent=2, cls=EnhancedJSONEncoder))
    print()
    # print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {num_gpus}')
    print(f'Batch size:          {experiment_config.optimizer_config.batch_size} images')
    print(f'Training duration:   {experiment_config.train_setup.total_kimg} kimg')
    print(f'Dataset path:        {experiment_config.dataset_config.path}')
    print(f'Dataset size:        {experiment_config.dataset_config.max_size} images')
    print(f'Dataset resolution:  {experiment_config.dataset_config.resolution}')
    print(f'Dataset labels:      {experiment_config.dataset_config.use_labels}')
    print(f'Dataset x-flips:     {experiment_config.dataset_config.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Launch processes.
    print('Launching processes...')
    # set_start_method('spawn') is essential for the combination of DataLoader and zip Dataset to work
    # Otherwise, get random errors on Unix systems where fork is the default start method since parallel reading of zipfiles doesn't work there
    torch.multiprocessing.set_start_method('spawn')
    with TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            subprocess_fn(experiment_config=experiment_config, rank=0, c=c, temp_dir=temp_dir, name=name)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(experiment_config, c, temp_dir, name), nprocs=num_gpus)


# ----------------------------------------------------------------------------


def training_loop(
        # @formatter:off
        experiment_config: GGHeadExperimentConfig,
        run_dir                 = '.',      # Output directory.
        # training_set_kwargs     = {},       # Options for training set.
        data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
        G_kwargs                = {},       # Options for generator network.
        # D_kwargs                = {},       # Options for discriminator network.
        augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
        # loss_kwargs             = {},       # Options for loss function.
        # metrics                 = [],       # Metrics to evaluate during training.
        random_seed             = 0,        # Global random seed.
        num_gpus                = 1,        # Number of GPUs participating in the training.
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
        ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
        augment_p               = 0,        # Initial value of augmentation probability.
        ada_target              = None,     # ADA target value. None = fixed p.
        ada_interval            = 4,        # How often to perform ADA adjustment?
        ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
        kimg_per_tick           = 4,        # Progress snapshot interval.
        image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
        network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
        resume_pkl              = None,     # Network pickle to resume training from.
        resume_kimg             = 0,        # First kimg to report when resuming training.
        cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
        abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
        use_gaussians: bool     = False,
        use_vis_window: bool    = False,
        name: Optional[str]     = None,
        # @formatter:on
):
    dataset_config = experiment_config.dataset_config
    model_config = experiment_config.model_config

    # ----------------------------------------------------------
    # Create Model manager
    # ----------------------------------------------------------
    generator_type = experiment_config.model_config.generator_type
    if rank == 0:
        run_desc = (f'{experiment_config.dataset_config.get_eg3d_name():s}'
                    f'-gpus{experiment_config.train_setup.gpus:d}'
                    f'-batch{experiment_config.optimizer_config.batch_size:d}'
                    f'-gamma{experiment_config.optimizer_config.loss_config.r1_gamma:g}'
                    f'-res{model_config.generator_config.img_resolution}')
        if experiment_config.train_setup.resume_run is not None:
            run_desc += f'-resume{experiment_config.train_setup.resume_run}'
        if name is not None:
            run_desc += "_" + name
        model_manager = GGHeadModelFolder().new_run(run_desc)

        run_dir = model_manager.get_model_store_path()

        print("===============================================================")
        print(f"Start training {model_manager.get_run_name()}")
        print("===============================================================")

    Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    if rank == 0:
        ensure_directory_exists(model_manager.get_wandb_folder())
        wandb_logger = WandbLogger(
            project=experiment_config.train_setup.project_name,
            group=experiment_config.train_setup.group_name,
            name=model_manager.get_run_name(),
            config=experiment_config.to_json(),
            save_dir=model_manager.get_wandb_folder())
        experiment_config.train_setup.wandb_run_id = wandb_logger.experiment._run_id
        logger_bundle = LoggerBundle([wandb_logger], accumulate=experiment_config.train_setup.accumulate_metrics)
    else:
        # Other processes should not log anything
        logger_bundle = LoggerBundle()

    # ----------------------------------------------------------
    # Initialize
    # ----------------------------------------------------------
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = False  # Avoids errors with the augmentation pipe.

    # ----------------------------------------------------------
    # Load training set
    # ----------------------------------------------------------
    if rank == 0:
        print('Loading training set...')
    training_set = GGHeadMaskImageFolderDataset(dataset_config)
    eval_set = GGHeadMaskImageFolderDataset(dataset_config.eval())
    training_set_sampler = InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # ----------------------------------------------------------
    # Construct Networks
    # ----------------------------------------------------------
    if rank == 0:
        print('Constructing networks...')
    # common_kwargs = dict(img_channels=training_set.num_channels)
    generator_config = model_config.generator_config
    do_resume = (experiment_config.train_setup.resume_run is not None) and (rank == 0)

    if generator_type == 'gaussians':
        G = GGHeadModel(generator_config, logger_bundle, post_init=not do_resume).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    elif generator_type == 'triplanes':
        G = TriPlaneGenerator(
            z_dim=generator_config.z_dim,
            c_dim=generator_config.c_dim,
            w_dim=generator_config.w_dim,
            img_resolution=generator_config.img_resolution,
            neural_rendering_resolution=generator_config.neural_rendering_resolution,
            img_channels=training_set.num_channels,
            mapping_kwargs=asdict(generator_config.mapping_network_config),
            **asdict(generator_config.synthesis_network_config),
            **G_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    else:
        raise ValueError(f"Unkown generator type: {generator_type}")

    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    discriminator_config = model_config.discriminator_config
    if use_gaussians or not discriminator_config.use_dual_discrimination:
        D = GaussianDiscriminator(discriminator_config).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    else:
        D = DualDiscriminator(
            c_dim=discriminator_config.c_dim,
            img_resolution=discriminator_config.img_resolution,
            img_channels=discriminator_config.img_channels,
            architecture=discriminator_config.architecture,
            channel_base=discriminator_config.channel_base,
            channel_max=discriminator_config.channel_max,
            num_fp16_res=discriminator_config.num_fp16_res,
            conv_clamp=discriminator_config.conv_clamp,
            cmap_dim=discriminator_config.cmap_dim,
            disc_c_noise=discriminator_config.disc_c_noise,
            block_kwargs=asdict(discriminator_config.block_config),
            mapping_kwargs=asdict(discriminator_config.mapping_network_config),
            epilogue_kwargs=asdict(discriminator_config.epilogue_config)).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # ----------------------------------------------------------
    # Resume from existing pickle
    # ----------------------------------------------------------
    if do_resume:
        resume_run = experiment_config.train_setup.resume_run
        checkpoint = experiment_config.train_setup.resume_checkpoint

        print(f'Resuming from {resume_run} - checkpoint {checkpoint}')
        model_manager_loaded = find_model_manager(resume_run)
        G_loaded = model_manager_loaded.load_checkpoint(checkpoint)
        G_ema_loaded = model_manager_loaded.load_checkpoint(checkpoint, load_ema=True)
        D_loaded = model_manager_loaded.load_discriminator(checkpoint)

        def copy_params(src_module: torch.nn.Module, dst_module: torch.nn.Module, require_all: bool = False):
            important_buffer_names = [  # "_uv_grid", "_flame_vertices",
                "_maintenance_pos_gradients", "_maintenance_gaussian_counts", "_maintenance_max_opacities",
                "_maintenance_position_maps", "_maintenance_average_position_map", "_maintenance_position_map_counts"]
            buffer_names = [k for k in dict(src_module.named_buffers()).keys() if k not in important_buffer_names]
            if buffer_names:
                print(f"source module {type(src_module)} has buffers {buffer_names} which won't be loaded'")

            important_src_buffers = [(k, p) for k, p in src_module.named_buffers() if k in important_buffer_names]
            important_dest_buffers = [(k, p) for k, p in dst_module.named_buffers() if k in important_buffer_names]
            src_tensors = dict(src_module.named_parameters())
            src_tensors.update(important_src_buffers)
            for name, tensor in chain(dst_module.named_parameters(), important_dest_buffers):
                assert (name in src_tensors) or (not require_all)
                if name in src_tensors:
                    try:
                        tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                    except RuntimeError as e:
                        if not isinstance(src_module, GGHeadModel) and not 'torgb' in name:
                            raise e

                        # Assume, there is a shape mismatched because super-resolution module was added to a pre-trained model
                        target = torch.zeros_like(tensor)
                        cloned_src = src_tensors[name].detach().clone()
                        c_dst = 0
                        c_src = 0
                        n_channels_total = sum([att.get_n_channels(dst_module._config.gaussian_attribute_config) for att in dst_module._config.uv_attributes])
                        n_channels_exclude_color = n_channels_total - GaussianAttribute.COLOR.get_n_channels(dst_module._config.gaussian_attribute_config)
                        for attr in src_module._config.uv_attributes:
                            dim_channel = 0
                            n_channels_src = attr.get_n_channels(src_module._config.gaussian_attribute_config)
                            n_channels_dst = attr.get_n_channels(dst_module._config.gaussian_attribute_config)
                            if attr == GaussianAttribute.COLOR:
                                n_color_channels_src = src_module._config.gaussian_attribute_config.n_color_channels
                                n_color_channels_dst = dst_module._config.gaussian_attribute_config.n_color_channels
                                n_sh_dims = n_channels_src // n_color_channels_src
                                color_tensor_src = cloned_src[c_src: c_src + n_channels_src]
                                src_shape = color_tensor_src.shape
                                color_tensor_src = color_tensor_src.view(n_sh_dims, n_color_channels_src, *src_shape[1:])
                                zeros_tensor_src = torch.zeros((n_sh_dims, n_color_channels_dst - n_color_channels_src, *src_shape[1:]),
                                                               dtype=color_tensor_src.dtype, device=color_tensor_src.device)
                                torch.nn.init.normal_(zeros_tensor_src)
                                color_tensor_src = torch.cat([color_tensor_src, zeros_tensor_src], dim=1)
                                color_tensor_src = color_tensor_src.reshape(n_channels_dst, *src_shape[1:])
                                target[c_dst: c_dst + n_channels_dst] = color_tensor_src
                            else:
                                target[c_dst: c_dst + n_channels_src] = cloned_src[c_src: c_src + n_channels_src]  # TODO: Use dim_channel
                            c_src += n_channels_src
                            c_dst += n_channels_dst

                        print(f'Merging loaded tensor {cloned_src.shape} into model tensor {target.shape} for key {name}')
                        tensor.copy_(target).requires_grad_(tensor.requires_grad)

        copy_params(G_loaded, G, require_all=False)
        copy_params(G_ema_loaded, G_ema, require_all=False)
        copy_params(D_loaded, D, require_all=False)
        resume_kimg = checkpoint
        if not experiment_config.train_setup.reset_cur_nimg:
            logger_bundle.set_step(resume_kimg * 1000)

    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            copy_params_and_buffers(resume_data[name], module, require_all=False)

    # ----------------------------------------------------------
    # Print network summary tables
    # ----------------------------------------------------------
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        if use_gaussians:
            c = torch.tensor([1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1,
                              1, 0, 0,
                              0, 1, 0,
                              0, 0, 1], device=device, dtype=torch.float32).unsqueeze(0).repeat((batch_gpu, 1))
        else:
            c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = print_module_summary(G, [z, c])
        print_module_summary(D, [img, c])

    # ----------------------------------------------------------
    # Setup Augmentation
    # ----------------------------------------------------------
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = AugmentPipe(**augment_kwargs).train().requires_grad_(False).to(device)
        # augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # ----------------------------------------------------------
    # Distribute across GPUs
    # ----------------------------------------------------------
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    if do_resume and hasattr(G, 'post_init'):
        G.post_init()
        G_ema.post_init()

    # ----------------------------------------------------------
    # Setup training phases
    # ----------------------------------------------------------
    if rank == 0:
        print('Setting up training phases...')

    loss = GGHeadStyleGAN2Loss(device=device, G=G, D=D, augment_pipe=augment_pipe,
                               config=experiment_config.optimizer_config.loss_config,
                               logger_bundle=logger_bundle)
    # loss = StyleGAN2Loss(device=device, G=G, D=D, augment_pipe=augment_pipe,
    #                      **asdict(experiment_config.optimizer_config.loss_config))  # subclass of training.loss.Loss
    phases = []
    generator_optimizer_config = experiment_config.optimizer_config.generator_optimizer_config
    discriminator_optimizer_config = experiment_config.optimizer_config.discriminator_optimizer_config
    for name, module, opt_config, reg_interval in [('G', G, generator_optimizer_config, G_reg_interval),
                                                   ('D', D, discriminator_optimizer_config, D_reg_interval)]:
        if reg_interval is None:
            if name == 'G' and experiment_config.optimizer_config.separate_lr_template_offsets is not None:
                params = [
                    {'params': module.named_parameters(),
                     'lr': opt_config.lr,
                     'betas': (opt_config.beta1, opt_config.beta2),
                     'eps': opt_config.eps}
                ]
                opt = Adam(params=params)
            else:
                opt = Adam(params=module.parameters(), lr=opt_config.lr, betas=(opt_config.beta1, opt_config.beta2),
                           eps=opt_config.eps)  # subclass of torch.optim.Optimizer
            for _ in range(opt_config.n_phases):
                phases += [EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt = Adam(module.parameters(),
                       lr=opt_config.lr * mb_ratio,
                       betas=(opt_config.beta1 ** mb_ratio, opt_config.beta2 ** mb_ratio),
                       eps=opt_config.eps)
            for _ in range(opt_config.n_phases):
                phases += [EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # ----------------------------------------------------------
    # Export sample images
    # ----------------------------------------------------------
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=eval_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    # ----------------------------------------------------------
    # Setup Viewer
    # ----------------------------------------------------------
    if use_vis_window:
        vis_img_buffer = np.zeros((generator_config.img_resolution, generator_config.img_resolution, 3), dtype=np.float32)
        image_window = ImageWindow(vis_img_buffer)
        z_valid = torch.randn((1, model_config.generator_config.z_dim), device=device)

    # ----------------------------------------------------------
    # Register evaluation metrics
    # ----------------------------------------------------------
    register_metric(fid100)
    register_metric(fid1k)
    register_metric(fid50k_full)

    # ----------------------------------------------------------
    # Store configs
    # ----------------------------------------------------------
    if rank == 0:
        model_manager.store_model_config(experiment_config.model_config)
        model_manager.store_dataset_config(dataset_config)
        model_manager.store_optimization_config(experiment_config.optimizer_config)
        model_manager.store_train_setup(experiment_config.train_setup)

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if experiment_config.train_setup.reset_cur_nimg:
        cur_nimg = 0
    else:
        cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    profiler = torch.autograd.profiler.profile(with_stack=True, profile_memory=True)
    profile_batch_idx = 10

    while True:
        if batch_idx == profile_batch_idx:
            profiler.__enter__()

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            if experiment_config.optimizer_config.freeze_generator and phase.name in ['Gmain', 'Gboth']:
                for k, p in phase.module.named_parameters():
                    if "super_resolution" not in k:
                        p.requires_grad_(False)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, (k, b) in zip(G_ema.buffers(), G.named_buffers()):
                if b.shape != b_ema.shape and ('flame_vertices' in k or 'uv_grid' in k or 'maintenance_' in k):
                    setattr(G_ema, k, b.clone())
                else:
                    b_ema.copy_(b)
            if isinstance(G_ema, TriPlaneGenerator):
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Profiler
        if batch_idx == profile_batch_idx:
            profiler.__exit__(*sys.exc_info())
            print(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        logger_bundle.log_metrics({
            'Progress/n_samples_seen': cur_nimg,
            'Progress/n_batches_seen': batch_idx
        }, step=cur_nimg)

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(constant(0, device=device)))

        if use_vis_window:
            with torch.no_grad():
                c_valid = torch.tensor(training_set.get_label(0), device=device).unsqueeze(0)
                rendering_dict = G.forward(z_valid, c_valid)
                vis_img_buffer[:] = (rendering_dict['image'][0].permute(1, 2, 0).cpu().numpy()[..., :3] + 1) / 2

        if hasattr(G, '_cnn_adaptor'):
            G._cnn_adaptor.progressive_update(cur_nimg / 1000)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # ----------------------------------------------------------
        # IMPORTANT: EVERYTHING BELOW HERE IS ONLY EXECUTED ONCE "PER TICK"
        # ----------------------------------------------------------

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()

        if rank == 0:
            fields = []
            fields += [f"tick {cur_tick:<5d}"]
            fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
            fields += [f"time {format_time(tick_end_time - start_time):<12s}"]
            fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
            fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
            fields += [f"maintenance {maintenance_time:<6.1f}"]
            fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2 ** 30:<6.2f}"]
            fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2 ** 30:<6.2f}"]
            fields += [f"reserved {torch.cuda.max_memory_reserved(device) / 2 ** 30:<6.2f}"]
            fields += [f"augment {float(augment_pipe.p.cpu()) if augment_pipe is not None else 0:.3f}"]
            print(' '.join(fields))

            logger_bundle.log_metrics({
                'Progress/tick': cur_tick,
                'Progress/kimg': cur_nimg / 1e3,
                'Timing/total_sec': tick_end_time - start_time,
                'Timing/sec_per_tick': tick_end_time - tick_start_time,
                'Timing/sec_per_kimg': (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3,
                'Timing/maintenance_sec': maintenance_time,
                'Resources/cpu_mem_gb': psutil.Process(os.getpid()).memory_info().rss / 2 ** 30,
                'Resources/peak_gpu_mem_gb': torch.cuda.max_memory_allocated(device) / 2 ** 30,
                'Resources/peak_gpu_mem_reserved_gb': torch.cuda.max_memory_reserved(device) / 2 ** 30,
                'Progress/augment': float(augment_pipe.p.cpu()) if augment_pipe is not None else 0,
                'Timing/total_hours': (tick_end_time - start_time) / (60 * 60),
                'Timing/total_days': (tick_end_time - start_time) / (24 * 60 * 60)
            },
                step=cur_nimg)

        torch.cuda.reset_peak_memory_stats()

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
                images = torch.cat([o['image'].cpu() for o in out]).numpy()
                images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
                images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1], grid_size=grid_size)
                save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_raw.png'), drange=[-1, 1], grid_size=grid_size)
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()],
                                grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=asdict(dataset_config))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    if hasattr(module, '_logger_bundle'):
                        module._logger_bundle = None  # Don't persist wandb loggers. Otherwise, can get error during unpickling
                snapshot_data[name] = module
                del module  # conserve memory

            if rank == 0:
                if use_gaussians:
                    checkpoint_name = model_manager._checkpoints_folder.substitute(model_manager._checkpoint_name_format, cur_nimg // 1000)
                    snapshot_pkl = f"{model_manager._checkpoints_folder.get_location()}/{checkpoint_name}"
                    ensure_directory_exists_for_file(snapshot_pkl)
                else:
                    snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')

                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        metrics = experiment_config.train_setup.metrics
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
                evaluation_results = dict()
                for metric in metrics:
                    result_dict = calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                              dataset_kwargs=dataset_config.get_eval_dict(), num_gpus=1, rank=rank,
                                              device=device)
                    report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)

                    evaluation_results[metric] = result_dict.results[metric]

                evaluation_config = GGHeadEvaluationConfig(checkpoint=cur_nimg // 1000, load_ema=True)
                evaluation_result = GGHeadEvaluationResult(**evaluation_results)
                model_manager.store_evaluation_result(evaluation_config, evaluation_result)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
                logger_bundle.log_metrics({
                    'Timing/' + phase.name: value
                }, step=cur_nimg)
            # training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        for name, value in stats_dict.items():
            logger_bundle.log_metrics({
                name: value.mean,
                'Progress/n_samples_seen': cur_nimg,
                'Progress/n_batches_seen': batch_idx
            }, step=cur_nimg)
        for name, value in stats_metrics.items():
            logger_bundle.log_metrics({
                f'Metrics/{name}': value,
                'Progress/n_samples_seen': cur_nimg,
                'Progress/n_batches_seen': batch_idx
            }, step=cur_nimg)

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
