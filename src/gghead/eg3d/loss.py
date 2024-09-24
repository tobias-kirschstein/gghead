from dataclasses import dataclass
from typing import Optional, Literal, Dict

import numpy as np
import torch
from eg3d.torch_utils import training_stats
from eg3d.torch_utils.ops import conv2d_gradfix
from eg3d.torch_utils.ops import upfirdn2d
from eg3d.training.dual_discriminator import filtered_resizing
from eg3d.training.loss import StyleGAN2Loss
from elias.config import Config, implicit

from gghead.config.gaussian_attribute import GaussianAttribute
from gghead.models.gghead_model import GGHeadModel
from gghead.util.logging import LoggerBundle


@dataclass
class GGHStyleGAN2LossConfig(Config):
    r1_gamma: float = 10
    style_mixing_prob: float = 0
    pl_weight: float = 0
    pl_batch_shrink: float = 2
    pl_decay: float = 0.01
    pl_no_weight_grad: bool = False
    blur_init_sigma: int = 0
    blur_fade_kimg: int = 0
    r1_gamma_init: float = 0
    r1_gamma_fade_kimg: int = 0
    neural_rendering_resolution_initial: int = 64
    neural_rendering_resolution_final: Optional[int] = None
    neural_rendering_resolution_fade_kimg: int = 0
    gpc_reg_fade_kimg: int = 1000
    gpc_reg_prob: Optional[float] = None
    dual_discrimination: bool = False
    filter_mode: Literal['antialiased', 'classic', 'none'] = 'antialiased'
    aug: Optional[Literal['noaug', 'ada', 'fixed']] = 'noaug'
    ada_target: Optional[float] = 0.6

    # Discriminator resizing
    effective_res_disc: float = 1  # If < 1, images will be downscaled and upscaled again before fed to the discriminator

    # Progressive Discriminator growing
    new_layers_disc_start_kimg: Optional[int] = None
    new_layers_disc_blend_kimg: int = 1000
    new_layers_gen_start_kimg: Optional[int] = None
    new_layers_gen_blend_kimg: int = 1000
    plane_resolution_start_kimg: Optional[int] = None
    plane_resolution_blend_kimg: int = 1000
    effective_res_disc_start_kimg: Optional[int] = None
    effective_res_disc_blend_kimg: int = 1000

    # Gaussian regularization
    lambda_gaussian_position: float = 0
    lambda_gaussian_scale: float = 0
    reg_gaussian_position_above: float = 0
    reg_gaussian_position_below: float = 0
    reg_gaussian_scale_above: float = 0
    reg_gaussian_scale_below: float = 0
    reg_raw_gaussian_scale_above: float = 0
    reg_raw_gaussian_scale_below: float = 0
    use_l1_scale_reg: bool = False
    lambda_raw_gaussian_position: float = 0
    lambda_raw_gaussian_scale: float = 0
    lambda_raw_scale_std: float = 0
    lambda_raw_gaussian_rotation: float = 0
    lambda_raw_gaussian_color: float = 0
    lambda_raw_gaussian_opacity: float = 0
    lambda_learnable_template_offsets: float = 0
    lambda_tv_learnable_template_offsets: float = 0
    lambda_tv_uv_rendering: float = 0
    tv_uv_include_transparent_gaussians: bool = False  # Whether, to apply the UV TV loss on a UV rendering that also includes transparent gaussians
    lambda_auxiliary_opacity: float = 0
    lambda_beta_loss: float = 0

    use_gaussian_maintenance: bool = implicit(False)
    pretrained_resolution: Optional[int] = implicit()

    # Masks
    blur_masks: bool = True
    mask_swap_prob: float = 0
    r1_gamma_mask: Optional[float] = None

    # GSM
    decode_first: str = 'all'
    reg_weight: float = 0.1
    opacity_reg: float = 1
    l1_loss_reg: bool = True
    clamp: bool = False
    ref_scale: float = -5
    progressive_scale_reg_kimg: int = 0
    progressive_scale_reg_end: float = 0.01

    def requires_raw_gaussian_attributes(self) -> bool:
        return (self.lambda_raw_gaussian_position > 0
                or self.lambda_raw_gaussian_scale > 0
                or self.lambda_raw_gaussian_rotation > 0
                or self.lambda_raw_gaussian_color > 0
                or self.lambda_raw_gaussian_opacity > 0
                or self.lambda_raw_scale_std > 0)


class GGHStyleGAN2Loss(StyleGAN2Loss):
    def __init__(self,
                 device,
                 G,
                 D,
                 augment_pipe=None,
                 config: GGHStyleGAN2LossConfig = GGHStyleGAN2LossConfig(),
                 logger_bundle: LoggerBundle = LoggerBundle()) -> None:
        self._config = config
        self._logger_bundle = logger_bundle
        super().__init__(device, G, D, augment_pipe=augment_pipe,
                         r1_gamma=config.r1_gamma,
                         style_mixing_prob=config.style_mixing_prob,
                         pl_weight=config.pl_weight,
                         pl_batch_shrink=config.pl_batch_shrink,
                         pl_decay=config.pl_decay,
                         pl_no_weight_grad=config.pl_no_weight_grad,
                         blur_init_sigma=config.blur_init_sigma,
                         blur_fade_kimg=config.blur_fade_kimg,
                         r1_gamma_init=config.r1_gamma_init,
                         r1_gamma_fade_kimg=config.r1_gamma_fade_kimg,
                         neural_rendering_resolution_initial=config.neural_rendering_resolution_initial,
                         neural_rendering_resolution_final=config.neural_rendering_resolution_final,
                         neural_rendering_resolution_fade_kimg=config.neural_rendering_resolution_fade_kimg,
                         gpc_reg_fade_kimg=config.gpc_reg_fade_kimg,
                         gpc_reg_prob=config.gpc_reg_prob,
                         dual_discrimination=config.dual_discrimination)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, **synthesis_kwargs):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, **synthesis_kwargs)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, alpha_new_layers_disc: Optional[float] = None, update_emas=False,
              effective_res_disc: Optional[int] = None, other_img: Optional[Dict[str, torch.Tensor]] = None):
        blur_size = np.floor(blur_sigma * 3)

        if self._config.mask_swap_prob > 0 and other_img is not None:
            idx_self_other = torch.rand((img['image'].shape[0], 1, 1), device=img['image'].device) > self._config.mask_swap_prob
            img['image'][:, 3] = torch.where(idx_self_other, img['image'][:, 3], other_img['image'][:, 3].detach())

        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                if self._config.blur_masks:
                    img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
                else:
                    img['image'] = torch.cat([upfirdn2d.filter2d(img['image'][:, :3], f / f.sum()), img['image'][:, 3:]], dim=1)

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear',
                                                                                          antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear',
                                                               antialias=True)

        if effective_res_disc is not None:
            image = img['image']
            original_size = image.shape[-1]
            image_low = filtered_resizing(image, size=effective_res_disc, f=self.resample_filter, filter_mode=self.filter_mode)
            image_high = filtered_resizing(image_low, size=original_size, f=self.resample_filter)
            img['image'] = image_high

        if alpha_new_layers_disc is None:
            logits = self.D(img, c, update_emas=update_emas)
        else:
            logits = self.D(img, c, update_emas=update_emas, alpha_new_layers=alpha_new_layers_disc)
        return logits

    def loss_clamp_l2(self, source, target, mask=None, clamp=True):
        """
        Args:
            source: (bs, sh, h, w, c)
            target: float value
            mask: (bs, 1, h, w)
        Returns:
            float
        """
        if clamp:
            loss_map = torch.clamp((source - target), min=0) ** 2
        else:
            loss_map = (source - target) ** 2
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(
                source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def loss_clamp_l1(self, source, target_value, mask=None, clamp=False):
        """
        Args:
            source: (bs, sh, h, w, c)
            target_value: float value
            mask: (1, h, w)
        Returns:
            loss: float
        """
        if clamp:
            loss_map = torch.abs(torch.clamp(source - target_value, min=0))
        else:
            loss_map = torch.abs(source - target_value)
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(
                source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if not hasattr(self.G, 'rendering_kwargs') or not isinstance(self.G.rendering_kwargs, dict) or self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        alpha_new_layers_disc = min((cur_nimg - self._config.new_layers_disc_start_kimg * 1e3) / (self._config.new_layers_disc_blend_kimg * 1e3),
                                    1) if self._config.new_layers_disc_start_kimg is not None else None
        alpha_new_layers_gen = min((cur_nimg - self._config.new_layers_gen_start_kimg * 1e3) / (self._config.new_layers_gen_blend_kimg * 1e3),
                                    1) if self._config.new_layers_gen_start_kimg is not None else None
        alpha_plane_resolution = min((cur_nimg - self._config.plane_resolution_start_kimg * 1e3) / (self._config.plane_resolution_blend_kimg * 1e3),
                                   1) if self._config.plane_resolution_start_kimg is not None else None
        effective_res_disc = None
        if self._config.effective_res_disc_start_kimg is not None:
            alpha_effective_res_disc = min((cur_nimg - self._config.effective_res_disc_start_kimg * 1e3) / (self._config.effective_res_disc_blend_kimg * 1e3),
                                        1)
            pretrained_resolution = self._config.pretrained_resolution
            new_resolution = real_img.shape[2]
            effective_res_disc = int(alpha_effective_res_disc * new_resolution + (1 - alpha_effective_res_disc) * pretrained_resolution)

        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        if self._config.progressive_scale_reg_kimg > 0:
            reg_weight_cur = self._config.reg_weight - min(cur_nimg / (self._config.progressive_scale_reg_kimg * 1e3), 1) * (
                    self._config.reg_weight - self._config.progressive_scale_reg_end)
        else:
            reg_weight_cur = self._config.reg_weight

        real_img = {'image': real_img, 'image_raw': real_img_raw}
        gen_img = None

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Proper adversarial loss
                if isinstance(self.G, GGHeadModel):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution,
                                                  return_raw_attributes=self._config.requires_raw_gaussian_attributes(),
                                                  return_gaussian_maintenance=self._config.use_gaussian_maintenance,
                                                  return_auxiliary_attributes=self._config.lambda_auxiliary_opacity > 0,
                                                  return_uv_map=self._config.use_gaussian_maintenance,
                                                  alpha_new_layers=alpha_new_layers_gen,
                                                  alpha_plane_resolution=alpha_plane_resolution)
                else:
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc, effective_res_disc=effective_res_disc,
                                        other_img=real_img)

                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                # TODO: Remove!!!!!!!!!
                # L2 Reconstruction loss
                # gen_img, _gen_ws = self.run_G(gen_z, real_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                # loss_Gmain = (gen_img['image'] - real_img['image']).square().mean()
                if loss_Gmain.isnan().any():
                    print("loss_Gmain IS NAN!")
                # print(f"loss_G: {loss_Gmain.mean().item():0.3f}")
                # training_stats.report('Loss/G/loss', loss_Gmain)

                self._logger_bundle.log_metrics({
                    'Loss/scores/fake': gen_logits,
                    'Loss/signs/fake': gen_logits.sign(),
                    'Loss/G/loss': loss_Gmain
                }, step=cur_nimg)

                if isinstance(self.G, GGHeadModel):
                    loss_Gmain = loss_Gmain.squeeze(1)  # [B, 1] -> [B]
                    raw_gaussian_attributes = gen_img.gaussian_attribute_output.raw_gaussian_attributes
                    gaussian_attributes = gen_img.gaussian_attribute_output.gaussian_attributes

                    # Raw Gaussian Attributes (Before activation functions and adding offsets)
                    if self._config.lambda_raw_gaussian_position > 0:
                        reg_raw_gaussian_position = raw_gaussian_attributes[GaussianAttribute.POSITION].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_position': reg_raw_gaussian_position
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_position * reg_raw_gaussian_position
                    if self._config.lambda_raw_gaussian_scale > 0:
                        if self._config.reg_raw_gaussian_scale_above != 0 or self._config.reg_raw_gaussian_scale_below != 0:
                            raw_gaussian_scales = raw_gaussian_attributes[GaussianAttribute.SCALE]  # [B, G]
                            raw_gaussian_scales_to_regularize = torch.cat(
                                [raw_gaussian_scales[raw_gaussian_scales > self._config.reg_raw_gaussian_scale_above] - self._config.reg_raw_gaussian_scale_above,
                                 self._config.reg_raw_gaussian_scale_below - raw_gaussian_scales[raw_gaussian_scales < self._config.reg_raw_gaussian_scale_below]])
                            if len(raw_gaussian_scales_to_regularize) > 0:
                                if self._config.use_l1_scale_reg:
                                    reg_raw_gaussian_scale = raw_gaussian_scales_to_regularize.abs().mean()
                                else:
                                    reg_raw_gaussian_scale = raw_gaussian_scales_to_regularize.square().mean()

                                self._logger_bundle.log_metrics({
                                    'Loss/G/reg_raw_gaussian_scale': reg_raw_gaussian_scale
                                }, step=cur_nimg)
                                loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_scale * reg_raw_gaussian_scale
                        else:
                            if self._config.use_l1_scale_reg:
                                reg_raw_gaussian_scale = raw_gaussian_attributes[GaussianAttribute.SCALE].norm(dim=-1, p=1).mean(dim=1)  # [B]
                            else:
                                reg_raw_gaussian_scale = raw_gaussian_attributes[GaussianAttribute.SCALE].norm(dim=-1).mean(dim=1)  # [B]
                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_raw_gaussian_scale': reg_raw_gaussian_scale
                            }, step=cur_nimg)
                            loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_scale * reg_raw_gaussian_scale

                    if self._config.lambda_raw_scale_std > 0:
                        reg_raw_scale_std = raw_gaussian_attributes[GaussianAttribute.SCALE].std(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_scale_std': reg_raw_scale_std
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain - self._config.lambda_raw_scale_std * reg_raw_scale_std

                    if self._config.lambda_raw_gaussian_rotation > 0:
                        reg_raw_gaussian_rotation = raw_gaussian_attributes[GaussianAttribute.ROTATION].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_rotation': reg_raw_gaussian_rotation
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_rotation * reg_raw_gaussian_rotation
                    if self._config.lambda_raw_gaussian_color > 0:
                        reg_raw_gaussian_color = raw_gaussian_attributes[GaussianAttribute.COLOR].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_color': reg_raw_gaussian_color
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_color * reg_raw_gaussian_color
                    if self._config.lambda_raw_gaussian_opacity > 0:
                        reg_raw_gaussian_opacity = raw_gaussian_attributes[GaussianAttribute.OPACITY].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_opacity': reg_raw_gaussian_opacity
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_opacity * reg_raw_gaussian_opacity

                    # Learnable Template offsets
                    if self._config.lambda_learnable_template_offsets > 0:
                        reg_learnable_template_offsets = self.G._learnable_template_offsets.norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_learnable_template_offsets': reg_learnable_template_offsets
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_learnable_template_offsets * reg_learnable_template_offsets

                    if self._config.lambda_tv_learnable_template_offsets > 0:
                        reg_tv_learnable_template_offsets_y = (self.G._learnable_template_offsets[:, :, 1:] - self.G._learnable_template_offsets[:, :, :-1]).square().sum()
                        reg_tv_learnable_template_offsets_x = (self.G._learnable_template_offsets[:, :, :, 1:] - self.G._learnable_template_offsets[:, :, :, :-1]).square().sum()
                        reg_tv_learnable_template_offsets = reg_tv_learnable_template_offsets_x + reg_tv_learnable_template_offsets_y

                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_tv_learnable_template_offsets': reg_tv_learnable_template_offsets
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_tv_learnable_template_offsets * reg_tv_learnable_template_offsets

                    # Gaussian Attributes
                    if self._config.lambda_gaussian_position > 0:
                        gaussian_positions = self.G._apply_position_activation(raw_gaussian_attributes[GaussianAttribute.POSITION])

                        gaussian_positions_to_regularize = torch.cat(
                            [gaussian_positions[gaussian_positions > self._config.reg_gaussian_position_above] - self._config.reg_gaussian_position_above,
                             gaussian_positions[gaussian_positions < self._config.reg_gaussian_position_below] - self._config.reg_gaussian_position_below])
                        if len(gaussian_positions_to_regularize) > 0:
                            reg_gaussian_position = gaussian_positions_to_regularize.square().mean()
                            # reg_gaussian_position = gaussian_positions.norm(dim=-1).mean(dim=1)  # [B]
                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_gaussian_position': reg_gaussian_position
                            }, step=cur_nimg)
                            loss_Gmain = loss_Gmain + self._config.lambda_gaussian_position * reg_gaussian_position

                    if self._config.lambda_gaussian_scale > 0:
                        gaussian_scales = self.G._gaussian_model.scaling_activation(gaussian_attributes[GaussianAttribute.SCALE])
                        gaussian_scales_to_regularize = torch.cat(
                            [gaussian_scales[gaussian_scales > self._config.reg_gaussian_scale_above] - self._config.reg_gaussian_scale_above,
                             gaussian_scales[
                                 gaussian_scales < self._config.reg_gaussian_scale_below] - self._config.reg_gaussian_scale_below])
                        if len(gaussian_scales_to_regularize) > 0:
                            reg_gaussian_scale = torch.cat(
                                [gaussian_scales[gaussian_scales > self._config.reg_gaussian_scale_above] - self._config.reg_gaussian_scale_above,
                                 gaussian_scales[
                                     gaussian_scales < self._config.reg_gaussian_scale_below] - self._config.reg_gaussian_scale_below]).square().mean()

                            # reg_gaussian_scale = gaussian_attributes[GaussianAttribute.SCALE].norm(dim=-1).mean(dim=1)  # [B]
                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_gaussian_scale': reg_gaussian_scale
                            }, step=cur_nimg)
                            loss_Gmain = loss_Gmain + self._config.lambda_gaussian_scale * reg_gaussian_scale

                    if self._config.lambda_auxiliary_opacity > 0:
                        auxiliary_opacities = self.G._apply_opacity_activation(gen_img.gaussian_attribute_output.auxiliary_gaussian_attributes[GaussianAttribute.OPACITY])
                        reg_auxiliary_opacities = auxiliary_opacities.mean()
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_auxiliary_opacities': reg_auxiliary_opacities
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_auxiliary_opacity * reg_auxiliary_opacities

                    if self._config.lambda_beta_loss > 0:
                        opacities = self.G._apply_opacity_activation(gen_img.gaussian_attribute_output.gaussian_attributes[GaussianAttribute.OPACITY])
                        beta_loss = ((0.1 + opacities).log() + (1.1 - opacities).log() + 2.20727).mean()
                        self._logger_bundle.log_metrics({
                            'Loss/G/beta_loss': beta_loss
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_beta_loss * beta_loss

                    if self._config.lambda_tv_uv_rendering > 0:
                        uv_renderings = self.G.get_uv_rendering(gen_c, gen_img, include_transparent_gaussians=self._config.tv_uv_include_transparent_gaussians)
                        mask = 1 - (uv_renderings[:, [2]] + 1)/2

                        uv_renderings_no_blend = ((uv_renderings - (1 - mask)) / mask)  # Undo effect of alpha blending. Background pixels will be inf
                        background_mask = mask == 0

                        mask_y = (background_mask[:, :, 1:] | background_mask[:, :, :-1]).repeat(1, 3, 1, 1)
                        mask_x = (background_mask[:, :, :, 1:] | background_mask[:, :, :, :-1]).repeat(1, 3, 1, 1)

                        uv_difference_y = uv_renderings_no_blend[:, :, 1:] - uv_renderings_no_blend[:, :, :-1]
                        uv_difference_x = uv_renderings_no_blend[:, :, :, 1:] - uv_renderings_no_blend[:, :, :, :-1]

                        reg_tv_uv_rendering_y = uv_difference_y[~mask_y].abs().mean()
                        reg_tv_uv_rendering_x = uv_difference_x[~mask_x].abs().mean()
                        reg_tv_uv_rendering = reg_tv_uv_rendering_x + reg_tv_uv_rendering_y

                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_tv_uv_rendering': reg_tv_uv_rendering
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_tv_uv_rendering * reg_tv_uv_rendering


            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.G.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Gmain NAN GRADIENTS: {gradients_with_nan}")

            # Gaussian Maintenance
            if self._config.use_gaussian_maintenance:
                if isinstance(self.G, GGHeadModel):
                    opacities = self.G._apply_opacity_activation(gen_img.gaussian_attribute_output.gaussian_attributes[GaussianAttribute.OPACITY])[..., 0].detach()
                    # TODO: This assumes position_start_channel to be correct
                    uv_maps = gen_img.gaussian_attribute_output.uv_map.detach()
                    self.G.do_gaussian_maintenance(gen_img.viewspace_points, gen_img.visibility_filters, gen_img.radii, opacities, uv_maps, cur_nimg)

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                if isinstance(self.G, GGHeadModel):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution,
                                                  update_emas=True, alpha_new_layers=alpha_new_layers_gen, alpha_plane_resolution=alpha_plane_resolution)
                else:
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution,
                                                  update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, alpha_new_layers_disc=alpha_new_layers_disc,
                                        effective_res_disc=effective_res_disc, other_img=real_img)
                self._logger_bundle.log_metrics({
                    'Loss/scores/fake': gen_logits,
                    'Loss/signs/fake': gen_logits.sign()
                }, step=cur_nimg)
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                if loss_Dgen.isnan().any():
                    print("loss_Dgen IS NAN !")
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.D.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Dgen NAN GRADIENTS: {gradients_with_nan}")

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc, effective_res_disc=effective_res_disc,
                                         other_img=gen_img)
                self._logger_bundle.log_metrics({
                    'Loss/scores/real': real_logits,
                    'Loss/signs/real': real_logits.sign()
                }, step=cur_nimg)
                # training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    if loss_Dreal.isnan().any():
                        print("loss_Dreal IS NAN !")
                    self._logger_bundle.log_metrics({
                        'Loss/D/loss': loss_Dgen + loss_Dreal,
                    }, step=cur_nimg)
                    # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            # TODO: Is this used for Gaussian Discriminator? maybe dual_discrimination should be set to False
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']],
                                                           create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        if self._config.r1_gamma_mask is not None:
                            r1_grads_image[:, 3] *= np.sqrt(self._config.r1_gamma_mask)

                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    self._logger_bundle.log_metrics({
                        'Loss/r1_penalty': r1_penalty,
                        'Loss/D/reg': loss_Dr1
                    }, step=cur_nimg)
                    # training_stats.report('Loss/r1_penalty', r1_penalty)
                    # training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.D.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Dreal NAN GRADIENTS: {gradients_with_nan}")

# ----------------------------------------------------------------------------
