from typing import List, Optional, Union

import mediapy
import numpy as np
import torch
import tyro
from dreifus.image import normalized_torch_to_numpy_img
from dreifus.matrix import Intrinsics
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from eg3d.datamanager.nersemble import encode_camera_params
from elias.util import ensure_directory_exists_for_file
from elias.util.batch import batchify_sliced
from tqdm import tqdm

from gghead.model_manager.finder import find_model_folder
from gghead.models.gghead_model import GGHeadModel
from gghead.render_manager.interpolation_rendering_manager import InterpolationRenderingConfig, \
    InterpolationRenderingManager


def interpolate_codes(latent_codes: List[torch.Tensor], n_frames: int, loop: bool = False) -> List[torch.Tensor]:
    n_codes = len(latent_codes)
    interpolation_i_codes = list(range(n_codes))  # Just interpolate between first n codes
    if loop:
        interpolation_i_codes.append(0)
        n_codes += 1
    interpolation_target_frames = np.linspace(0, n_frames - 1, n_codes, dtype=int)
    # interpolation_target_frames:  0| 11| 22| 33|
    # interpolation_phase:           | 0 | 1 | 2 |
    #
    interpolated_codes = []
    for frame_id in range(n_frames):
        interpolation_phase = (frame_id > interpolation_target_frames).sum() - 1

        if frame_id == 0:
            i_code_1 = interpolation_i_codes[0]
            i_code_2 = interpolation_i_codes[0]
            alpha = 1
        else:
            frame_id_1 = interpolation_target_frames[interpolation_phase]
            frame_id_2 = interpolation_target_frames[interpolation_phase + 1]
            i_code_1 = interpolation_i_codes[interpolation_phase]
            i_code_2 = interpolation_i_codes[interpolation_phase + 1]

            alpha = (frame_id - frame_id_1) / (frame_id_2 - frame_id_1)

        code_1 = latent_codes[i_code_1]
        code_2 = latent_codes[i_code_2]

        interpolated_code = (1 - alpha) * code_1 + alpha * code_2
        interpolated_codes.append(interpolated_code)

    return interpolated_codes


# move_z
def render_interpolation(model: GGHeadModel,
                         identity_codes: List[torch.Tensor],
                         config: InterpolationRenderingConfig,
                         output_path: Optional[str] = None):
    # Create trajectory

    batch_size = 1
    move_z = 2.7 if config.move_z is None else config.move_z
    trajectory = circle_around_axis(config.n_frames, up=Vec3(0, 1, 0), move=Vec3(0, 0, move_z),
                                    distance=0.6 * move_z / 2.7,
                                    theta_to=2 * np.pi * config.n_circles)

    # EG3D uses a fixed intrinsics for all images
    intrinsics = Intrinsics(np.array([[4.2647, 0., 0.5],
                                      [0., 4.2647, 0.5],
                                      [0., 0., 1.]]))

    # Create interpolated codes and camera inputs
    interpolated_identity_codes = interpolate_codes(identity_codes, config.n_frames, loop=True)
    interpolated_identity_codes = torch.stack(interpolated_identity_codes)
    cs = []
    for pose in trajectory:
        c = encode_camera_params(pose, intrinsics)
        cs.append(torch.tensor(c, device='cuda'))
    cs = torch.stack(cs)

    # Forward
    predicted_images = []
    progress = tqdm(
        enumerate(zip(batchify_sliced(interpolated_identity_codes, batch_size), batchify_sliced(cs, batch_size))))
    for i_frame, (code_batch, c_batch) in progress:
        torch.cuda.empty_cache()
        c_batch_mapping = c_batch.clone()
        extrinsics = c_batch_mapping[..., :16].view(-1, 4, 4)
        t = extrinsics[:, :3, 3]
        radius = t.norm(dim=1)
        t /= (radius / 2.7)
        ws = model.mapping(code_batch, c_batch_mapping, truncation_psi=config.truncation_psi)
        gen_output = model.synthesis(ws, c_batch,
                                     neural_rendering_resolution=config.resolution,
                                     noise_mode='const')

        for image in gen_output['image']:
            image = normalized_torch_to_numpy_img(image)[..., :3]
            predicted_images.append(image)

    if output_path is not None:
        mediapy.write_video(output_path, predicted_images, fps=24)

    return predicted_images


def main(run_name: str,
         /,
         checkpoint: int = -1,
         load_ema: bool = True,
         seeds: Optional[Union[List[int], str]] = None,
         n_persons: int = 5,
         n_frames: int = 1000,
         resolution: int = 512,
         truncation_psi: float = 0.7):
    model_folder = find_model_folder(run_name)
    model_manager = model_folder.open_run(run_name)

    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    model = model_manager.load_checkpoint(checkpoint, load_ema=load_ema)
    model.cuda()
    model.eval()

    if seeds is None:
        seeds = list(range(n_persons))
    else:
        if isinstance(seeds, str):
            seed_start, seed_end = seeds.split('-')
            seeds = list(range(int(seed_start), int(seed_end) + 1))
        n_persons = len(seeds)

    identity_codes = []
    for identity_seed in seeds:
        rng = torch.Generator(device='cuda')
        rng.manual_seed(identity_seed)
        identity_code = torch.randn(model.z_dim, device='cuda', generator=rng)
        identity_codes.append(identity_code)

    render_config = InterpolationRenderingConfig(
        run_name=run_name,
        checkpoint=checkpoint,
        n_persons=n_persons,
        n_circles=n_persons,
        n_frames=n_frames,
        resolution=resolution,
        seeds=seeds,
        load_ema=load_ema,
        truncation_psi=truncation_psi,
    )

    render_manager = InterpolationRenderingManager(render_config)
    rendering_path = render_manager.get_rendering_path()
    ensure_directory_exists_for_file(rendering_path)

    model = model.eval().requires_grad_(False)

    with torch.no_grad():
        render_interpolation(model, identity_codes, render_config, output_path=rendering_path)


if __name__ == '__main__':
    tyro.cli(main)
