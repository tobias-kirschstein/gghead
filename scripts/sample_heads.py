from typing import Optional

import mediapy
import numpy as np
import torch
import tyro
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from eg3d.datamanager.nersemble import encode_camera_params, decode_camera_params
from elias.util import ensure_directory_exists
from elias.util.batch import batchify_sliced
from tqdm import tqdm

from gghead.constants import DEFAULT_INTRINSICS
from gghead.env import GGH_RENDERINGS_PATH
from gghead.model_manager.finder import find_model_manager


def main(run_name: str,
         /,
         checkpoint: int = -1,
         n_seeds: int = 8,  # If no seeds specified, generate heads for seeds 0-n
         seeds: Optional[str] = None,  # comma-separated list of seeds to sample from
         truncation_psi: float = 0.7,
         batch_size: int = 8,
         resolution: int = 512):
    device = torch.device('cuda')

    model_manager = find_model_manager(run_name)
    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)

    move_z = 2.7
    poses = circle_around_axis(96, up=Vec3(0, 1, 0), move=Vec3(0, 0, move_z), distance=0.6 * move_z / 2.7,
                               theta_to=2 * np.pi)

    cs = [encode_camera_params(pose, DEFAULT_INTRINSICS) for pose in poses]
    cs = torch.stack([torch.from_numpy(c).cuda() for c in cs])

    c_front = encode_camera_params(
        Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
             camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
    c_front = torch.from_numpy(c_front).cuda().unsqueeze(0)

    if seeds is None:
        seeds = list(range(n_seeds))
    else:
        seeds = [int(seed) for seed in seeds.split(',')]

    with torch.no_grad():
        for seed in tqdm(seeds):
            rng = torch.Generator(device)
            rng.manual_seed(seed)
            z = torch.randn((1, G._config.z_dim), device='cuda', generator=rng)
            sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())

            w = G.mapping(z, c_front, truncation_psi=truncation_psi)
            w = w.repeat(len(cs), 1, 1)

            all_frames = []
            for c_batch, w_batch in zip(batchify_sliced(cs, batch_size=batch_size),
                                        batchify_sliced(w, batch_size=batch_size)):
                output = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     neural_rendering_resolution=resolution)
                frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in output['image']]
                all_frames.extend(frames)
            output_folder = f"{GGH_RENDERINGS_PATH}/sampled_heads/{run_name}"
            ensure_directory_exists(output_folder)
            mediapy.write_video(f"{output_folder}/{seed:04d}.mp4", all_frames, fps=24)


if __name__ == '__main__':
    tyro.cli(main)
