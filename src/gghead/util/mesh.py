from typing import Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm
try:
    import pyvista as pv
except OSError as e:
    # Sometimes vtk has this weird bug "Error loading vtkioss-9.3-c0f4add4a2b52302512f2df0aa56b1e8.dll; The operation completed successfully." ...
    print(e)

from gaussian_splatting.utils.general_utils import build_scaling_rotation, build_rotation


def gaussians_to_mesh(
        gaussian_positions: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_rotations: torch.Tensor,
        gaussian_colors: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        use_spheres: bool = True,
        random_colors: bool = False,
        scale_factor: float = 1.5,
        ellipsoid_res: int = 5,
        opacity_threshold: float = 0.01,
        max_n_gaussians: Optional[int] = None,
        include_alphas: bool = False
) -> trimesh.Trimesh:
    gaussian_positions = gaussian_positions.detach().cpu().numpy()
    gaussian_colors = gaussian_colors.detach().cpu().numpy()
    gaussian_opacities = gaussian_opacities.detach().cpu().numpy()

    n_gaussians = len(gaussian_positions) if max_n_gaussians is None else max_n_gaussians

    if use_spheres:
        points = []
        faces = []
        points_count = 0
        face_count = 0
        all_vertex_colors = []

        base = trimesh.creation.icosphere(subdivisions=1)  # radius=0.5, count=16)

        rotm = build_scaling_rotation(gaussian_scales * scale_factor, gaussian_rotations).cpu().numpy()
        for i in range(n_gaussians):
            if gaussian_opacities[i] >= opacity_threshold:
                points.append(base.vertices @ rotm[i, ...].T + gaussian_positions[i:i + 1, :])
                tris = base.faces
                face_count += tris.shape[0]
                faces.append(tris + points_count)
                points_count += base.vertices.shape[0]

                if random_colors:
                    sphere_color = np.random.rand(3)
                else:
                    sphere_color = gaussian_colors[i]
                if include_alphas:
                    vertex_colors = np.tile(np.concatenate([sphere_color[None, :], np.clip(gaussian_opacities[[i]], 0, 1)], axis=1), [base.vertices.shape[0], 1])
                else:
                    vertex_colors = np.tile(sphere_color[None, :], [base.vertices.shape[0], 1])
                all_vertex_colors.append(vertex_colors)

        points = np.concatenate(points, axis=0)
        all_vertex_colors = np.concatenate(all_vertex_colors, axis=0)
        faces = np.concatenate(faces, axis=0)
        combined_mesh = trimesh.Trimesh(points, faces, process=False, vertex_colors=all_vertex_colors)

    else:
        gaussian_scales = gaussian_scales.cpu()
        gaussian_rotations = build_rotation(gaussian_rotations).cpu().numpy()

        ellipsoids = []
        for i in tqdm(list(range(n_gaussians))):
            scale = gaussian_scales[i] * scale_factor
            ellipsoid = pv.ParametricEllipsoid(scale[0], scale[1], scale[2], center=gaussian_positions[i], u_res=ellipsoid_res, v_res=ellipsoid_res,
                                               w_res=ellipsoid_res)
            ellipsoids.append(ellipsoid)

        all_vertex_colors = []
        ellipsoid_meshes = []
        for ellipsoid, ellipsoid_center, ellipsoid_color, ellipsoid_opacity, ellipsoid_rotation in zip(ellipsoids, gaussian_positions, gaussian_colors,
                                                                                                       gaussian_opacities, gaussian_rotations):
            if ellipsoid_opacity >= opacity_threshold:
                faces_as_array = ellipsoid.faces.reshape((ellipsoid.n_cells, 4))[:, 1:]
                # tmesh = trimesh.Trimesh(ellipsoid.points, faces_as_array, process=False, vertex_colors=np.concatenate([ellipsoid_color, ellipsoid_opacity]))
                vertices = ellipsoid.points
                vertices = ((vertices - ellipsoid_center) @ ellipsoid_rotation) + ellipsoid_center
                if random_colors:
                    ellipsoid_color = np.random.rand(3)
                tmesh = trimesh.Trimesh(vertices, faces_as_array, process=False, vertex_colors=ellipsoid_color)
                all_vertex_colors.extend(tmesh.visual.vertex_colors)
                ellipsoid_meshes.append(tmesh)
        combined_mesh = trimesh.util.concatenate(ellipsoid_meshes)

    return combined_mesh
