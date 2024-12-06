from time import sleep
from fvdb import GridBatch, volume_render
import numpy as np
import torch
import warp as wp

from warpnerf.models.batch import create_ray_batch
from warpnerf.models.camera import TrainingCamera
from warpnerf.models.cascaded_occupancy_grid import CascadedOccupancyGrid
from warpnerf.server.visualization import VisualizationServer

import numpy as np

from warpnerf.utils.ray import get_rays_for_camera_kernel

def generate_debug_image(width, height):
    # Create an empty 3D array with dimensions (height, width, 3)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, 0] = np.linspace(0, 1, width).reshape(1, width)
    image[:, :, 2] = np.linspace(0, 1, height).reshape(height, 1)
    
    return image

def generate_density_features(grid: GridBatch, grid_shape: list[int], value: float) -> torch.Tensor:
    # make tuple of grid_shape
    grid_shape = tuple(grid_shape)
    
    # Create a tensor of shape (num_voxels_x, num_voxels_y, num_voxels_z) filled with the given value
    density_features = torch.full(grid_shape, value, device=grid.device)

    # Reshape to 2D: (total_voxels, 1)
    density_features = density_features.view(-1, 1)

    return density_features

def generate_occ_grid_density_feats(grid: CascadedOccupancyGrid) -> torch.Tensor:
    return torch.ones((grid.total_voxels, 1), device=grid.device)

def generate_occ_grid_rgb_feats(grid: CascadedOccupancyGrid) -> torch.Tensor:
    num_voxels = grid.total_voxels
    rgb_feats = torch.ones((num_voxels, 3), device=grid.device)
    return rgb_feats

def generate_rgb_features(grid: GridBatch, grid_shape: list[int]) -> torch.Tensor:
    # Assuming grid_batch has num_voxels as a list of voxel counts along each axis.
    num_voxels_x = int(grid_shape[0])  # Voxels along x-axis
    num_voxels_y = int(grid_shape[1])  # Voxels along y-axis
    num_voxels_z = int(grid_shape[2])  # Voxels along z-axis

    # Create the base tensors for each channel, scaled from 0 to 1
    r_values = torch.linspace(0, 1, num_voxels_x, device=grid.device).view(num_voxels_x, 1, 1)
    g_values = torch.linspace(0, 1, num_voxels_y, device=grid.device).view(1, num_voxels_y, 1)
    b_values = torch.linspace(0, 1, num_voxels_z, device=grid.device).view(1, 1, num_voxels_z)

    # Expand each to match the full (x, y, z) grid and stack them along the last dimension
    r_channel = r_values.expand(num_voxels_x, num_voxels_y, num_voxels_z)
    g_channel = g_values.expand(num_voxels_x, num_voxels_y, num_voxels_z)
    b_channel = b_values.expand(num_voxels_x, num_voxels_y, num_voxels_z)

    # Stack the RGB channels to form a final tensor with shape (num_voxels_x, num_voxels_y, num_voxels_z, 3)
    rgb_features = torch.stack([r_channel, g_channel, b_channel], dim=-1)
    
    # Reshape to 2D: (total_voxels, 3), where total_voxels = num_voxels_x * num_voxels_y * num_voxels_z
    rgb_features = rgb_features.view(-1, 3)
    
    return rgb_features


def hello_fvdb():
    wp.init()

    server = VisualizationServer()

    # initialize grid
    aabb_scale = 4.0
    grid = CascadedOccupancyGrid(
        aabb_scale_roi=aabb_scale,
        n_levels=3,
        resolution_roi=128,
        device="cuda"
    )
    rgb_feats = generate_occ_grid_rgb_feats(grid)
    alpha_feats = generate_occ_grid_density_feats(grid)
    step_size = np.sqrt(3.0) * grid.smallest_voxel_size

    # grid = GridBatch(device="cuda")
    # grid_res = 8
    # grid_shape = [grid_res] * 3
    # grid.set_from_dense_grid(
    #     num_grids=1,
    #     dense_dims=grid_shape,
    #     ijk_min=[0] * 3,
    #     voxel_sizes=aabb_scale / grid_res,
    #     origins=[0.5 * aabb_scale * (1.0 / grid_res - 1.0)] * 3
    # )

    # step_size = aabb_scale * np.sqrt(3.0) / grid_res

    # rgb_feats = generate_rgb_features(grid, grid_shape)
    # alpha_feats = generate_density_features(grid, grid_shape, 1.0)

    print(rgb_feats.shape)
    print(alpha_feats.shape)

    def render(camera: TrainingCamera):
        w, h = camera.image_dims
        rays = create_ray_batch(w * h)
        wp.launch(
            kernel=get_rays_for_camera_kernel,
            dim=w * h,
            inputs=[
                camera.camera_data,
                w,
                h,
                0
            ],
            outputs=[
                rays
            ]
        )

        ray_ori = wp.to_torch(rays.ori, requires_grad=False)
        ray_dir = wp.to_torch(rays.dir, requires_grad=False)
        t_min = torch.zeros(ray_ori.shape[0]).to(ray_ori)
        t_max = torch.full_like(t_min, fill_value=1e9)

        ray_intervals = grid.uniform_ray_samples(
            ray_origins=ray_ori,
            ray_directions=ray_dir,
            t_min=t_min,
            t_max=t_max,
            step_size=step_size
        )

        ray_idx = ray_intervals.jidx.int()
        pack_info = ray_intervals.joffsets

        sample_t = ray_intervals.jdata.mean(dim=1)
        sample_start = ray_intervals.jdata[:, 0]
        sample_end = ray_intervals.jdata[:, 1]
        sample_dt = (sample_end - sample_start).contiguous()
        sample_xyz = ray_ori[ray_idx] + ray_dir[ray_idx] * sample_t[:, None]
        sample_dir = ray_dir[ray_idx]

        if sample_xyz.numel() == 0:
            return np.zeros((h, w, 3), dtype=np.float32)

        sample_rgb = grid.sample_trilinear(
            points=sample_xyz,
            voxel_data=rgb_feats
        )

        sample_alpha = grid.sample_trilinear(
            points=sample_xyz,
            voxel_data=alpha_feats
        )

        rgb, depth, opacity, _, _ = volume_render(
            sigmas=sample_alpha.jdata.squeeze(-1),
            rgbs=sample_rgb.jdata,
            deltaTs=sample_dt,
            ts=sample_t,
            packInfo=pack_info,
            transmittanceThresh=1e-5
        )

        rgb = rgb.reshape(w, h, 3)
        # transpose to (h, w, 3) for visualization
        rgb = rgb.permute(1, 0, 2)
        rgb = np.array(rgb.detach().cpu().numpy(), dtype=np.float32)
        return rgb


    while True:
        sleep(0.1)
        cam = server.get_viewport_camera()
        if cam is None:
            continue
        w, h = cam.image_dims
        debug_img = render(cam)
        server.set_background_image(debug_img)