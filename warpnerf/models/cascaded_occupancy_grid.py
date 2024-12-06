import math
from typing import Callable, Union
import fvdb
import torch

JaggedTensorOrTensor = Union[torch.Tensor, fvdb.JaggedTensor]

class CascadedOccupancyGrid:
    def __init__(
        self,
        aabb_scale_roi: float, # region of interest of the finest grid
        n_levels: int,
        resolution_roi: int = 128,
        device: str = "cuda",
    ):
        assert n_levels <= 8, "n_levels must be <= 8"

        self.device = device
        self.n_levels = n_levels
        self.aabb_scale = (2 ** n_levels) * aabb_scale_roi
        self.grid_res = resolution_roi
        self.grid = fvdb.GridBatch(device)
        self.grid.set_from_dense_grid(
            num_grids=1,
            dense_dims=[self.grid_res] * 3,
            ijk_min=[0] * 3,
            voxel_sizes=self.aabb_scale / self.grid_res,
            origins=[0.5 * self.aabb_scale * (1.0 / self.grid_res - 1.0)] * 3
        )

        all_ijk = self.grid.ijk.jdata

        # create subdivision marker data

        n_subdivisions = torch.empty(
            size=(self.grid.total_voxels,),
            device=device,
            dtype=torch.float32,
        )

        for i in range(n_levels):
            n_vox_in_level = self.grid_res // (2 ** i)
            index_min = (self.grid_res - n_vox_in_level) // 2
            index_max = index_min + n_vox_in_level
            print(f"{n_vox_in_level} in level {i} from {index_min} to {index_max}")
            level_mask = torch.where((all_ijk >= index_min) & (all_ijk < index_max))
            print(level_mask)
            level_indices = all_ijk[level_mask]
            n_subdivisions[level_indices] = i + 1
            print(n_subdivisions[level_indices].shape, "set to", i + 1)
            print(torch.where(n_subdivisions == float(i + 1)))
        
        print(self.grid.total_voxels, "!!!!!!")
        # subdivide grid

        for i in range(0, n_levels):
            new_n_subdivisions, new_grid = self.grid.subdivide(
                subdiv_factor=i + 1,
                data=n_subdivisions.unsqueeze(-1),
                mask=(n_subdivisions == i + 1),
            )

            n_subdivisions = torch.round(new_n_subdivisions.jdata.squeeze(-1))
            self.grid = new_grid
            print("!", self.grid.total_voxels)

        self.voxel_sizes = self.aabb_scale / (self.grid_res * (2 ** n_subdivisions))

        self.smallest_voxel_size = self.aabb_scale / (self.grid_res * (2 ** n_levels))

        self.n_subdivisions = n_subdivisions

    @property
    def total_voxels(self) -> int:
        return self.grid.total_voxels

    def uniform_ray_samples(
        self,
        ray_origins: JaggedTensorOrTensor,
        ray_directions: JaggedTensorOrTensor,
        t_min: JaggedTensorOrTensor,
        t_max: JaggedTensorOrTensor,
        step_size: float,
    ) -> torch.Tensor:
        return self.grid.uniform_ray_samples(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            t_min=t_min,
            t_max=t_max,
            step_size=step_size
        )

    def update_occupancy(
        self,
        query_sigma_fn: callable,
        threshold: float,
    ):
        vox_ijk = self.grid.ijk.jdata
        voxel_centers = self.grid.grid_to_world(vox_ijk.to(torch.float32)).jdata
        random_offsets = torch.rand_like(voxel_centers) - 0.5
        xyz = voxel_centers + random_offsets * self.voxel_sizes[vox_ijk.int()]
        sigma = query_sigma_fn(xyz)
        
        if self.grid_sigma is None:
            self.grid_sigma = sigma
            return

        self.grid_sigma = torch.max(self.decay_rate * self.grid_sigma, sigma)
        self.grid.enable_ijk(torch.where(self.grid_sigma >= threshold))
        self.grid.disable_ijk(torch.where(self.grid_sigma < threshold))

    def sample_trilinear(
        self,
        points: JaggedTensorOrTensor,
        voxel_data: JaggedTensorOrTensor,
    ) -> torch.Tensor:
        return self.grid.sample_trilinear(
            points=points,
            voxel_data=voxel_data
        )