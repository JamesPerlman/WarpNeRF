import math
from typing import Tuple, Union
import tinycudann as tcnn
import torch
import warp as wp

from torch import Tensor
from fvdb import GridBatch

from warpnerf.encodings.spherical_harmonics import SHDeg4Encoding
from warpnerf.models.camera import CameraData
from warpnerf.models.cascaded_occupancy_grid import CascadedOccupancyGrid
from warpnerf.models.mlp import MLP
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.utils.cameras import increment_point_visibility_kernel
from warpnerf.utils.merf_contraction import MERFContraction
from warpnerf.utils.spherical_harmonics import evaluate_sh, evaluate_sh_bases
from warpnerf.utils.trunc_exp import TruncExp

# Custom Neural Radiance Field model based on a lot of different research

class WarpNeRFModel(torch.nn.Module):

    def __init__(
        self,
        aabb_scale: float,
        n_appearance_embeddings: int
    ) -> None:
        super().__init__()

        device = torch.device("cuda")

        self.aabb_scale = aabb_scale
        self.n_subdivisions = 0

        self.contraction = MERFContraction.apply

        self.dir_enc = SHDeg4Encoding()

        self.density_activation = TruncExp.apply

        self.percent_occupied = 0.0

        # initialize grid
        # grid_res = 128
        # self.grid = CascadedOccupancyGrid(
        #     aabb_scale_roi=aabb_scale,
        #     n_levels=5,
        #     resolution_roi=grid_res,
        #     device=device
        # )

        # initialize grid
        self.grid = GridBatch(device, mutable=True)
        self.grid.set_from_dense_grid(
            num_grids=1,
            dense_dims=[self.grid_res] * 3,
            ijk_min=[0] * 3,
            voxel_sizes=self.voxel_size,
            origins=[0.5 * self.aabb_scale * (1.0 / self.grid_res - 1.0)] * 3
        )
        self.grid_sigma = None
        self.occupancy_mask = None

        # init appearance embedding
        self.appearance_embedding = torch.nn.Embedding(
            num_embeddings=n_appearance_embeddings,
            embedding_dim=16,
            device=device
        )

        # density network with hash encoding
        num_levels = 16
        min_res = 16
        max_res = 1024
        log2_hashmap_size = 19
        features_per_level = 2
        per_level_scale = math.exp((math.log(max_res) - math.log(min_res)) / (num_levels - 1))

        hashgrid_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": per_level_scale,
        }

        n_density_features = 15

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + n_density_features,
            encoding_config=hashgrid_config,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        # color network with spherical harmonics encoding
        self.dir_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            }
        )

        color_net_input_dims = n_density_features + self.dir_enc.n_output_dims + self.appearance_embedding.embedding_dim

        self.mlp_head = tcnn.Network(
            n_input_dims=color_net_input_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
    
    @property
    def grid_res(self) -> int:
        return 256 * (2 ** self.n_subdivisions)

    @property
    def voxel_size(self) -> float:
        return self.aabb_scale / self.grid_res

    @property
    def step_size(self) -> float:
        return self.voxel_size * math.sqrt(3.0) / 2.0

    def query_sigma(
        self,
        xyz: Tensor,
        return_feat: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        xyz_in = torch.clamp(xyz / self.aabb_scale + 0.5, 0.0, 1.0)
        density_output = self.mlp_base(xyz_in)
        d_raw, feat = density_output.split([1, 15], dim=-1)
        
        d_raw = d_raw.squeeze(-1)
        feat = feat.squeeze(-1)

        sigma = self.density_activation(d_raw.float())

        if return_feat:
            return sigma, feat
        else:
            return sigma

    def query_rgb(
        self,
        dir: Tensor,
        density_feat: Tensor,
        embedding_idx: Tensor,
    ) -> Tensor:
        dir_norm = (dir + 1.0) / 2.0
        dir_encoded = self.dir_enc(dir_norm)
        embedded_appearance = self.appearance_embedding(embedding_idx)
        color_in = torch.cat([density_feat, dir_encoded, embedded_appearance], dim=-1)
        color_out = self.mlp_head(color_in).float()
        return color_out
    
    def update_grid_occupancy(self, threshold: float = 0.01, decay_rate: float = 0.95):
        with torch.no_grad():
            batch_size = 128 ** 3
            vox_ijk = self.grid.ijk.jdata
            n_batches = (self.grid.total_voxels + batch_size - 1) // batch_size
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, self.grid.total_voxels)
                vox_batch = vox_ijk[start:end]
                voxel_centers = self.grid.grid_to_world(vox_batch.to(torch.float32)).jdata
                random_offsets = torch.rand_like(voxel_centers) - 0.5
                xyz = voxel_centers + random_offsets * self.voxel_size
                sigma = self.query_sigma(xyz, return_feat=False)
                
                if self.grid_sigma is None:
                    self.grid_sigma = torch.empty((self.grid.total_voxels,), dtype=torch.float32, device=vox_ijk.device)
                    self.grid_sigma[start:end] = sigma
                else:
                    self.grid_sigma[start:end] = torch.max(decay_rate * self.grid_sigma[start:end], sigma)

                enable_mask = self.grid_sigma[start:end] >= threshold
                # if self.nonvisible_voxel_mask is not None:
                #     enable_mask = enable_mask & ~self.nonvisible_voxel_mask

                disable_mask = ~enable_mask
                if self.occupancy_mask is None or self.occupancy_mask.shape[0] != self.grid.total_voxels:
                    self.occupancy_mask = torch.zeros((self.grid.total_voxels,), dtype=torch.bool, device=vox_ijk.device)
                self.occupancy_mask[start:end] = enable_mask

                # self.grid.enable_ijk(vox_ijk[enable_mask])
                self.grid.disable_ijk(vox_batch[disable_mask])
            
            self.percent_occupied = 100 * (self.grid.total_enabled_voxels / self.grid.total_voxels)

    def subdivide_grid(self):
        with torch.no_grad():
            new_grid_sigma, new_grid = self.grid.subdivide(2, self.grid_sigma.unsqueeze(-1), mask=self.occupancy_mask)
            self.grid_sigma = new_grid_sigma.jdata.squeeze(-1)
            self.grid = new_grid
            self.n_subdivisions += 1

    def update_nonvisible_voxel_mask(self, cameras: wp.array1d(dtype=CameraData)):
        return
        with torch.no_grad():
            vox_ijk = self.grid.ijk.jdata
            voxel_centers = self.grid.grid_to_world(vox_ijk.to(torch.float32)).jdata
            voxel_centers = wp.from_torch(voxel_centers, dtype=wp.vec3f)
            n_voxels = self.grid.total_voxels

            visibilities = wp.zeros(shape=(n_voxels,), dtype=wp.int32, device=cameras.device)
            wp.launch(
                kernel=increment_point_visibility_kernel,
                dim=n_voxels,
                inputs=[cameras, voxel_centers],
                outputs=[visibilities],
            )
            wp.synchronize()

            visibilities = wp.to_torch(visibilities)
            self.nonvisible_voxel_mask = visibilities == 0

            self.grid.disable_ijk(vox_ijk[self.nonvisible_voxel_mask])
