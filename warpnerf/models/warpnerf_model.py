import math
import tinycudann as tcnn
import torch

from torch import Tensor
from fvdb import GridBatch

from warpnerf.encodings.spherical_harmonics import SHDeg4Encoding
from warpnerf.models.cascaded_occupancy_grid import CascadedOccupancyGrid
from warpnerf.models.mlp import MLP
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.utils.merf_contraction import MERFContraction
from warpnerf.utils.spherical_harmonics import evaluate_sh, evaluate_sh_bases
from warpnerf.utils.trunc_exp import TruncExp

# Custom Neural Radiance Field model based on a lot of different research

class WarpNeRFModel(torch.nn.Module):

    def __init__(
        self,
        aabb_scale: float = 1.0,
    ) -> None:
        super().__init__()

        device = torch.device("cuda")

        self.aabb_scale = aabb_scale
        self.n_subdivisions = 0

        self.contraction = MERFContraction.apply

        self.dir_enc = SHDeg4Encoding()

        self.density_activation = TruncExp.apply

        # initialize grid
        # grid_res = 128
        # self.grid = CascadedOccupancyGrid(
        #     aabb_scale_roi=aabb_scale,
        #     n_levels=5,
        #     resolution_roi=grid_res,
        #     device=device
        # )

        # initialize grid
        self.grid_res = 256
        self.grid = GridBatch(device)
        self.grid.set_from_dense_grid(
            num_grids=1,
            dense_dims=[self.grid_res] * 3,
            ijk_min=[0] * 3,
            voxel_sizes=self.aabb_scale / self.grid_res,
            origins=[0.5 * self.aabb_scale * (1.0 / self.grid_res - 1.0)] * 3
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

        color_net_input_dims = n_density_features + self.dir_enc.n_output_dims

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
    def step_size(self) -> float:
        return self.aabb_scale * math.sqrt(3.0) / (2 * self.grid_res)

    def query_sigma(
        self,
        xyz: Tensor,
        return_feat: bool = False
    ) -> Tensor:
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
        density_feat: Tensor
    ) -> Tensor:
        dir_norm = (dir + 1.0) / 2.0
        dir_encoded = self.dir_enc(dir_norm)
        color_in = torch.cat([density_feat, dir_encoded], dim=-1)

        color_out = self.mlp_head(color_in).float()

        return color_out
