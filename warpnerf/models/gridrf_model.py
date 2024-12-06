import math
import fvdb
import torch

from torch import Tensor
from fvdb import GridBatch

from warpnerf.encodings.spherical_harmonics import SHDeg4Encoding
from warpnerf.models.mlp import MLP
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.utils.merf_contraction import MERFContraction
from warpnerf.utils.spherical_harmonics import evaluate_sh, evaluate_sh_bases
from warpnerf.utils.trunc_exp import TruncExp

# Grid Radiance Field model based on the fVDB example code

class GridRFModel(NeRFModel):

    o_features: Tensor
    sh_features: Tensor
    n_subdivisions: int

    on_subdivide_grid: callable
    
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

        self.color_activation = torch.sigmoid

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

        xyz = self.grid.ijk.jdata / self.grid_res
        
        self.register_parameter(
            "o_features",
            torch.nn.Parameter(
                torch.full(
                    size=(self.grid.total_voxels,),
                    fill_value=0.0,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True
                ),
                requires_grad=True
            )
        )

        self.register_parameter(
            "sh_features",
            torch.nn.Parameter(
                torch.stack([evaluate_sh_bases(9, torch.rand_like(xyz) - 0.5)] * 3, dim=1).to(device=device, dtype=torch.float32),
                requires_grad=True
            )
        )

        # set up contraction function
        self.contraction = MERFContraction.apply

    @property
    def step_size(self) -> float:
        return self.aabb_scale * math.sqrt(3.0) / (2 * self.grid_res)
    
    def query_sigma(
        self,
        xyz: Tensor
    ) -> Tensor:

        xyz_o_features = self.grid.sample_trilinear(
            xyz,
            self.o_features.unsqueeze(-1)
        ).jdata.squeeze(-1)
        
        return self.density_activation(xyz_o_features)

    def query_rgb(
        self,
        xyz: Tensor,
        dir: Tensor,
    ) -> Tensor:

        xyz_sh_features = self.grid.sample_trilinear(
            xyz,
            self.sh_features.view(self.sh_features.shape[0], -1)
        ).jdata.view(xyz.shape[0], 3, 9)

        rgb_raw = evaluate_sh(2, xyz_sh_features, dir)

        return self.color_activation(rgb_raw)
    
    def subdivide_grid(self, density_thresh: float = 0.25) -> None:
        vox_ijk = self.grid.ijk.jdata
        vox_xyz = self.grid.grid_to_world(vox_ijk.to(torch.float32)).jdata
        vox_density = self.query_sigma(vox_xyz)

        density_mask = vox_density > density_thresh

        sh_features, sub_grid = self.grid.subdivide(
            subdiv_factor=2,
            data=self.sh_features.view(self.sh_features.shape[0], -1),
            mask=density_mask
        )

        o_features, sub_grid = self.grid.subdivide(
            subdiv_factor=2,
            data=self.o_features.unsqueeze(-1),
            mask=density_mask
        )

        self.register_parameter(
            "sh_features",
            torch.nn.Parameter(
                sh_features.jdata.reshape(sh_features.rshape[0], 3, -1),
                requires_grad=True
            )
        )

        self.register_parameter(
            "o_features",
            torch.nn.Parameter(
                o_features.jdata.squeeze(-1),
                requires_grad=True
            )
        )

        self.grid = sub_grid
        self.grid_res *= 2
        self.n_subdivisions += 1

        self.on_subdivide_grid(self)

