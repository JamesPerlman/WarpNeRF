import math
import torch
import warp as wp
from fvdb import GridBatch
from torch import Tensor
from warpnerf.encodings.positional import PositionalEncoding
from warpnerf.encodings.spherical_harmonics import SHDeg4Encoding
from warpnerf.models.mlp import MLP
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.encodings.tri_mip import TriMipEncoding
from warpnerf.utils.merf_contraction import MERFContraction
from warpnerf.utils.trunc_exp import TruncExp

class TrimipRFModel(NeRFModel):
    def __init__(
        self,
        aabb_scale: float = 1.0,
        n_levels: int = 8,
        plane_size: int = 512,
        feature_dim: int = 16,
        geo_feat_dim: int = 15,
        net_depth_base: int = 8,
        net_depth_color: int = 4,
        net_width: int = 128
    ) -> None:
        super().__init__()

        device = torch.device("cuda")

        self.aabb_scale = aabb_scale
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.feature_dim = feature_dim
        self.geo_feat_dim = geo_feat_dim

        self.pos_enc = TriMipEncoding(
            n_levels=n_levels,
            plane_size=plane_size,
            feature_dim=feature_dim,
        )

        self.dir_enc = SHDeg4Encoding()

        self.density_activation = TruncExp.apply

        # input: encoded positions
        # output: density + geometric features (concatenated)
        self.mlp_base = MLP(
            input_dim=self.pos_enc.output_dim,
            hidden_dim=net_width,
            output_dim=1 + geo_feat_dim, # +1 for density
            n_hidden_layers=net_depth_base
        ).to(device)

        # input: encoded directions + geometric features (concatenated)
        # output: RGB
        self.mlp_head = MLP(
            input_dim=self.dir_enc.output_dim + geo_feat_dim,
            hidden_dim=net_width,
            output_dim=3,
            n_hidden_layers=net_depth_color,
            output_activation=torch.nn.Sigmoid
        ).to(device)

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

        # set up contraction function
        self.contraction = MERFContraction.apply

    
    def query_density(
        self,
        xyz: Tensor,
        vol: Tensor,
        return_feat: bool = False
    ) -> tuple[Tensor, Tensor]:
        
        # vol = torch.clamp(vol + self.log2_plane_size, 0.0, self.log2_plane_size - 1.0)
        vol = vol + self.log2_plane_size
        # within_model = ((xyz > 0.0) & (xyz < 1.0)).all(dim=-1)

        # encode positions
        contracted_xyz = self.contraction(xyz)
        encoded_xyz = self.pos_enc(contracted_xyz.view(-1, 3), vol.view(-1, 1))

        # run head MLP
        mlp_result = (
            self.mlp_base(encoded_xyz)
                .view(xyz.shape[0], 1 + self.geo_feat_dim)
                .to(xyz)
        )

        # extract density
        d_raw, geo_feat = torch.split(mlp_result, [1, self.geo_feat_dim], dim=-1)

        # apply density activation and mask
        density = self.density_activation(d_raw.squeeze(-1))
        
        if return_feat:
            return density, geo_feat
        else:
            return density, None


    def query_rgb(
        self,
        dir: Tensor,
        geo_feat: Tensor,
    ) -> Tensor:
        
        # dirs are in [-1, 1]; normalize them to [0, 1]
        dir = (dir + 1.0) / 2.0

        # encode directions
        encoded_dir = self.dir_enc(dir)

        # concatenate directions with geometric features
        feat = torch.cat([encoded_dir, geo_feat], dim=-1)

        # query RGB
        return self.mlp_head(feat)
