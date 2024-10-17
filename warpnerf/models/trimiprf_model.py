import math
import torch
from torch import Tensor
from warpnerf.encodings.spherical_harmonics import SHDeg4Encoding
from warpnerf.models.mlp import MLP
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.encodings.tri_mip import TriMipEncoding
from warpnerf.utils.trunc_exp import TruncExp

class TrimipRFModel(NeRFModel):
    def __init__(
        self,
        n_levels: int = 8,
        plane_size: int = 512,
        feature_dim: int = 16,
        geo_feat_dim: int = 15,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.feature_dim = feature_dim
        self.geo_feat_dim = geo_feat_dim

        self.pos_enc = TriMipEncoding(
            n_levels=n_levels,
            base_resolution=plane_size,
            feature_dim=feature_dim,
        )

        self.dir_enc = SHDeg4Encoding()

        self.density_activation = TruncExp()

        # input: encoded positions
        # output: density + geometric features (concatenated)
        self.mlp_base = MLP(
            input_dim=self.pos_enc.output_dim,
            hidden_dim=net_width,
            output_dim=1 + geo_feat_dim, # +1 for density
            n_hidden_layers=net_depth_base
        )

        # input: encoded directions + geometric features (concatenated)
        # output: RGB
        self.mlp_head = MLP(
            input_dim=self.dir_enc.output_dim + geo_feat_dim,
            hidden_dim=net_width,
            output_dim=3,
            n_hidden_layers=net_depth_color,
            output_activation=torch.nn.Sigmoid
        )
    
    def query_density(
        self,
        xyz: Tensor,
        vol: Tensor,
        return_feat: bool = False
    ) -> tuple[Tensor, Tensor]:
        
        vol = vol + self.log2_plane_size

        within_model = ((xyz > 0.0) & (xyz < 1.0)).all(dim=-1)

        # encode positions
        encoded_xyz = self.pos_enc(xyz, vol)

        # run head MLP
        mlp_result = (
            self.mlp_base(encoded_xyz)
                .view(xyz.shape[0], 1 + self.geo_feat_dim)
                .to(xyz)
        )

        # extract density
        d_raw, geo_feat = torch.split(mlp_result, [1, self.geo_feat_dim], dim=-1)

        # apply density activation and mask
        density = within_model[..., None] * self.density_activation(d_raw)
        
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
