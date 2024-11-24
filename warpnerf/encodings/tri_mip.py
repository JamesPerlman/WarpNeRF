import numpy as np
import nvdiffrast.torch
import torch
import warp as wp

class TriMipEncoding(torch.nn.Module):
    def __init__(
        self,
        n_levels: int,
        plane_size: int,
        feature_dim: int,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim

        self.register_parameter(
            "texture",
            torch.nn.Parameter(
                data=torch.zeros(
                    3, plane_size, plane_size, feature_dim, device="cuda"
                ),
                # requires_grad=True,
            )
        )
        
        torch.nn.init.uniform_(self.texture, -1e-2, 1e-2)

    def forward(
        self,
        x, # normalized to [0, 1]
        level # in [0, n_levels - 1]
    ):

        if x.shape[0] == 0:
            return torch.zeros([0, self.feature_dim * 3], device=x.device)
        
        x_decomposed = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )

        # torch.stack([level, level, level], dim=0)
        level = level.view(1, -1, 1)
        level = torch.broadcast_to(level, x_decomposed.shape[:3]).contiguous()

        # shape is 3xNx1xC
        result = nvdiffrast.torch.texture(
            tex=self.texture,
            uv=x_decomposed,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )

        # shape is Nx(3C)
        return result.permute(1, 2, 0, 3).contiguous().view(x.shape[0], self.feature_dim * 3)

    @property
    def input_dim(self):
        return 3
    
    @property
    def output_dim(self):
        return self.feature_dim * 3
