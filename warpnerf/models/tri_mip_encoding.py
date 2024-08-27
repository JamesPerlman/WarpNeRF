import numpy as np
import warp as wp

import nvdiffrast.torch
import torch


class TriMipEncoding(torch.nn.Module):
    def __init__(
        self,
        n_levels: int,
        base_resolution: int,
        feature_dim: int,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.feature_dim = feature_dim

        self.register_parameter(
            "texture",
            torch.nn.parameter(torch.zeros(3, base_resolution, base_resolution, feature_dim)),
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

        level = torch.stack([level, level, level], dim=0)

        level = torch.broadcast_to(level, x_decomposed.shape[:3]).contiguous()

        # shape is 3xNx1xC
        enc = nvdiffrast.torch.texture(
            tex=self.texture,
            uv=x_decomposed,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )

        # shape is Nx(3C)
        enc = enc.permute(1, 2, 0, 3).contiguous().view(x.shape[0], self.feature_dim * 3)

        return enc
