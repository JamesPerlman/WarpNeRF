import torch
import warp as wp
from fvdb import GridBatch
from warpnerf.models.mlp import MLP

class NeRFModel(torch.nn.Module):
    grid: GridBatch
    grid_res: int
    contraction: torch.autograd.Function
    dir_enc: torch.nn.Module
    pos_enc: torch.nn.Module
    mlp_base: MLP
    mlp_head: MLP
    aabb_size: float

    def __init__(self) -> None:
        super().__init__()
