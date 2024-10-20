import torch
from fvdb import GridBatch
from warpnerf.models.mlp import MLP

class NeRFModel(torch.nn.Module):
    grid: GridBatch
    contraction: torch.autograd.Function
    dir_enc: torch.nn.Module
    pos_enc: torch.nn.Module
    mlp_base: MLP
    mlp_head: MLP

    def __init__() -> None:
        super().__init__()
