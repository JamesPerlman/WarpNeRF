import torch
import warp as wp
from warpnerf.models.batch import RayBatch
from warpnerf.training.dataset import Dataset
from warpnerf.training.mlp import MLP

class Trainer:

    mlp: MLP
    opt: torch.optim.Optimizer
    dataset: Dataset

    def __init__(self, dataset: Dataset, mlp: MLP, optimizer: torch.optim.Optimizer):
        self.mlp = mlp
        self.opt = optimizer
        self.dataset = dataset
    
    def step(self, batch: RayBatch, target_rgb: wp.array(dtype=wp.vec4)):
        self.opt.zero_grad()
        
        sample_xyz = Raymarcher(batch)
        sample_rgb = self.mlp(sample_xyz)
        pred_rgb = Accumulator(sample_rgb)
        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        loss.backward()

        self.opt.step()
    
