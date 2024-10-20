import fvdb
import torch
import warp as wp
from fvdb import GridBatch
from warpnerf.models.batch import RayBatch
from warpnerf.models.dataset import Dataset
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.models.mlp import MLP
from warpnerf.rendering.trimiprf_training_renderer import TriMipRFTrainingRenderer

class Trainer:

    grid: GridBatch
    mlp: MLP
    opt: torch.optim.Optimizer
    dataset: Dataset
    renderer: TriMipRFTrainingRenderer

    def __init__(
        self,
        dataset: Dataset,
        model: NeRFModel,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataset = dataset
        self.model = model
        self.opt = optimizer
        self.raymarcher = TrainingRaymarcher()
    
    def step(self):
        self.opt.zero_grad()

        rays, target_rgb = self.dataset.get_batch()
        pred_rgb = self.renderer(self.model, rays)
        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        loss.backward()

        self.opt.step()

