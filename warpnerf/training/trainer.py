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

    n_steps: int = 0

    n_rays_per_batch: int = 32768

    def __init__(
        self,
        dataset: Dataset,
        model: NeRFModel,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataset = dataset
        self.model = model
        self.opt = optimizer
        self.renderer = TriMipRFTrainingRenderer()
    
    def step(self):
        self.opt.zero_grad()

        rays, target_rgba = self.dataset.get_batch(
            n_rays=self.n_rays_per_batch,
            random_seed=self.n_steps
        )

        target_rgb = wp.to_torch(target_rgba)[:3, :].T
        target_rgb = target_rgb.to(dtype=torch.float32) / 255.0

        pred_rgb, pred_depth, pred_alpha = self.renderer(self.model, rays)

        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        loss.backward()

        print(f"the loss is {loss}")

        self.opt.step()
        self.n_steps += 1

