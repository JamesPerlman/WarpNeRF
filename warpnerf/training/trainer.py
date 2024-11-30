import fvdb
import torch
import warp as wp
from fvdb import GridBatch
from warpnerf.losses.cauchy_sparsity_loss import cauchy_sparsity_loss
from warpnerf.losses.tv_loss import tv_loss
from warpnerf.models.dataset import Dataset
from warpnerf.models.gridrf_model import GridRFModel
from warpnerf.models.mlp import MLP
from warpnerf.rendering.basic_grid_renderer import generate_samples, query_samples, render_samples
from warpnerf.utils.gradient_scaler import GradientScaler

class Trainer:

    grid: GridBatch
    mlp: MLP
    opt: torch.optim.Optimizer
    dataset: Dataset

    n_steps: int = 0
    n_steps_to_subdivide: int = 1000
    max_subdivisions: int = 30

    n_rays_per_batch: int = 4096

    def __init__(
        self,
        dataset: Dataset,
        model: GridRFModel,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataset = dataset
        self.model = model
        self.opt = optimizer
    
    def step(self):
        self.opt.zero_grad()

        rays, target_rgba = self.dataset.get_batch(
            n_rays=self.n_rays_per_batch,
            random_seed=self.n_steps
        )
        target_rgba = wp.to_torch(target_rgba)
        target_rgb = target_rgba[:3, :].T
        target_rgb = target_rgb.to(dtype=torch.float32) / 255.0
        target_alpha = target_rgba[3, :].to(dtype=torch.float32) / 255.0

        samples = generate_samples(self.model, rays, stratify=True)

        samples = query_samples(self.model, samples)

        samples.rgb, samples.density, samples.t = GradientScaler.apply(samples.rgb, samples.density, samples.t)

        pred_rgb, pred_depth, pred_alpha = render_samples(samples)

        # mix target and pred with random background colors
        random_rgb = torch.rand(target_rgb.shape, device=target_rgb.device)
        
        pred_alpha = pred_alpha.unsqueeze(-1)
        pred_rgb = pred_rgb * pred_alpha + random_rgb * (1 - pred_alpha)
        
        target_alpha = target_alpha.unsqueeze(-1)
        target_rgb = target_rgb * target_alpha + random_rgb * (1 - target_alpha)

        loss = torch.nn.functional.smooth_l1_loss(pred_rgb, target_rgb)

        if self.model.n_subdivisions > 0:
            all_ijk = self.model.grid.ijk.jdata
            n_random_voxels = self.model.grid.total_voxels // 100
            random_ijk = all_ijk[torch.randperm(all_ijk.shape[0])[:n_random_voxels]]
            tv_reg_sh, tv_reg_o = tv_loss(
                grid=self.model.grid,
                ijk=random_ijk,
                features=(self.model.sh_features, self.model.o_features),
                res=self.model.grid_res
            )

            tv_reg = 1e-4 * tv_reg_sh + 1e-4 * tv_reg_o
            loss += tv_reg
        
        # plenoxels cauchy sparsity loss
        # loss += 1e-5 * cauchy_sparsity_loss(
        #     num_rays=self.n_rays_per_batch,
        #     samples=samples,
        # )

        loss.backward()
        self.opt.step()
        self.n_steps += 1

        if self.n_steps % self.n_steps_to_subdivide == 0 and self.n_steps > 0 and self.model.n_subdivisions < 4:
            self.model.subdivide_grid()

        print(f"the loss is {loss}, the step is {self.n_steps}")
