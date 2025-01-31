import math
import fvdb
import torch
import warp as wp
from fvdb import GridBatch
from warpnerf.losses.cauchy_sparsity_loss import cauchy_sparsity_loss
from warpnerf.losses.mipnerf360_distortion import MipNeRF360DistortionLoss
from warpnerf.losses.tv_loss import tv_loss
from warpnerf.models.dataset import Dataset
from warpnerf.models.mlp import MLP
from warpnerf.models.warpnerf_model import WarpNeRFModel
from warpnerf.rendering.nerf_renderer import generate_samples, query_samples, render_samples
from warpnerf.utils.gradient_scaler import GradientScaler

class Trainer:

    grid: GridBatch
    mlp: MLP
    opt: torch.optim.Optimizer
    dataset: Dataset

    n_steps: int = 0
    n_steps_to_subdivide: int = 1024
    max_subdivisions: int = 2
    target_samples_per_batch: int = None
    n_rays_per_batch: int = 4096
    n_samples_per_batch: int = None

    def __init__(
        self,
        dataset: Dataset,
        model: WarpNeRFModel,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataset = dataset
        self.model = model
        self.opt = optimizer
        self.grad_scaler = torch.amp.GradScaler()
        self.model.update_nonvisible_voxel_mask(self.dataset.camera_data)
    
    def step(self):
        self.opt.zero_grad()

        # with torch.autocast(device_type="cuda", enabled=True):
        rays, target_rgba = self.dataset.get_batch(
            n_rays=self.n_rays_per_batch,
            random_seed=self.n_steps
        )
        target_rgba = wp.to_torch(target_rgba)
        target_rgb = target_rgba[:3, :].T
        target_rgb = target_rgb.to(dtype=torch.float32) / 255.0
        target_alpha = target_rgba[3, :].to(dtype=torch.float32) / 255.0

        samples = generate_samples(self.model, rays, stratify=True)
        if self.target_samples_per_batch is None:
            self.target_samples_per_batch = samples.count

        self.n_samples_per_ray_avg = samples.count / rays.count
        self.n_rays_per_batch = int(self.target_samples_per_batch // self.n_samples_per_ray_avg)

        # print(f"n_rays in this batch: {rays.count}, n_samples in this batch: {samples.count}")
        # print(f"n_samples per ray avg: {self.n_samples_per_ray_avg}, n_rays per batch: {self.n_rays_per_batch}")


        if samples.count == 0:
            print("No samples in batch!")
            return
        
        samples = query_samples(self.model, samples)

        samples.rgb, samples.sigma, samples.t = GradientScaler.apply(samples.rgb, samples.sigma, samples.t)

        pred_rgb, pred_depth, pred_alpha = render_samples(samples)
        # mix target and pred with random background colors

        random_rgb = torch.rand(target_rgb.shape, device=target_rgb.device)
        
        pred_alpha = pred_alpha.unsqueeze(-1)
        pred_rgb = pred_rgb * pred_alpha + random_rgb * (1 - pred_alpha)
        
        target_alpha = target_alpha.unsqueeze(-1)
        target_rgb = target_rgb * target_alpha + random_rgb * (1 - target_alpha)

        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)

        # if self.model.n_subdivisions < 0:
        #     all_ijk = self.model.grid.ijk.jdata
        #     n_random_voxels = self.model.grid.total_voxels // 100
        #     random_ijk = all_ijk[torch.randperm(all_ijk.shape[0])[:n_random_voxels]]
        #     tv_reg_sh, tv_reg_o = tv_loss(
        #         grid=self.model.grid,
        #         ijk=random_ijk,
        #         features=(self.model.sh_features, self.model.o_features),
        #         res=self.model.grid_res
        #     )

        #     tv_reg = 1e-4 * tv_reg_sh + 1e-4 * tv_reg_o
        #     loss += tv_reg
        
        # distortion loss
        distortion_loss_lambda = 1e-5
        loss += MipNeRF360DistortionLoss.apply(
            distortion_loss_lambda,
            samples.n_samples,
            samples.offsets,
            samples.t,
            samples.dt,
            samples.sigma,
        )

        # maybe return sigma from distortion loss?
        
        # plenoxels cauchy sparsity loss
        # loss += 1e-5 * cauchy_sparsity_loss(
        #     num_rays=self.n_rays_per_batch,
        #     samples=samples,
        # )
        loss.backward()
        # self.grad_scaler.scale(loss).backward()
        self.opt.step()
        self.n_steps += 1

        if self.n_steps % 16 == 0 and self.n_steps > 512:
            self.model.update_grid_occupancy(threshold=0.01 * self.model.grid_res / math.sqrt(3))

        if self.n_steps % self.n_steps_to_subdivide == 0 and self.n_steps > 0 and self.model.n_subdivisions < self.max_subdivisions:
            self.model.subdivide_grid()
            self.model.update_nonvisible_voxel_mask(self.dataset.camera_data)
        
        print(f"the loss is {loss:.3f}, the step is {self.n_steps}, the grid is {self.model.percent_occupied:.0f}% occupied, there are {self.n_rays_per_batch} rays in this batch")
