import numpy as np
import torch
import warp as wp
from torch.autograd import gradcheck

from warpnerf.losses.mipnerf360_distortion import MipNeRF360DistortionLoss

wp.init()
# Define example inputs
distortion_loss_lambda = 1.0
ray_n_samples = torch.tensor([32, 32], dtype=torch.int64, requires_grad=False, device="cuda")
ray_sample_offset = torch.tensor([0, 32], dtype=torch.int64, requires_grad=False, device="cuda")
sample_m_norm = torch.rand(64, dtype=torch.float32, requires_grad=False, device="cuda")
sample_dt_norm = torch.rand(64, dtype=torch.float32, requires_grad=False, device="cuda")
sample_sigma = torch.rand(64, dtype=torch.float32, requires_grad=True, device="cuda")  # Must have requires_grad=True

# Wrap the function to be tested
def func(sample_sigma):
    return MipNeRF360DistortionLoss.apply(
        distortion_loss_lambda,
        ray_n_samples,
        ray_sample_offset,
        sample_m_norm,
        sample_dt_norm,
        sample_sigma
    )

# Run gradcheck
inputs = (sample_sigma.clone().requires_grad_(),)  # Only the differentiable input
assert gradcheck(func, inputs, eps=1e-4, atol=1e-2), "Gradcheck failed!"