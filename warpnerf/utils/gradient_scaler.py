import torch

# Thank you https://gradient-scaling.github.io/#Code
class GradientScaler(torch.autograd.Function):
  @staticmethod
  def forward(ctx, colors, sigmas, ray_dist):
    ctx.save_for_backward(ray_dist)
    return colors, sigmas, ray_dist
  @staticmethod
  def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
    (ray_dist,) = ctx.saved_tensors
    scaling = torch.square(ray_dist).clamp(0, 1)
    return grad_output_colors * scaling.unsqueeze(-1), grad_output_sigmas * scaling, grad_output_ray_dist
