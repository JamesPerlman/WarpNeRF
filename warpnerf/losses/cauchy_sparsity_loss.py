import torch

from warpnerf.models.batch import SampleBatch

# from Plenoxels
def cauchy_sparsity_loss(num_rays: int, samples: SampleBatch) -> torch.Tensor:
    zeros = torch.zeros(num_rays, device=samples.density.device)
    idx = samples.ray_idx.to(dtype=torch.int64)
    density_sum = zeros.scatter_add(0, idx, samples.density)
    return density_sum.mean()
