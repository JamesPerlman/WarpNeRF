import fvdb
import math
import torch
import warp as wp
from warpnerf.models.batch import RayBatch, SampleBatch
from warpnerf.models.nerf_model import NeRFModel

class TrainingRaymarcher:
    def __call__(self, model: NeRFModel, rays: RayBatch) -> SampleBatch:

        ray_ori = wp.to_torch(rays.ori, requires_grad=False)
        ray_dir = wp.to_torch(rays.dir, requires_grad=False)
        t_min = torch.zeros(ray_ori.shape[0]).to(ray_ori)
        t_max = torch.full_like(t_min, fill_value=1e9)
        step_size = math.sqrt(3.0) / 512.0

        pack_info, ray_idx, ray_intervals = model.grid.uniform_ray_samples(ray_ori, ray_dir, t_min, t_max, step_size)
        
        sample_t = ray_intervals.jdata.mean(dim=1)
        sample_dt = (ray_intervals.jdata[:, 1] - ray_intervals.jdata[:, 0]).contiguous()
        sample_xyz = ray_ori[ray_idx.jdata] + ray_dir[ray_idx.jdata] * sample_t[:, None]

        samples = SampleBatch()
        samples.t = wp.from_torch(sample_t)
        samples.dt = wp.from_torch(sample_dt)
        samples.xyz = wp.from_torch(sample_xyz)
        samples.offset = wp.from_torch(pack_info.jdata)

        return samples
