import fvdb
import math
import torch
import warp as wp
from torch import Tensor
from warpnerf.models.batch import RayBatch, SampleBatch
from warpnerf.models.trimiprf_model import TrimipRFModel

class TriMipRFTrainingRenderer:
    def __call__(self, model: TrimipRFModel, rays: RayBatch) -> tuple[Tensor, Tensor, Tensor]:

        ray_ori = wp.to_torch(rays.ori, requires_grad=False)
        ray_dir = wp.to_torch(rays.dir, requires_grad=False)
        t_min = torch.zeros(ray_ori.shape[0]).to(ray_ori)
        t_max = torch.full_like(t_min, fill_value=1e9)
        step_size = math.sqrt(3.0) / 512.0

        # pack_info, ray_idx, ray_intervals = model.grid.uniform_ray_samples(ray_ori, ray_dir, t_min, t_max, step_size)
        ray_intervals = model.grid.uniform_ray_samples(
            ray_origins=ray_ori,
            ray_directions=ray_dir,
            t_min=t_min,
            t_max=t_max,
            step_size=step_size
        )
        
        ray_idx = ray_intervals.jidx.int() #ray_idx.jdata
        pack_info = ray_intervals.joffsets

        sample_t = ray_intervals.jdata.mean(dim=1)
        sample_start = ray_intervals.jdata[:, 0]
        sample_end = ray_intervals.jdata[:, 1]
        sample_dt = (sample_end - sample_start).contiguous()
        sample_xyz = ray_ori[ray_idx] + ray_dir[ray_idx] * sample_t[:, None]
        sample_dir = ray_dir[ray_idx]

        # contract xyz
        sample_xyz = model.contraction(sample_xyz / model.aabb_size)

        # calculate sample radii for cone marching / sphere tracing
        ray_cos = wp.to_torch(rays.cos, requires_grad=False)
        ray_radius = wp.to_torch(rays.radius, requires_grad=False)
        ray_icos = 1.0 / ray_cos
        ray_tmp = (ray_icos * ray_icos - 1.0).sqrt() - ray_radius

        sample_tmp = ray_tmp[ray_idx]
        sample_cos = ray_cos[ray_idx]
        sample_radius = sample_t * ray_radius[ray_idx] * sample_cos / (sample_tmp * sample_tmp + 1.0).sqrt()
        sample_vol = torch.log2(2.0 * sample_radius / model.aabb_size)

        # query density
        sample_d, sample_geo = model.query_density(sample_xyz, sample_vol, return_feat=True)

        # query color
        sample_rgb = model.query_rgb(sample_dir, sample_geo)

        # volume render

        ray_rgb, ray_depth, ray_opacity, ray_w, ray_num_samples = fvdb.volume_render(
            sigmas=sample_d,
            rgbs=sample_rgb,
            deltaTs=sample_dt,
            ts=sample_t,
            packInfo=pack_info,
            transmittanceThresh=1e-5,
        )

        return ray_rgb, ray_depth, ray_opacity
