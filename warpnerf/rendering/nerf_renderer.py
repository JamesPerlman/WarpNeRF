import fvdb
import torch
import warp as wp
from torch import Tensor
from warpnerf.models.batch import RayBatch, SampleBatch
from warpnerf.models.warpnerf_model import WarpNeRFModel

def generate_samples(
    model: WarpNeRFModel,
    rays: RayBatch,
    stratify: bool = False
) -> SampleBatch:
    with torch.no_grad():
        ray_ori = wp.to_torch(rays.ori, requires_grad=False)
        ray_dir = wp.to_torch(rays.dir, requires_grad=False)
        t_min = torch.zeros(ray_ori.shape[0]).to(ray_ori)
        t_max = torch.full_like(t_min, fill_value=1e9)
        cam_idx = wp.to_torch(rays.cam_idx, requires_grad=False)

        # pack_info, ray_idx, ray_intervals = model.grid.uniform_ray_samples(ray_ori, ray_dir, t_min, t_max, step_size)
        ray_intervals = model.grid.uniform_ray_samples(
            ray_origins=ray_ori,
            ray_directions=ray_dir,
            t_min=t_min,
            t_max=t_max,
            step_size=model.step_size
        )

        ray_idx = ray_intervals.jidx.int() #ray_idx.jdata

        samples = SampleBatch()
        
        samples.offsets = ray_intervals.joffsets
        n_samples = ray_intervals.joffsets[1:] - ray_intervals.joffsets[:-1]
        samples.count = ray_intervals.jdata.shape[0]

        #last element is the number of samples in the last ray
        samples.n_samples = torch.cat([
            n_samples,
            torch.tensor([samples.count - samples.offsets[-1]]).to(n_samples.device)
        ])

        samples.start = ray_intervals.jdata[:, 0]
        samples.end = ray_intervals.jdata[:, 1]
        samples.dt = (samples.end - samples.start).contiguous()

        t = ray_intervals.jdata.mean(dim=1)
        if stratify:
            samples.t = t + samples.dt * (torch.rand_like(t) - 0.5)
        else:
            samples.t = t
        
        samples.xyz = ray_ori[ray_idx] + ray_dir[ray_idx] * samples.t[:, None]
        samples.dir = ray_dir[ray_idx]
        samples.ray_idx = ray_idx
        samples.cam_idx = cam_idx[ray_idx]

        return samples

def query_samples(model: WarpNeRFModel, samples: SampleBatch) -> SampleBatch:

    # query density
    samples.sigma, geo_feat = model.query_sigma(samples.xyz, return_feat=True)
    
    # query color
    samples.rgb = model.query_rgb(samples.dir, geo_feat, samples.cam_idx)

    # scale dt
    # samples.dt = samples.dt / model.aabb_scale

    return samples

def render_samples(samples: SampleBatch) -> tuple[Tensor, Tensor, Tensor]:
    # volume render
    ray_rgb, ray_depth, ray_opacity, ray_w, ray_num_samples = fvdb.volume_render(
        sigmas=samples.sigma,
        rgbs=samples.rgb,
        deltaTs=samples.dt,
        ts=samples.t,
        packInfo=samples.offsets,
        transmittanceThresh=1e-5,
    )

    return ray_rgb, ray_depth, ray_opacity
