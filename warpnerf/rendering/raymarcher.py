import torch
import warp as wp

from warpnerf.models.batch import RayBatch, SampleBatch
import fvdb

@wp.struct
class RaymarcherOptions:
    step_size: wp.float32

@wp.struct
class BoundingBox:
    min: wp.vec3
    max: wp.vec3

@wp.func
def is_point_inside_bbox(point: wp.vec3, bbox: BoundingBox) -> wp.bool:
    return all(wp.logical_and(point >= bbox.min, point <= bbox.max))

@wp.kernel
def march_and_count_steps_per_ray_kernel(
    options: RaymarcherOptions,
    bounding_box: BoundingBox,
    rays_in: RayBatch,
    samples_out: SampleBatch
):
    idx = wp.tid()

class Raymarcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch: RayBatch):
        return torch.zeros(batch.shape[0], 3)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)
    