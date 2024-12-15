import fvdb
import warp as wp
from torch import Tensor, LongTensor

@wp.struct
class Ray:
    ori: wp.vec3f
    dir: wp.vec3f

@wp.struct
class RayBatch:
    count: wp.int32
    dir: wp.array1d(dtype=wp.vec3f)
    ori: wp.array1d(dtype=wp.vec3f)
    # cos: wp.array1d(dtype=wp.float32)
    # radius: wp.array1d(dtype=wp.float32)
    alive: wp.array1d(dtype=wp.bool)
    t: wp.array1d(dtype=wp.float32)
    cam_idx: wp.array1d(dtype=wp.int32)

def create_ray_batch(count: int, device: str = "cuda") -> RayBatch:
    batch = RayBatch()
    
    batch.count = count
    batch.dir = wp.empty(shape=(count), dtype=wp.vec3f, device=device)
    batch.ori = wp.empty(shape=(count), dtype=wp.vec3f, device=device)
    # batch.cos = wp.empty(shape=(count), dtype=wp.float32, device=device)
    # batch.radius = wp.empty(shape=(count), dtype=wp.float32, device=device)
    # batch.alive = wp.zeros(shape=(count), dtype=wp.bool, device=device)
    batch.t = wp.zeros(shape=(count), dtype=wp.float32, device=device)
    batch.cam_idx = wp.empty(shape=(count), dtype=wp.int32, device=device)

    return batch

# ew, mixing torch and warp
class SampleBatch:
    count: int
    t: Tensor
    dt: Tensor
    xyz: Tensor
    start: Tensor
    end: Tensor
    dir: Tensor
    offsets: LongTensor
    n_samples: LongTensor
    rgb: Tensor
    sigma: Tensor
    geo_feat: Tensor
    ray_idx: Tensor
    cam_idx: Tensor
