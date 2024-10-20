import fvdb
import warp as wp

@wp.struct
class Ray:
    ori: wp.vec3f
    dir: wp.vec3f
    cos: wp.float32
    radius: wp.float32

@wp.struct
class RayBatch:
    count: wp.int32
    dir: wp.array1d(dtype=wp.vec3f)
    ori: wp.array1d(dtype=wp.vec3f)
    cos: wp.array1d(dtype=wp.float32)
    radius: wp.array1d(dtype=wp.float32)
    alive: wp.array1d(dtype=wp.bool)
    t: wp.array1d(dtype=wp.float32)

def create_ray_batch(count: int, device: str = "cuda") -> RayBatch:
    batch = RayBatch()
    
    batch.count = count
    batch.dir = wp.empty(shape=(count), dtype=wp.vec3f, device=device)
    batch.ori = wp.empty(shape=(count), dtype=wp.vec3f, device=device)
    batch.alive = wp.zeros(shape=(count), dtype=wp.bool, device=device)
    batch.t = wp.zeros(shape=(count), dtype=wp.float32, device=device)

    return batch

@wp.struct
class SampleBatch:
    t: wp.array1d(dtype=wp.float32)
    dt: wp.array1d(dtype=wp.float32)
    xyz: wp.array1d(dtype=wp.vec3f)
    offset: wp.array1d(dtype=wp.int32)
