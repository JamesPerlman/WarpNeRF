import warp as wp

@wp.struct
class Ray:
    ori: wp.vec3f
    dir: wp.vec3f

@wp.struct
class RayBatch:
    count: wp.int32
    ray: wp.array1d(dtype=Ray)
    alive: wp.array1d(dtype=wp.bool)
    t: wp.array1d(dtype=wp.float32)

def create_ray_batch(count: int, device: str = "cuda") -> RayBatch:
    batch = RayBatch()
    
    batch.count = count
    batch.ray = wp.empty(shape=(count), dtype=Ray, device=device)
    batch.alive = wp.zeros(shape=(count), dtype=wp.bool, device=device)
    batch.t = wp.zeros(shape=(count), dtype=wp.float32, device=device)

    return batch

@wp.struct
class SampleBatch:
    xyz: wp.array1d(dtype=wp.vec3f)
    n_samples: wp.array1d(dtype=wp.int32)
