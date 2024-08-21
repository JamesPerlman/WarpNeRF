import warp as wp

@wp.struct
class Ray:
    ori: wp.vec3f
    dir: wp.vec3f

@wp.struct
class RayBatch:
    count: wp.int32
    alive: wp.array1d(dtype=wp.bool)
    ori: wp.array1d(dtype=wp.vec3f)
    dir: wp.array1d(dtype=wp.vec3f)
    t: wp.array1d(dtype=wp.float32)

def create_ray_batch(count: int) -> RayBatch:
    batch = RayBatch()
    
    batch.count = count
    batch.alive = wp.ones(shape=(count), dtype=wp.bool)
    batch.ori = wp.zeros(shape=(count), dtype=wp.vec3f)
    batch.dir = wp.zeros(shape=(count), dtype=wp.vec3f)
    batch.t = wp.zeros(shape=(count), dtype=wp.float32)

@wp.struct
class SampleBatch:
    xyz: wp.array1d(dtype=wp.vec3f)
    n_samples: wp.array1d(dtype=wp.int32)
