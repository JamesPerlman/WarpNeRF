import warp as wp
import numpy as np


from pathlib import Path

from warpnerf.models.batch import RayBatch, create_ray_batch
from warpnerf.training.batcher import init_training_rays_and_pixels_kernel
from warpnerf.training.dataset import Dataset
from warpnerf.utils.image import save_image

wp.init()

# @wp.func
# def vec3f_lte(a: wp.vec3f, b: wp.vec3f) -> wp.bool:
#     return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]

# @wp.func
# def vec3f_gte(a: wp.vec3f, b: wp.vec3f) -> wp.bool:
#     return a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]

@wp.struct
class BoundingBox:
    min: wp.vec3
    max: wp.vec3

# @wp.func
# def bbox_contains_point(bbox: BoundingBox, point: wp.vec3) -> wp.bool:
#     return vec3f_lte(bbox.min, point) and vec3f_gte(bbox.max, point)

# @wp.kernel
# def test_kernel(bbox: BoundingBox, point: wp.vec3, result: wp.array(dtype=wp.bool)):
#     result[0] = bbox_contains_point(bbox, point)

a = wp.vec3f(1.0, 2.0, 3.0)
b = wp.vec3f(2.0, 3.0, 4.0)

c = wp.zeros(1, dtype=wp.bool)
bbox = BoundingBox()
bbox.min = a
bbox.max = b

p = wp.vec3f(2.5, 2.5, 3.5)

# wp.launch(test_kernel, dim=1, inputs=[bbox, p], outputs=[c], device="cuda")

print(c)

dataset = Dataset(path=Path("E:\\nerfs\\test\\turb1"))
dataset.load()

ray_batch = create_ray_batch(count=1000)
rgba_batch = wp.empty((4, 1000), dtype=wp.uint8, device="cuda")

random_seed = 1337
wp.launch(
    init_training_rays_and_pixels_kernel,
    dim=1000,
    inputs=[random_seed, dataset.num_images, dataset.image_dims, dataset.camera_data, dataset.image_data],
    outputs=[ray_batch, rgba_batch],
)

wp.synchronize()
print(ray_batch.ray)
print(rgba_batch)
