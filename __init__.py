import warp as wp
import numpy as np


from pathlib import Path

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

dataset = Dataset(path=Path("E:\\nerfs\\test\\sara3"))
dataset.load()

print(dataset.num_cameras)

img = dataset.training_cameras[0].image

@wp.kernel
def img_invert_kernel(img: wp.array3d(dtype=wp.float32), result: wp.array3d(dtype=wp.float32)):
    i, j = wp.tid()
    result[i][j][0] = 1.0 - img[i][j][0]
    result[i][j][1] = 1.0 - img[i][j][1]
    result[i][j][2] = 1.0 - img[i][j][2]
    
result = wp.ones_like(img)
wp.launch(img_invert_kernel, dim=(img.shape[0], img.shape[1]), inputs=[img], outputs=[result], device="cuda")

save_image(result, Path("E:\\nerfs\\test\\test.png"))