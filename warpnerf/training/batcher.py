import warp as wp

from warpnerf.models.batch import RayBatch
from warpnerf.models.camera import CameraData
from warpnerf.utils.ray import get_global_ray_at_pixel_xy

# Batcher generates batches of data for training and evaluation.

@wp.kernel
def init_training_rays_and_pixels_kernel(
    random_seed: wp.int32,
    n_images: wp.int32,
    image_dims: wp.vec2i,
    cams_in: wp.array1d(dtype=CameraData),
    rgba_in: wp.array4d(dtype=wp.uint8),
    rays_out: RayBatch,
    rgba_out: wp.array2d(dtype=wp.uint8),
):
    i = wp.tid()

    n_rays = rays_out.count
    image_idx = (n_images * i) // n_rays
    wp.printf("i: %d, image_idx: %d\n", i, image_idx)

    img_w = image_dims[0]
    img_h = image_dims[1]
    n_pixels_per_image = img_w * img_h
    
    rand_state = wp.rand_init(random_seed)
    rand_pixel_idx = wp.randi(rand_state + wp.uint32(i), 0, n_pixels_per_image)
    pixel_x = rand_pixel_idx % img_w
    pixel_y = rand_pixel_idx // img_w
    wp.printf("pixel_x: %d, pixel_y: %d\n", pixel_x, pixel_y)

    local_ray = get_global_ray_at_pixel_xy(cams_in[image_idx], img_w, img_h, pixel_x, pixel_y, 0.0)

    rgba_out[0][i] = rgba_in[image_idx][0][pixel_x][pixel_y]
    rgba_out[1][i] = rgba_in[image_idx][1][pixel_x][pixel_y]
    rgba_out[2][i] = rgba_in[image_idx][2][pixel_x][pixel_y]
    rgba_out[3][i] = rgba_in[image_idx][3][pixel_x][pixel_y]

    rays_out.alive[i] = True
    rays_out.ori[i] = local_ray.ori
    rays_out.dir[i] = local_ray.dir
    rays_out.t[i] = 0.0

# class Batcher:
    