import warp as wp

from warpnerf.models.batch import Ray
from warpnerf.models.camera import CameraData

@wp.func
def get_global_ray_at_pixel_xy(
    camera: CameraData,
    img_w: wp.int32,
    img_h: wp.int32,
    pixel_x: wp.int32,
    pixel_y: wp.int32,
    near: wp.float32 = 0.0,
) -> Ray:
    v = wp.vec3f(
        (wp.float32(pixel_x) - wp.float32(img_w) / 2.0 + 0.5) / camera.f,
        (wp.float32(pixel_y) - wp.float32(img_h) / 2.0 + 0.5) / camera.f,
        1.0
    )

    v_len = wp.length(v)
    v_norm = wp.div(v, v_len)

    ray = Ray()
    ray.dir = wp.normalize(wp.mul(camera.R, v))
    ray.ori = camera.t + (wp.mul(near * v_len, ray.dir))
    ray.cos = v_norm.z

    # radius is half way between the apothem and circumradius of the pixel
    pixel_side_len = 1.0 / (camera.f)
    pixel_apothem = 0.5 * pixel_side_len
    pixel_circumradius = wp.sqrt(2) * pixel_apothem 
    ray.radius = 0.5 * (pixel_apothem + pixel_circumradius)

    return ray
