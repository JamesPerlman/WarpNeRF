import warp as wp

from warpnerf.models.batch import Ray, RayBatch
from warpnerf.models.camera import CameraData

@wp.func
def get_global_ray_at_pixel_xy(
    camera: CameraData,
    img_w: wp.int32,
    img_h: wp.int32,
    pixel_x: wp.int32,
    pixel_y: wp.int32,
    near: wp.float32 = 0.05,
) -> Ray:
    wf = wp.float32(img_w)
    hf = wp.float32(img_h)
    xf = wp.float32(pixel_x)
    yf = wp.float32(pixel_y)

    v = wp.vec3f(
        ((xf + 0.5) / wf - 0.5) * camera.sx / camera.f,
        ((yf + 0.5) / hf - 0.5) * camera.sy / camera.f,
        1.0
    )

    v_len = wp.length(v)
    v_norm = wp.div(v, v_len)

    ray = Ray()
    ray.dir = wp.normalize(wp.mul(camera.R, v))
    ray.ori = camera.t + (wp.mul(near * v_len, ray.dir))
    ray.cos = v_norm.z

    # radius is half way between the apothem and circumradius of the pixel
    pixel_w = camera.sx / wf
    pixel_h = camera.sy / hf
    pixel_circumradius = wp.sqrt(pixel_w * pixel_w + pixel_h * pixel_h) / 2.0
    pixel_avg_apothem = (pixel_w + pixel_h) / 4.0
    ray.radius = 0.5 * (pixel_avg_apothem + pixel_circumradius)

    return ray


@wp.kernel
def get_rays_for_camera_kernel(
    camera: CameraData,
    img_w: wp.int32,
    img_h: wp.int32,
    rays_out: RayBatch,
):
    i, j = wp.tid()
    idx = wp.tid()

    ray = get_global_ray_at_pixel_xy(camera, img_w, img_h, i, j)

    rays_out.ori[idx] = ray.ori
    rays_out.dir[idx] = ray.dir
    rays_out.cos[idx] = ray.cos
    rays_out.radius[idx] = ray.radius
