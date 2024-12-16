import warp as wp

from warpnerf.models.batch import Ray, RayBatch
from warpnerf.models.camera import CameraData

@wp.func
def get_global_ray_at_pixel_xy(
    camera: CameraData,
    img_w: wp.int32,
    img_h: wp.int32,
    pixel_x: wp.float32,
    pixel_y: wp.float32,
    near: wp.float32 = 0.0,
) -> Ray:
    wf = wp.float32(img_w)
    hf = wp.float32(img_h)

    v = wp.vec3f(
        ((pixel_x + 0.5) / wf - 0.5) * camera.sx / camera.f,
        ((pixel_y + 0.5) / hf - 0.5) * camera.sy / camera.f,
        1.0
    )
    
    v_len = wp.length(v)
    # v_norm = wp.div(v, v_len)

    ray = Ray()
    ray.dir = wp.normalize(wp.mul(camera.R, v))
    ray.ori = camera.t + (wp.mul(near * v_len, ray.dir))

    return ray

    # ray.cos = v_norm.z

    # # radius is half way between the apothem and circumradius of the pixel
    # pixel_w = camera.sx / wf
    # pixel_h = camera.sy / hf
    # pixel_circumradius = wp.sqrt(pixel_w * pixel_w + pixel_h * pixel_h) / 2.0
    # pixel_avg_apothem = (pixel_w + pixel_h) / 4.0
    # ray.radius = 0.5 * (pixel_avg_apothem + pixel_circumradius)


@wp.kernel
def get_rays_for_camera_kernel(
    camera: CameraData,
    img_w: wp.int32,
    img_h: wp.int32,
    offset: wp.int32,
    rays_out: RayBatch,
):
    idx = wp.tid()
    i = idx + offset
    px = wp.float32(i // img_h)
    py = wp.float32(i % img_h)
    ray = get_global_ray_at_pixel_xy(camera, img_w, img_h, px, py)

    rays_out.ori[idx] = ray.ori
    rays_out.dir[idx] = ray.dir
    rays_out.cam_idx[idx] = 0
    # rays_out.cos[idx] = ray.cos
    # rays_out.radius[idx] = ray.radius

@wp.kernel(enable_backward=False)
def increment_point_visibility_kernel(
    cameras: wp.array1d(dtype=CameraData),
    points: wp.array1d(dtype=wp.vec3f),
    visibilities: wp.array1d(dtype=wp.int32),
):
    idx = wp.tid()
    point = points[idx]

    for i in range(cameras.shape[0]):
        
        camera = cameras[i]
        
        # transform point to camera space
        point_cam = wp.mul(camera.R, point - camera.t)

        if point_cam.z <= 0.0:
            continue
        
        # project point to 2D

        point_2d = wp.vec2f(
            camera.f * point_cam.x / point_cam.z,
            camera.f * point_cam.y / point_cam.z
        )

        # check if point is visible
        
        if wp.abs(point_2d.x) > camera.sx or wp.abs(point_2d.y) > camera.sy:
            continue
        
        visibilities[idx] += 1
