from typing import Tuple
import warp as wp

from warpnerf.models.batch import Ray, RayBatch
from warpnerf.models.camera import CameraData

# adapted from Nerfies!  Thanks Google!
# https://github.com/google/nerfies/blob/main/nerfies/camera.py#L26

@wp.func
def compute_residual_and_jacobian(
    camera: CameraData,
    x: wp.float32,
    y: wp.float32,
    xd: wp.float32,
    yd: wp.float32,
) -> Tuple[wp.float32, wp.float32, wp.float32, wp.float32, wp.float32, wp.float32]:

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    r = x * x + y * y
    d = 1.0 + r * (camera.k1 + r * camera.k2)

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);

    # Let's define
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;

    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    
    fx = d * x + 2.0 * camera.p1 * x * y + camera.p2 * (r + 2.0 * x * x) - xd
    fy = d * y + 2.0 * camera.p2 * x * y + camera.p1 * (r + 2.0 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = camera.k1 + r * (2.0 * camera.k2)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * camera.p1 * y + 6.0 * camera.p2 * x
    fx_y = d_y * x + 2.0 * camera.p1 * x + 2.0 * camera.p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * camera.p2 * y + 2.0 * camera.p1 * x
    fy_y = d + d_y * y + 2.0 * camera.p2 * x + 6.0 * camera.p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y

@wp.func
def radial_and_tangential_undistort(
    camera: CameraData,
    xd: wp.float32,
    yd: wp.float32,
) -> Tuple[wp.float32, wp.float32]:
    eps = 1e-9

    # initial guess
    xu = xd
    yu = yd

    # Newton's method
    for _ in range(10):
        
        fx, fy, fx_x, fx_y, fy_x, fy_y = compute_residual_and_jacobian(camera, xu, yu, xd, yd)

        # compute the Jacobian
        det =  fx_y * fy_x - fx_x * fy_y
        if wp.abs(det) < eps:
            break

        # compute the update
        dx = (fx * fy_y - fy * fx_y) / det
        dy = (fy * fx_x - fx * fy_x) / det

        # Update the solution.
        xu += dx
        yu += dy

        # Check for convergence.
        if wp.abs(dx) < eps and wp.abs(dy) < eps:
            break
    
    return xu, yu

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
    
    x = pixel_x + 0.5
    y = pixel_y + 0.5
    # undistort if necessary
    if camera.k1 != 0.0 or camera.k2 != 0.0 or camera.p1 != 0.0 or camera.p2 != 0.0:
        xd = (x - camera.cx) / camera.f
        yd = (y - camera.cy) / camera.f

        xu, yu = radial_and_tangential_undistort(camera, xd, yd)

        x = camera.f * xu + camera.cx
        y = camera.f * yu + camera.cy

    v = wp.vec3f(
        (x / wf - 0.5) * camera.sx / camera.f,
        (y / hf - 0.5) * camera.sy / camera.f,
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
    visibilities[idx] = 1
    return
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
