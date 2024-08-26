import numpy as np
import warp as wp

@wp.struct
class MipEncoding3D:
    n_levels: wp.int32
    n_channels: wp.int32
    base_res: wp.int32
    base_radius: wp.float32
    data_xy: wp.array3d(dtype=wp.float32)
    data_yz: wp.array3d(dtype=wp.float32)
    data_xz: wp.array3d(dtype=wp.float32)


def create_mip_encoding3D(n_levels: wp.int32, n_channels: wp.int32, base_res: wp.int32, device: str = "cuda") -> MipEncoding3D:

    res_f = wp.float32(base_res)
    base_radius = wp.sqrt(1.0 / (wp.pi * res_f * res_f))
    
    encoding = MipEncoding3D()
    encoding.n_levels = n_levels
    encoding.n_channels = n_channels
    encoding.base_res = base_res
    encoding.base_radius = base_radius
    encoding.data_xy = wp.array(np.random.rand(n_channels, base_res, base_res), dtype=wp.float32, device=device)
    encoding.data_yz = wp.array(np.random.rand(n_channels, base_res, base_res), dtype=wp.float32, device=device)
    encoding.data_xz = wp.array(np.random.rand(n_channels, base_res, base_res), dtype=wp.float32, device=device)


@wp.func
def get_mip_level_for_radius(encoding: MipEncoding3D, radius: wp.float32) -> wp.float32:

    return wp.float32(wp.log2(encoding.base_radius / radius))


@wp.func
def get_nearest_mip2D_pixels(
    data: wp.array3d(dtype=wp.float32),
    channel: wp.int32,
    base_res: wp.int32,
    level: wp.int32,
    x: wp.float32,
    y: wp.float32
) -> wp.float32:
    
    base_res = wp.float32(base_res)
    level_res = wp.float32(base_res / (2 ** level))
    
    # Calculate the region in the base level that corresponds to the requested mip level pixel
    x_start = wp.int32(wp.clamp(x * base_res / level_res, 0.0, base_res - 1.0))
    y_start = wp.int32(wp.clamp(y * base_res / level_res, 0.0, base_res - 1.0))
    
    x_end = wp.int32(wp.clamp((x + 1.0) * base_res / level_res, 0.0, base_res))
    y_end = wp.int32(wp.clamp((y + 1.0) * base_res / level_res, 0.0, base_res))

    # Average the pixels in this region
    pixel_sum = wp.float32(0.0)
    count = (x_end - x_start) * (y_end - y_start)

    if count == 0.0:
        return 0.0

    for j in range(y_start, y_end):
        for i in range(x_start, x_end):
            pixel_sum += data[channel][j][i]

    return pixel_sum / count


@wp.func
def get_mip3D_pixel(
    encoding: MipEncoding3D,
    channel: wp.int32,
    pos: wp.vec3f,
    radius: wp.float32
) -> wp.vec3f:
    
    level = get_mip_level_for_radius(encoding, radius)
    lower_level = wp.int32(level)
    upper_level = wp.max(lower_level + 1, encoding.n_levels - 1)

    values = wp.empty(encoding.n_channels, dtype=wp.vec3f)

    for i in range(encoding.n_channels):
        lower_value = wp.vec3f(
            get_mip2D_pixel(encoding.data_xy, i, encoding.base_res, lower_level, pos.x, pos.y),
            get_mip2D_pixel(encoding.data_yz, i, encoding.base_res, lower_level, pos.y, pos.z),
            get_mip2D_pixel(encoding.data_xz, i, encoding.base_res, lower_level, pos.x, pos.z)
        )

        upper_value = wp.vec3f(
            get_mip2D_pixel(encoding.data_xy, i, encoding.base_res, lower_level, pos.x, pos.y),
            get_mip2D_pixel(encoding.data_yz, i, encoding.base_res, lower_level, pos.y, pos.z),
            get_mip2D_pixel(encoding.data_xz, i, encoding.base_res, lower_level, pos.x, pos.z)
        ) if lower_level != upper_level else lower_value

        # Interpolate between the two mip levels
        t = level - wp.float32(lower_level)
