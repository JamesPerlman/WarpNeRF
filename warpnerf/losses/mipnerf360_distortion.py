import warp as wp

@wp.kernel
def mipNeRF360_distortion_loss_kernel(
    # input buffers
    # per ray
    ray_n_samples: wp.array1d(dtype=wp.int32),
    ray_sample_offset: wp.array1d(dtype=wp.int32),

    # per sample
    sample_density: wp.array1d(dtype=wp.float32),
    sample_m_norm: wp.array1d(dtype=wp.float32),
    sample_dt_norm: wp.array1d(dtype=wp.float32),

    # output buffers
    # per ray
    ray_distortion_loss: wp.array1d(dtype=wp.float32),
):
    ray_idx = wp.tid()

    n_samples = ray_n_samples[ray_idx]

    if n_samples == 0:
        ray_distortion_loss[ray_idx] = 0.0
        return
    
    sample_offset = ray_sample_offset[ray_idx]

    for i in range(n_samples):
        dt = sample_dt_norm[sample_offset + i]
        density = sample_density[sample_offset + i]

