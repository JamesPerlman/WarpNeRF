import torch
import warp as wp

from warpnerf.utils.volume_rendering import sigma_to_alpha

# kind of a hack here, we're not really using warp's differentiability here
# instead, we are just using it for the JIT compilation into a CUDA kernel for forward and backward
# I believe this means we can't use this in a warp graph
# but that's okay since we're just going to wrap it in a PyTorch autograd function
@wp.kernel(enable_backward=False)
def mipNeRF360_distortion_loss_forward_kernel(
    # constants
    distortion_loss_lambda: wp.float32,

    # input buffers

    # per ray
    ray_n_samples: wp.array1d(dtype=wp.int64),
    ray_sample_offset: wp.array1d(dtype=wp.int64),

    # per sample
    sample_m_norm: wp.array1d(dtype=wp.float32),
    sample_dt_norm: wp.array1d(dtype=wp.float32),
    sample_sigma: wp.array1d(dtype=wp.float32),

    # output buffers

    # per ray
    ray_dtw2_cs: wp.array1d(dtype=wp.float32),
    ray_w_cs: wp.array1d(dtype=wp.float32),
    ray_wm_cs: wp.array1d(dtype=wp.float32),
    ray_wm_w_cs1_cs: wp.array1d(dtype=wp.float32),
    ray_w_wm_cs1_cs: wp.array1d(dtype=wp.float32),
    ray_distortion_loss: wp.array1d(dtype=wp.float32), # this is the only differentiable output we care about

    # per sample
    sample_w_cs: wp.array1d(dtype=wp.float32),
    sample_wm_cs: wp.array1d(dtype=wp.float32),
):
    ray_idx = wp.tid()

    n_samples = ray_n_samples[ray_idx]

    if n_samples == 0:
        ray_distortion_loss[ray_idx] = 0.0
        ray_w_cs[ray_idx] = 0.0
        ray_wm_cs[ray_idx] = 0.0
        ray_dtw2_cs[ray_idx] = 0.0
        ray_wm_w_cs1_cs[ray_idx] = 0.0
        ray_w_wm_cs1_cs[ray_idx] = 0.0
        return
    
    sample_offset = ray_sample_offset[ray_idx]

    cumsum_w = wp.float32(0.0)
    cumsum_wm = wp.float32(0.0)
    cumsum_dtw2 = wp.float32(0.0)
    cumsum_wm_w_cs1 = wp.float32(0.0)
    cumsum_w_wm_cs1 = wp.float32(0.0)

    trans = wp.float32(1.0)
    prev_w_cs = wp.float32(0.0)
    prev_wm_cs = wp.float32(0.0)

    loss_bi_0 = wp.float32(0.0)
    loss_bi_1 = wp.float32(0.0)
    loss_uni = wp.float32(0.0)

    for i in range(int(sample_offset), int(sample_offset + n_samples)):
        dt = sample_dt_norm[i]
        sigma = sample_sigma[i]
        alpha = sigma_to_alpha(sigma, dt)
        weight = alpha * trans
        m = sample_m_norm[i]
        wm = weight * m

        wm_w_cs_1 = wm * prev_w_cs
        w_wm_cs_1 = weight * prev_wm_cs
        dtw2 = dt * weight * weight

        cumsum_w += weight
        cumsum_wm += wm
        cumsum_dtw2 += dtw2
        cumsum_wm_w_cs1 += wm_w_cs_1
        cumsum_w_wm_cs1 += w_wm_cs_1

        sample_w_cs[i] = cumsum_w
        sample_wm_cs[i] = cumsum_wm

        loss_bi_0 += wm_w_cs_1
        loss_bi_1 += w_wm_cs_1
        loss_uni += dtw2

        prev_w_cs = cumsum_w
        prev_wm_cs = cumsum_wm

        trans *= (1.0 - alpha)

    k = distortion_loss_lambda / wp.float32(n_samples)
    ray_distortion_loss[ray_idx] = k * ((1.0 / 3.0) * loss_uni + 2.0 * (loss_bi_0 - loss_bi_1))
    ray_w_cs[ray_idx] = cumsum_w
    ray_wm_cs[ray_idx] = cumsum_wm
    ray_dtw2_cs[ray_idx] = cumsum_dtw2
    ray_wm_w_cs1_cs[ray_idx] = cumsum_wm_w_cs1
    ray_w_wm_cs1_cs[ray_idx] = cumsum_w_wm_cs1


@wp.kernel(enable_backward=False)
def mipNeRF360_distortion_loss_backward_kernel(
    # constants
    distortion_loss_lambda: wp.float32,

    # input buffers

    # per ray
    ray_n_samples: wp.array1d(dtype=wp.int64),
    ray_sample_offset: wp.array1d(dtype=wp.int64),
    ray_dtw2_cs: wp.array1d(dtype=wp.float32),
    ray_w_cs: wp.array1d(dtype=wp.float32),
    ray_wm_cs: wp.array1d(dtype=wp.float32),
    ray_wm_w_cs1_cs: wp.array1d(dtype=wp.float32),
    ray_w_wm_cs1_cs: wp.array1d(dtype=wp.float32),

    # per sample
    sample_m_norm: wp.array1d(dtype=wp.float32),
    sample_dt_norm: wp.array1d(dtype=wp.float32),
    sample_sigma: wp.array1d(dtype=wp.float32),
    sample_w_cs: wp.array1d(dtype=wp.float32),
    sample_wm_cs: wp.array1d(dtype=wp.float32),

    # output buffer
    sample_dloss_dsigma: wp.array1d(dtype=wp.float32),
):
    ray_idx = wp.tid()

    n_samples = ray_n_samples[ray_idx]

    if n_samples == 0:
        return
    
    sample_offset = ray_sample_offset[ray_idx]

    cumsum_w = wp.float32(ray_w_cs[ray_idx])
    cumsum_wm = wp.float32(ray_wm_cs[ray_idx])
    cumsum_dtw2 = wp.float32(ray_dtw2_cs[ray_idx])
    cumsum_wm_w_cs1 = wp.float32(ray_wm_w_cs1_cs[ray_idx])
    cumsum_w_wm_cs1 = wp.float32(ray_w_wm_cs1_cs[ray_idx])

    trans = wp.float32(1.0)
    prev_w_cs = wp.float32(0.0)
    prev_wm_cs = wp.float32(0.0)

    k = distortion_loss_lambda / wp.float32(n_samples)

    for i in range(int(sample_offset), int(sample_offset + n_samples)):
        dt = sample_dt_norm[i]
        sigma = sample_sigma[i]
        alpha = sigma_to_alpha(sigma, dt)
        weight = alpha * trans
        m = sample_m_norm[i]
        wm = weight * m
        w_cs = sample_w_cs[i]
        wm_cs = sample_wm_cs[i]

        dalpha_dsigma = dt * (1.0 - alpha)
        dw_dsigma = trans * dalpha_dsigma
        dwm_dsigma = m * dw_dsigma
        t_w = trans - weight

        cumsum_w -= weight
        cumsum_wm -= wm
        cumsum_dtw2 -= dt * weight * weight
        cumsum_wm_w_cs1 -= wm * prev_w_cs
        cumsum_w_wm_cs1 -= weight * prev_wm_cs

        dloss_bi_0_dsigma = dwm_dsigma * prev_w_cs + dt * (cumsum_wm * (w_cs + t_w) - 2.0 * cumsum_wm_w_cs1)
        dloss_bi_1_dsigma = dw_dsigma * prev_wm_cs + dt * (cumsum_w * (m * t_w + wm_cs) - 2.0 * cumsum_w_wm_cs1)
        dloss_uni_dsigma = 2.0 * dt * (dt * weight * t_w - cumsum_dtw2)

        sample_dloss_dsigma[i] = k * ((dloss_bi_0_dsigma - dloss_bi_1_dsigma) + (1.0 / 3.0) * dloss_uni_dsigma)

        prev_w_cs = w_cs
        prev_wm_cs = wm_cs

        trans *= (1.0 - alpha)

# PyTorch autograd function wrapper

class MipNeRF360DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        distortion_loss_lambda: float,
        ray_n_samples: torch.Tensor,
        ray_sample_offset: torch.Tensor,
        sample_m_norm: torch.Tensor,
        sample_dt_norm: torch.Tensor,
        sample_sigma: torch.Tensor,
    ):
        ctx.distortion_loss_lambda = distortion_loss_lambda
        ctx.ray_n_samples = wp.from_torch(ray_n_samples)
        ctx.ray_sample_offset = wp.from_torch(ray_sample_offset)
        ctx.sample_m_norm = wp.from_torch(sample_m_norm)
        ctx.sample_dt_norm = wp.from_torch(sample_dt_norm)
        ctx.sample_sigma = wp.from_torch(sample_sigma)
        ctx.save_for_backward(sample_sigma)

        device = ctx.ray_n_samples.device
        ctx.n_rays = ray_n_samples.shape[0]
        ctx.n_samples = sample_sigma.shape[0]

        ctx.ray_dtw2_cs = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=False)
        ctx.ray_w_cs = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=False)
        ctx.ray_wm_cs = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=False)
        ctx.ray_wm_w_cs1_cs = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=False)
        ctx.ray_w_wm_cs1_cs = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=False)
        ctx.ray_distortion_loss = wp.empty(shape=(ctx.n_rays), dtype=wp.float32, device=device, requires_grad=True)

        ctx.sample_w_cs = wp.empty(shape=(ctx.n_samples), dtype=wp.float32, device=device, requires_grad=False)
        ctx.sample_wm_cs = wp.empty(shape=(ctx.n_samples), dtype=wp.float32, device=device, requires_grad=False)

        wp.launch(
            kernel=mipNeRF360_distortion_loss_forward_kernel,
            dim=ctx.n_rays,
            inputs=[
                ctx.distortion_loss_lambda,
                ctx.ray_n_samples,
                ctx.ray_sample_offset,
                ctx.sample_m_norm,
                ctx.sample_dt_norm,
                ctx.sample_sigma,
            ],
            outputs=[
                ctx.ray_dtw2_cs,
                ctx.ray_w_cs,
                ctx.ray_wm_cs,
                ctx.ray_wm_w_cs1_cs,
                ctx.ray_w_wm_cs1_cs,
                ctx.ray_distortion_loss,
                ctx.sample_w_cs,
                ctx.sample_wm_cs,
            ]
        )
        
        # just return the distortion loss
        return torch.mean(wp.to_torch(ctx.ray_distortion_loss))

    @staticmethod
    def backward(ctx, grad_output):

        sigma_grad = wp.empty_like(ctx.sample_sigma)

        wp.launch(
            kernel=mipNeRF360_distortion_loss_backward_kernel,
            dim=ctx.n_rays,
            inputs=[
                ctx.distortion_loss_lambda,
                ctx.ray_n_samples,
                ctx.ray_sample_offset,
                ctx.ray_dtw2_cs,
                ctx.ray_w_cs,
                ctx.ray_wm_cs,
                ctx.ray_wm_w_cs1_cs,
                ctx.ray_w_wm_cs1_cs,
                ctx.sample_m_norm,
                ctx.sample_dt_norm,
                ctx.sample_sigma,
                ctx.sample_w_cs,
                ctx.sample_wm_cs,
            ],
            outputs=[
                sigma_grad,
            ]
        )

        sample_sigma_grad = grad_output * wp.to_torch(sigma_grad)
        return None, None, None, None, None, sample_sigma_grad

