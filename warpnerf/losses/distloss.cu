
__global__ void mipNeRF360_distortion_loss_forward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,

    // input buffers
    // per ray
    const uint32_t* __restrict__ ray_steps,
    const uint32_t* __restrict__ ray_offset,

    // per sample
    const tcnn::network_precision_t* __restrict__ sample_density_buf,
    const float* __restrict__ sample_m_norm_buf,
    const float* __restrict__ sample_dt_norm_buf,

    // output buffers 
    // per ray
    float* __restrict__ ray_dtw2_cs_buf,
    float* __restrict__ ray_w_cs_buf,
    float* __restrict__ ray_wm_cs_buf,
    float* __restrict__ ray_wm_w_cs1_cs_buf,
    float* __restrict__ ray_w_wm_cs1_cs_buf,
    float* __restrict__ ray_dist_loss_buf,

    // per sample
    float* __restrict__ sample_w_cs_buf,
    float* __restrict__ sample_wm_cs_buf
) {

    const uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    const uint32_t n_samples = ray_steps[ray_idx];

    if (n_samples == 0) {
        ray_dist_loss_buf[ray_idx] = 0.0f;
        ray_w_cs_buf[ray_idx] = 0.0f;
        ray_wm_cs_buf[ray_idx] = 0.0f;
        ray_dtw2_cs_buf[ray_idx] = 0.0f;
        ray_wm_w_cs1_cs_buf[ray_idx] = 0.0f;
        ray_w_wm_cs1_cs_buf[ray_idx] = 0.0f;
        return;
    }

    const uint32_t sample_offset = ray_offset[ray_idx];

    const tcnn::network_precision_t* density = sample_density_buf + sample_offset;
    const float* dt = sample_dt_norm_buf + sample_offset;
    const float* m = sample_m_norm_buf + sample_offset;

    float* w_cs = sample_w_cs_buf + sample_offset;
    float* wm_cs = sample_wm_cs_buf + sample_offset;

    float cumsum_w = 0.0f;
    float cumsum_wm = 0.0f;
    float cumsum_dtw2 = 0.0f;
    float cumsum_wm_w_cs1 = 0.0f;
    float cumsum_w_wm_cs1 = 0.0f;

    float trans = 1.0f;
    float prev_w_cs = 0.0f;
    float prev_wm_cs = 0.0f;

    float loss_bi_0 = 0.0f;
    float loss_bi_1 = 0.0f;
    float loss_uni = 0.0f;

    for (uint32_t i = 0; i < n_samples; ++i) {
        const float dt_i = dt[i];
        const float sigma_i = density_to_sigma(density[i]);
        const float alpha_i = sigma_to_alpha(sigma_i, dt_i);
        const float trans_i = trans;
        const float weight_i = alpha_i * trans_i;
        const float m_i = m[i];
        const float wm_i = weight_i * m_i;

        const float wm_w_cs_1_i = wm_i * prev_w_cs;
        const float w_wm_cs_1_i = weight_i * prev_wm_cs;
        const float dtw2_i = dt_i * weight_i * weight_i;

        cumsum_w += weight_i;
        cumsum_wm += wm_i;
        cumsum_dtw2 += dtw2_i;
        cumsum_wm_w_cs1 += wm_w_cs_1_i;
        cumsum_w_wm_cs1 += w_wm_cs_1_i;

        w_cs[i] = cumsum_w;
        wm_cs[i] = cumsum_wm;

        loss_bi_0 += wm_w_cs_1_i;
        loss_bi_1 += w_wm_cs_1_i;
        loss_uni += dtw2_i;

        prev_w_cs = cumsum_w;
        prev_wm_cs = cumsum_wm;

        trans *= (1.0f - alpha_i);
    }

    const float k = NeRFConstants::mipNeRF360_distortion_loss_lambda / (float)n_samples;
    ray_dist_loss_buf[ray_idx] = k * ((1.0 / 3.0) * loss_uni + 2.0 * (loss_bi_0 - loss_bi_1));
    ray_w_cs_buf[ray_idx] = cumsum_w;
    ray_wm_cs_buf[ray_idx] = cumsum_wm;
    ray_dtw2_cs_buf[ray_idx] = cumsum_dtw2;
    ray_wm_w_cs1_cs_buf[ray_idx] = cumsum_wm_w_cs1;
    ray_w_wm_cs1_cs_buf[ray_idx] = cumsum_w_wm_cs1;
}


__global__ void mipNeRF360_distortion_loss_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,

    // input buffers
    // per ray
    const uint32_t* __restrict__ ray_steps,
    const uint32_t* __restrict__ ray_offset,
    // per sample
    const tcnn::network_precision_t* __restrict__ sample_density_buf,

    // output buffers
    // per ray
    const float* __restrict__ ray_dtw2_cs_buf,
    const float* __restrict__ ray_w_cs_buf,
    const float* __restrict__ ray_wm_cs_buf,
    const float* __restrict__ ray_wm_w_cs1_cs_buf,
    const float* __restrict__ ray_w_wm_cs1_cs_buf,
    const float* __restrict__ ray_dist_loss_buf,
    
    // per sample
    const float* __restrict__ sample_m_norm_buf,
    const float* __restrict__ sample_dt_norm_buf,
    const float* __restrict__ sample_w_cs_buf,
    const float* __restrict__ sample_wm_cs_buf,

    // output buffer
    float* __restrict__ sample_dloss_ddensity_buf
) {

    const uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    const uint32_t n_samples = ray_steps[ray_idx];
    
    if (n_samples == 0) {
        return;
    }

    const uint32_t sample_offset = ray_offset[ray_idx];

    const tcnn::network_precision_t* density = sample_density_buf + sample_offset;
    const float* dt = sample_dt_norm_buf + sample_offset;
    const float* m = sample_m_norm_buf + sample_offset;
    const float* w_cs = sample_w_cs_buf + sample_offset;
    const float* wm_cs = sample_wm_cs_buf + sample_offset;

    float* sample_dloss_ddensity = sample_dloss_ddensity_buf + sample_offset;

    float cumsum_w = ray_w_cs_buf[ray_idx];
    float cumsum_wm = ray_wm_cs_buf[ray_idx];
    float cumsum_dtw2 = ray_dtw2_cs_buf[ray_idx];
    float cumsum_wm_w_cs1 = ray_wm_w_cs1_cs_buf[ray_idx];
    float cumsum_w_wm_cs1 = ray_w_wm_cs1_cs_buf[ray_idx];

    float trans = 1.0f;
    float prev_w_cs = 0.0f;
    float prev_wm_cs = 0.0f;

    const float n_samples_f = (float)n_samples;

    const float k = NeRFConstants::mipNeRF360_distortion_loss_lambda / n_samples_f;

    for (uint32_t i = 0; i < n_samples; ++i) {
        const float dt_i = dt[i];
        const float sigma_i = density_to_sigma(density[i]);
        const float alpha_i = sigma_to_alpha(sigma_i, dt_i);
        const float trans_i = trans;
        const float weight_i = alpha_i * trans_i;
        const float m_i = m[i];
        const float wm_i = weight_i * m_i;
        const float w_cs_i = w_cs[i];
        const float wm_cs_i = wm_cs[i];

        const float dalpha_dsigma_i = dt_i * (1.0f - alpha_i);
        const float dw_dsigma_i = trans_i * dalpha_dsigma_i;
        const float dwm_dsigma_i = m_i * dw_dsigma_i;
        const float t_w_i = trans_i - weight_i;

        cumsum_w -= weight_i;
        cumsum_wm -= wm_i;
        cumsum_dtw2 -= dt_i * weight_i * weight_i;
        cumsum_wm_w_cs1 -= wm_i * prev_w_cs;
        cumsum_w_wm_cs1 -= weight_i * prev_wm_cs;
        
        const float dloss_bi_0_dsigma = dwm_dsigma_i * prev_w_cs + dt_i * (cumsum_wm * (w_cs_i + t_w_i) - 2.0f * cumsum_wm_w_cs1);
        const float dloss_bi_1_dsigma = dw_dsigma_i * prev_wm_cs + dt_i * (cumsum_w * (m_i * t_w_i + wm_cs_i) - 2.0f * cumsum_w_wm_cs1);
        const float dloss_uni_dsigma = 2.0f * dt_i * (dt_i * weight_i * t_w_i - cumsum_dtw2);

        const float dloss_dsigma_i = k * (2.0f * (dloss_bi_0_dsigma - dloss_bi_1_dsigma) + (1.0f / 3.0f) * dloss_uni_dsigma);
        const float dsigma_ddensity_i = sigma_i;

        sample_dloss_ddensity[i] = dloss_dsigma_i * dsigma_ddensity_i;

        prev_w_cs = w_cs_i;
        prev_wm_cs = wm_cs_i;

        trans *= (1.0f - alpha_i);
    }
}