// Fused LAMB optimizer kernel — computes update vector + updated moments
//
// Trust ratio computation requires a global reduction (norms), so this kernel
// only computes the per-element update, m, v. The caller computes norms and
// applies the final param = param - effective_lr * update.

extern "C" __global__ void fused_lamb_f32(
    const float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ update_out,
    float beta1, float beta2, float eps, float wd,
    float bias_corr1, float bias_corr2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = grad[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * gi;
    float vi = beta2 * v[idx] + (1.0f - beta2) * gi * gi;

    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_corr1;
    float v_hat = vi / bias_corr2;
    float adam_update = m_hat / (sqrtf(v_hat) + eps);

    update_out[idx] = (wd > 0.0f) ? (adam_update + wd * param[idx]) : adam_update;
}

extern "C" __global__ void fused_lamb_f64(
    const double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ m,
    double* __restrict__ v,
    double* __restrict__ update_out,
    double beta1, double beta2, double eps, double wd,
    double bias_corr1, double bias_corr2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double gi = grad[idx];
    double mi = beta1 * m[idx] + (1.0 - beta1) * gi;
    double vi = beta2 * v[idx] + (1.0 - beta2) * gi * gi;

    m[idx] = mi;
    v[idx] = vi;

    double m_hat = mi / bias_corr1;
    double v_hat = vi / bias_corr2;
    double adam_update = m_hat / (sqrt(v_hat) + eps);

    update_out[idx] = (wd > 0.0) ? (adam_update + wd * param[idx]) : adam_update;
}

#if __CUDA_ARCH__ >= 700
#include "../dtype_traits.cuh"

extern "C" __global__ void fused_lamb_f16(
    const __half* __restrict__ param,
    const __half* __restrict__ grad,
    __half* __restrict__ m,
    __half* __restrict__ v,
    __half* __restrict__ update_out,
    float beta1, float beta2, float eps, float wd,
    float bias_corr1, float bias_corr2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __half2float(grad[idx]);
    float mi = beta1 * __half2float(m[idx]) + (1.0f - beta1) * gi;
    float vi = beta2 * __half2float(v[idx]) + (1.0f - beta2) * gi * gi;

    m[idx] = __float2half(mi);
    v[idx] = __float2half(vi);

    float m_hat = mi / bias_corr1;
    float v_hat = vi / bias_corr2;
    float adam_update = m_hat / (sqrtf(v_hat) + eps);

    float u = (wd > 0.0f) ? (adam_update + wd * __half2float(param[idx])) : adam_update;
    update_out[idx] = __float2half(u);
}

extern "C" __global__ void fused_lamb_bf16(
    const __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ update_out,
    float beta1, float beta2, float eps, float wd,
    float bias_corr1, float bias_corr2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __bfloat162float(grad[idx]);
    float mi = beta1 * __bfloat162float(m[idx]) + (1.0f - beta1) * gi;
    float vi = beta2 * __bfloat162float(v[idx]) + (1.0f - beta2) * gi * gi;

    m[idx] = __float2bfloat16(mi);
    v[idx] = __float2bfloat16(vi);

    float m_hat = mi / bias_corr1;
    float v_hat = vi / bias_corr2;
    float adam_update = m_hat / (sqrtf(v_hat) + eps);

    float u = (wd > 0.0f) ? (adam_update + wd * __bfloat162float(param[idx])) : adam_update;
    update_out[idx] = __float2bfloat16(u);
}
#endif
