// Fused AdamW optimizer kernel
//
// Single-pass: read param, grad, m, v → update all four in one kernel.
// Eliminates 6-8 intermediate buffer allocations per parameter.

#include "../dtype_traits.cuh"

extern "C" __global__ void fused_adamw_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd, float step_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = grad[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * gi;
    float vi = beta2 * v[idx] + (1.0f - beta2) * gi * gi;

    float update = step_size * mi / (sqrtf(vi) + eps);
    float decayed = param[idx] * (1.0f - lr * wd);

    param[idx] = decayed - update;
    m[idx] = mi;
    v[idx] = vi;
}

extern "C" __global__ void fused_adamw_f64(
    double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ m,
    double* __restrict__ v,
    double lr, double beta1, double beta2, double eps, double wd, double step_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double gi = grad[idx];
    double mi = beta1 * m[idx] + (1.0 - beta1) * gi;
    double vi = beta2 * v[idx] + (1.0 - beta2) * gi * gi;

    double update = step_size * mi / (sqrt(vi) + eps);
    double decayed = param[idx] * (1.0 - lr * wd);

    param[idx] = decayed - update;
    m[idx] = mi;
    v[idx] = vi;
}

#if __CUDA_ARCH__ >= 700
extern "C" __global__ void fused_adamw_f16(
    __half* __restrict__ param,
    const __half* __restrict__ grad,
    __half* __restrict__ m,
    __half* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd, float step_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __half2float(grad[idx]);
    float mi = beta1 * __half2float(m[idx]) + (1.0f - beta1) * gi;
    float vi = beta2 * __half2float(v[idx]) + (1.0f - beta2) * gi * gi;

    float update = step_size * mi / (sqrtf(vi) + eps);
    float decayed = __half2float(param[idx]) * (1.0f - lr * wd);

    param[idx] = __float2half(decayed - update);
    m[idx] = __float2half(mi);
    v[idx] = __float2half(vi);
}

extern "C" __global__ void fused_adamw_bf16(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd, float step_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __bfloat162float(grad[idx]);
    float mi = beta1 * __bfloat162float(m[idx]) + (1.0f - beta1) * gi;
    float vi = beta2 * __bfloat162float(v[idx]) + (1.0f - beta2) * gi * gi;

    float update = step_size * mi / (sqrtf(vi) + eps);
    float decayed = __bfloat162float(param[idx]) * (1.0f - lr * wd);

    param[idx] = __float2bfloat16(decayed - update);
    m[idx] = __float2bfloat16(mi);
    v[idx] = __float2bfloat16(vi);
}
#endif
