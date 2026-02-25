// Fused AdaGrad optimizer kernel
//
// Single-pass: read param, grad, accum → update all in one kernel.

extern "C" __global__ void fused_adagrad_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ accum,
    float lr, float eps, float wd,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = grad[idx];
    float grad_wd = (wd > 0.0f) ? (gi + wd * param[idx]) : gi;
    float acc = accum[idx] + grad_wd * grad_wd;

    accum[idx] = acc;
    param[idx] = param[idx] - lr * grad_wd / (sqrtf(acc) + eps);
}

extern "C" __global__ void fused_adagrad_f64(
    double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ accum,
    double lr, double eps, double wd,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double gi = grad[idx];
    double grad_wd = (wd > 0.0) ? (gi + wd * param[idx]) : gi;
    double acc = accum[idx] + grad_wd * grad_wd;

    accum[idx] = acc;
    param[idx] = param[idx] - lr * grad_wd / (sqrt(acc) + eps);
}

#if __CUDA_ARCH__ >= 700
#include "../dtype_traits.cuh"

extern "C" __global__ void fused_adagrad_f16(
    __half* __restrict__ param,
    const __half* __restrict__ grad,
    __half* __restrict__ accum,
    float lr, float eps, float wd,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __half2float(grad[idx]);
    float pi = __half2float(param[idx]);
    float grad_wd = (wd > 0.0f) ? (gi + wd * pi) : gi;
    float acc = __half2float(accum[idx]) + grad_wd * grad_wd;

    accum[idx] = __float2half(acc);
    param[idx] = __float2half(pi - lr * grad_wd / (sqrtf(acc) + eps));
}

extern "C" __global__ void fused_adagrad_bf16(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ accum,
    float lr, float eps, float wd,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __bfloat162float(grad[idx]);
    float pi = __bfloat162float(param[idx]);
    float grad_wd = (wd > 0.0f) ? (gi + wd * pi) : gi;
    float acc = __bfloat162float(accum[idx]) + grad_wd * grad_wd;

    accum[idx] = __float2bfloat16(acc);
    param[idx] = __float2bfloat16(pi - lr * grad_wd / (sqrtf(acc) + eps));
}
#endif
