// Fused SGD optimizer kernel with momentum
//
// Single-pass: read param, grad, momentum_buf → update all in one kernel.

extern "C" __global__ void fused_sgd_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ buf,
    float lr, float momentum, float dampening, float wd,
    int nesterov, int has_buf,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = grad[idx];
    float grad_wd = (wd > 0.0f) ? (gi + wd * param[idx]) : gi;

    float b;
    if (momentum > 0.0f) {
        if (has_buf) {
            b = momentum * buf[idx] + (1.0f - dampening) * grad_wd;
        } else {
            b = grad_wd;
        }
        buf[idx] = b;
    } else {
        b = grad_wd;
        buf[idx] = b;
    }

    float update;
    if (nesterov && momentum > 0.0f) {
        update = grad_wd + momentum * b;
    } else if (momentum > 0.0f) {
        update = b;
    } else {
        update = grad_wd;
    }

    param[idx] = param[idx] - lr * update;
}

extern "C" __global__ void fused_sgd_f64(
    double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ buf,
    double lr, double momentum, double dampening, double wd,
    int nesterov, int has_buf,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double gi = grad[idx];
    double grad_wd = (wd > 0.0) ? (gi + wd * param[idx]) : gi;

    double b;
    if (momentum > 0.0) {
        if (has_buf) {
            b = momentum * buf[idx] + (1.0 - dampening) * grad_wd;
        } else {
            b = grad_wd;
        }
        buf[idx] = b;
    } else {
        b = grad_wd;
        buf[idx] = b;
    }

    double update;
    if (nesterov && momentum > 0.0) {
        update = grad_wd + momentum * b;
    } else if (momentum > 0.0) {
        update = b;
    } else {
        update = grad_wd;
    }

    param[idx] = param[idx] - lr * update;
}

#if __CUDA_ARCH__ >= 700
#include "../dtype_traits.cuh"

extern "C" __global__ void fused_sgd_f16(
    __half* __restrict__ param,
    const __half* __restrict__ grad,
    __half* __restrict__ buf,
    float lr, float momentum, float dampening, float wd,
    int nesterov, int has_buf,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __half2float(grad[idx]);
    float pi = __half2float(param[idx]);
    float grad_wd = (wd > 0.0f) ? (gi + wd * pi) : gi;

    float b;
    if (momentum > 0.0f) {
        if (has_buf) {
            b = momentum * __half2float(buf[idx]) + (1.0f - dampening) * grad_wd;
        } else {
            b = grad_wd;
        }
        buf[idx] = __float2half(b);
    } else {
        b = grad_wd;
        buf[idx] = __float2half(b);
    }

    float update;
    if (nesterov && momentum > 0.0f) {
        update = grad_wd + momentum * b;
    } else if (momentum > 0.0f) {
        update = b;
    } else {
        update = grad_wd;
    }

    param[idx] = __float2half(pi - lr * update);
}

extern "C" __global__ void fused_sgd_bf16(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ buf,
    float lr, float momentum, float dampening, float wd,
    int nesterov, int has_buf,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gi = __bfloat162float(grad[idx]);
    float pi = __bfloat162float(param[idx]);
    float grad_wd = (wd > 0.0f) ? (gi + wd * pi) : gi;

    float b;
    if (momentum > 0.0f) {
        if (has_buf) {
            b = momentum * __bfloat162float(buf[idx]) + (1.0f - dampening) * grad_wd;
        } else {
            b = grad_wd;
        }
        buf[idx] = __float2bfloat16(b);
    } else {
        b = grad_wd;
        buf[idx] = __float2bfloat16(b);
    }

    float update;
    if (nesterov && momentum > 0.0f) {
        update = grad_wd + momentum * b;
    } else if (momentum > 0.0f) {
        update = b;
    } else {
        update = grad_wd;
    }

    param[idx] = __float2bfloat16(pi - lr * update);
}
#endif
