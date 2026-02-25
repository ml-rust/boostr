// Fused gradient unscale + clip + inf/nan detect kernel
//
// Phase 1 (fused_grad_unscale_clip_*): unscale by inv_scale, detect inf/nan,
// accumulate L2 norm² into a device scalar via shared-memory + atomicAdd.
//
// Phase 2 (clip_scale_*): reads norm_sq and found_inf from device memory,
// computes clip_coef on-device, applies it — no host roundtrip required.

#include "../dtype_traits.cuh"

// ─── Clip scale (second pass) ────────────────────────────────────────────
// Reads norm_sq and found_inf from device memory; computes and applies
// clip_coef entirely on-device. No host synchronization between phase 1 and 2.

extern "C" __global__ void clip_scale_f32(
    float* __restrict__ data,
    const float* __restrict__ norm_sq,
    const int* __restrict__ found_inf,
    float max_norm,
    int n
) {
    if (*found_inf || max_norm <= 0.0f) return;
    float norm = sqrtf(*norm_sq);
    if (norm <= max_norm) return;
    float clip_coef = max_norm / (norm + 1e-6f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= clip_coef;
    }
}

extern "C" __global__ void clip_scale_f64(
    double* __restrict__ data,
    const float* __restrict__ norm_sq,
    const int* __restrict__ found_inf,
    double max_norm,
    int n
) {
    if (*found_inf || max_norm <= 0.0) return;
    double norm = sqrt((double)(*norm_sq));
    if (norm <= max_norm) return;
    double clip_coef = max_norm / (norm + 1e-6);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= clip_coef;
    }
}

// ─── F32 ────────────────────────────────────────────────────────────────

extern "C" __global__ void fused_grad_unscale_clip_f32(
    float* __restrict__ out,
    const float* __restrict__ grad,
    int* __restrict__ found_inf,
    float* __restrict__ norm_sq,
    float inv_scale,
    int n
) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    float local_sq = 0.0f;

    if (idx < n) {
        float gi = grad[idx];
        if (isinf(gi) || isnan(gi)) {
            atomicExch(found_inf, 1);
        }
        val = gi * inv_scale;
        local_sq = val * val;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq, sdata[0]);
    }

    if (idx < n) {
        out[idx] = val;
    }
}

// ─── F64 ────────────────────────────────────────────────────────────────

extern "C" __global__ void fused_grad_unscale_clip_f64(
    double* __restrict__ out,
    const double* __restrict__ grad,
    int* __restrict__ found_inf,
    float* __restrict__ norm_sq,
    double inv_scale,
    int n
) {
    // norm_sq is f32 (sufficient precision for loss scale decisions)
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double val = 0.0;
    float local_sq = 0.0f;

    if (idx < n) {
        double gi = grad[idx];
        if (isinf(gi) || isnan(gi)) {
            atomicExch(found_inf, 1);
        }
        val = gi * inv_scale;
        local_sq = (float)(val * val);
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq, sdata[0]);
    }

    if (idx < n) {
        out[idx] = val;
    }
}

// ─── F16 / BF16 (Ampere+) ──────────────────────────────────────────────

#if __CUDA_ARCH__ >= 700

#include <cuda_fp16.h>

extern "C" __global__ void fused_grad_unscale_clip_f16(
    __half* __restrict__ out,
    const __half* __restrict__ grad,
    int* __restrict__ found_inf,
    float* __restrict__ norm_sq,
    float inv_scale,
    int n
) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    float local_sq = 0.0f;

    if (idx < n) {
        float gi = __half2float(grad[idx]);
        if (isinf(gi) || isnan(gi)) {
            atomicExch(found_inf, 1);
        }
        val = gi * inv_scale;
        local_sq = val * val;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq, sdata[0]);
    }

    if (idx < n) {
        out[idx] = __float2half(val);
    }
}

extern "C" __global__ void clip_scale_f16(
    __half* __restrict__ data,
    const float* __restrict__ norm_sq,
    const int* __restrict__ found_inf,
    float max_norm,
    int n
) {
    if (*found_inf || max_norm <= 0.0f) return;
    float norm = sqrtf(*norm_sq);
    if (norm <= max_norm) return;
    float clip_coef = max_norm / (norm + 1e-6f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2half(__half2float(data[idx]) * clip_coef);
    }
}

#endif // __CUDA_ARCH__ >= 700

#if __CUDA_ARCH__ >= 800

#include <cuda_bf16.h>

extern "C" __global__ void fused_grad_unscale_clip_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ grad,
    int* __restrict__ found_inf,
    float* __restrict__ norm_sq,
    float inv_scale,
    int n
) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    float local_sq = 0.0f;

    if (idx < n) {
        float gi = __bfloat162float(grad[idx]);
        if (isinf(gi) || isnan(gi)) {
            atomicExch(found_inf, 1);
        }
        val = gi * inv_scale;
        local_sq = val * val;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq, sdata[0]);
    }

    if (idx < n) {
        out[idx] = __float2bfloat16(val);
    }
}

extern "C" __global__ void clip_scale_bf16(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ norm_sq,
    const int* __restrict__ found_inf,
    float max_norm,
    int n
) {
    if (*found_inf || max_norm <= 0.0f) return;
    float norm = sqrtf(*norm_sq);
    if (norm <= max_norm) return;
    float clip_coef = max_norm / (norm + 1e-6f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * clip_coef);
    }
}

#endif // __CUDA_ARCH__ >= 800
