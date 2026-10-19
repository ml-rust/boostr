// The canonical sampling filter and the id-order draw, as block-wide device functions.
//
// Order: temperature, softmax, top-k, top-p, min-p, renormalize.
// `FilterRule` in `ops/traits/inference/sampling_probs.rs` defines each step.
//
// Every function runs on the whole block. `blockDim.x` must be a power of two.
// `scratch` holds `blockDim.x` floats of shared memory.
// Every probability is non-negative, so its bit pattern orders like its value.
// The thresholds come from a binary search over bit patterns, so each one is an exact row value.

#ifndef BOOSTR_SAMPLING_FILTER_CUH
#define BOOSTR_SAMPLING_FILTER_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float sf_to_float(float x) { return x; }
__device__ __forceinline__ float sf_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float sf_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// Copies `vocab` logits from `raw + offset` into `p` as F32.
//
// `dtype` is 0 for F32, 1 for F16, 2 for BF16. Callers must sync before they read `p`.
__device__ __forceinline__ void sf_load_row(const void* raw, int dtype, int offset, float* p, int vocab) {
    if (dtype == 0) {
        const float* logits = (const float*)raw + offset;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) p[i] = logits[i];
    } else if (dtype == 1) {
        const __half* logits = (const __half*)raw + offset;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) p[i] = sf_to_float(logits[i]);
    } else {
        const __nv_bfloat16* logits = (const __nv_bfloat16*)raw + offset;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) p[i] = sf_to_float(logits[i]);
    }
}

// The block-wide sum of `v`. Every thread gets the result.
__device__ __forceinline__ float sf_block_sum(float v, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float r = scratch[0];
    __syncthreads();
    return r;
}

// The block-wide max of `v`. Every thread gets the result.
__device__ __forceinline__ float sf_block_max(float v, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) scratch[tid] = scratch[tid + s];
        __syncthreads();
    }
    float r = scratch[0];
    __syncthreads();
    return r;
}

// The block-wide sum of `v`, in integers. Every thread gets the result.
__device__ __forceinline__ int sf_block_count(int v, float* scratch) {
    int* s_int = (int*)scratch;
    int tid = threadIdx.x;
    s_int[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_int[tid] += s_int[tid + s];
        __syncthreads();
    }
    int r = s_int[0];
    __syncthreads();
    return r;
}

// The number of entries of `p` whose bits are at least `b`.
__device__ __forceinline__ int sf_count_at_least(const float* p, int vocab, uint32_t b, float* scratch) {
    int local = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (__float_as_uint(p[i]) >= b) local++;
    }
    return sf_block_count(local, scratch);
}

// The sum of the entries of `p` whose bits are at least `b`.
__device__ __forceinline__ float sf_mass_at_least(const float* p, int vocab, uint32_t b, float* scratch) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float x = p[i];
        if (__float_as_uint(x) >= b) local += x;
    }
    return sf_block_sum(local, scratch);
}

// Filters one row in place: `p` holds F32 logits on entry and renormalized probabilities on exit.
//
// `top_k` must be 0 when top-k is disabled. A removed token holds 0.
static __device__ void sf_canonical_filter(
    float* p,
    int vocab,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float* scratch
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Temperature and the softmax max.
    float local_max = __uint_as_float(0xff800000u);  // -inf
    for (int i = tid; i < vocab; i += nthreads) {
        float x = p[i];
        if (temperature != 1.0f) x = x / temperature;
        p[i] = x;
        local_max = fmaxf(local_max, x);
    }
    float row_max = sf_block_max(local_max, scratch);

    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += nthreads) {
        float e = expf(p[i] - row_max);
        p[i] = e;
        local_sum += e;
    }
    float denom = sf_block_sum(local_sum, scratch);

    float local_max_p = 0.0f;
    for (int i = tid; i < vocab; i += nthreads) {
        float x = p[i] / denom;
        p[i] = x;
        local_max_p = fmaxf(local_max_p, x);
    }
    float max_p = sf_block_max(local_max_p, scratch);
    uint32_t max_bits = __float_as_uint(max_p);

    // Top-k: the largest bits whose count reaches k is the kth largest value, ties kept.
    uint32_t keep_bits = 0u;
    if (top_k > 0 && top_k < vocab) {
        uint32_t lo = 0u;
        uint32_t hi = max_bits;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo + 1u) / 2u;
            if (sf_count_at_least(p, vocab, mid, scratch) >= top_k) {
                lo = mid;
            } else {
                hi = mid - 1u;
            }
        }
        keep_bits = lo;
    }

    // Top-p: tau is the largest value whose kept mass reaches top_p times the mass top-k left.
    if (top_p < 1.0f) {
        float mass = sf_mass_at_least(p, vocab, keep_bits, scratch);
        float target = top_p * mass;
        uint32_t lo = keep_bits;
        uint32_t hi = max_bits;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo + 1u) / 2u;
            if (sf_mass_at_least(p, vocab, mid, scratch) >= target) {
                lo = mid;
            } else {
                hi = mid - 1u;
            }
        }
        keep_bits = lo;
    }

    // Min-p, then renormalize over the kept set.
    float threshold = __uint_as_float(keep_bits);
    if (min_p > 0.0f) threshold = fmaxf(threshold, min_p * max_p);

    float local_kept = 0.0f;
    for (int i = tid; i < vocab; i += nthreads) {
        float x = p[i];
        if (x >= threshold) {
            local_kept += x;
        } else {
            p[i] = 0.0f;
        }
    }
    float kept = sf_block_sum(local_kept, scratch);
    for (int i = tid; i < vocab; i += nthreads) {
        p[i] = p[i] / kept;
    }
    __syncthreads();
}

// The token whose cumulative probability, in id order, first exceeds `u` times the row mass.
//
// Thread 0 scans. A token with probability 0 is never drawn.
// Rounding past the end returns the last kept token.
static __device__ int64_t sf_draw_id_order(const float* p, int vocab, float u, float* scratch) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) local += p[i];
    float total = sf_block_sum(local, scratch);
    int64_t last_kept = 0;
    if (threadIdx.x == 0) {
        float target = u * total;
        float cumsum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            float x = p[i];
            if (x <= 0.0f) continue;
            cumsum += x;
            last_kept = (int64_t)i;
            if (cumsum > target) break;
        }
    }
    return last_kept;
}

#endif  // BOOSTR_SAMPLING_FILTER_CUH
