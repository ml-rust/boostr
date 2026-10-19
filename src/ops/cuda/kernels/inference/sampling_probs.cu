// The canonical sampling filter over a batch of logits rows.
//
// One block per row. `blockDim.x` must be a power of two.
// Dynamic shared memory holds `blockDim.x` floats.
// `probs` is `[rows, vocab]` F32. It holds the filtered, renormalized row on exit.

#include "sampling_filter.cuh"

template <typename T>
__device__ void sampling_probs_row(
    const T* __restrict__ logits,
    float* __restrict__ probs,
    int vocab,
    float temperature,
    int top_k,
    float top_p,
    float min_p
) {
    extern __shared__ float scratch[];
    const size_t row = (size_t)blockIdx.x * (size_t)vocab;
    const T* in = logits + row;
    float* out = probs + row;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        out[i] = sf_to_float(in[i]);
    }
    __syncthreads();
    sf_canonical_filter(out, vocab, temperature, top_k, top_p, min_p, scratch);
}

#define SAMPLING_PROBS_KERNEL(suffix, T)                                              \
    extern "C" __global__ void sampling_probs_##suffix(                               \
        const T* __restrict__ logits,                                                 \
        float* __restrict__ probs,                                                    \
        int vocab,                                                                    \
        float temperature,                                                            \
        int top_k,                                                                    \
        float top_p,                                                                  \
        float min_p                                                                   \
    ) {                                                                               \
        sampling_probs_row<T>(logits, probs, vocab, temperature, top_k, top_p, min_p); \
    }

SAMPLING_PROBS_KERNEL(f32, float)
SAMPLING_PROBS_KERNEL(f16, __half)
SAMPLING_PROBS_KERNEL(bf16, __nv_bfloat16)
