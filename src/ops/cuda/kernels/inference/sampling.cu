// Fused token sampling: the canonical filter, then the id-order draw.
//
// One block. `blockDim.x` must be a power of two.
// Dynamic shared memory holds `blockDim.x` floats.
// `probs` is a `[vocab]` F32 scratch buffer. `output[0]` receives the token id.

#include "sampling_filter.cuh"

extern "C" __global__ void sample_token_kernel(
    const void* __restrict__ logits_raw,      // [vocab] — f32, f16 or bf16
    float* __restrict__ probs,                // [vocab] — scratch
    int vocab_size,
    float temperature,
    int top_k,                                // 0 = disabled
    float top_p,                              // 1.0 = disabled
    float min_p,                              // 0.0 = disabled
    const float* __restrict__ random_val_ptr, // [1] — uniform [0, 1) on the device
    int64_t* __restrict__ output,             // [1] — sampled token id
    int dtype                                 // 0 = f32, 1 = f16, 2 = bf16
) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;

    sf_load_row(logits_raw, dtype, 0, probs, vocab_size);
    __syncthreads();

    sf_canonical_filter(probs, vocab_size, temperature, top_k, top_p, min_p, scratch);
    int64_t token = sf_draw_id_order(probs, vocab_size, random_val_ptr[0], scratch);
    if (tid == 0) output[0] = token;
}
