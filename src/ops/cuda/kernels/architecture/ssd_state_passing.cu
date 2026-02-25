// Fused SSD state passing kernel for Mamba2
//
// Sequential scan across chunks:
//   states_out[b, c, h, d, n] = states[b, c, h, d, n] + scale[b, h, c] * states_out[b, c-1, h, d, n]
//
// Each thread handles one (batch, head, headdim_idx, dstate_idx) position
// and loops over chunks sequentially.

#include "dtype_traits.cuh"

template<typename T>
__device__ void ssd_state_passing_impl(
    const T* __restrict__ states,
    const T* __restrict__ dA_cumsum,
    T* __restrict__ states_out,
    int batch,
    int nchunks,
    int nheads,
    int headdim,
    int dstate,
    int chunk_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (unsigned int)batch * nheads * headdim * dstate;
    if (idx >= total) return;

    int n_idx = idx % dstate;
    int rem = idx / dstate;
    int d_idx = rem % headdim;
    rem = rem / headdim;
    int h = rem % nheads;
    int b = rem / nheads;

    // states layout: [batch, nchunks, nheads, headdim, dstate]
    int s_chunk_stride = nheads * headdim * dstate;
    int s_batch_stride = nchunks * s_chunk_stride;
    int s_base = b * s_batch_stride + h * headdim * dstate + d_idx * dstate + n_idx;

    // dA_cumsum layout: [batch, nheads, nchunks, chunk_size]
    int da_chunk_stride = chunk_size;
    int da_head_stride = nchunks * chunk_size;
    int da_batch_stride = nheads * da_head_stride;
    int da_base = b * da_batch_stride + h * da_head_stride + (chunk_size - 1);

    // Copy first chunk
    float prev = (float)states[s_base];
    states_out[s_base] = (T)prev;

    // Sequential scan
    for (int c = 1; c < nchunks; c++) {
        float dA_val = (float)dA_cumsum[da_base + c * da_chunk_stride];
        float scale = expf(fminf(dA_val, 0.0f));

        int s_offset = s_base + c * s_chunk_stride;
        float curr = (float)states[s_offset];
        prev = curr + scale * prev;
        states_out[s_offset] = (T)prev;
    }
}

extern "C" {

__global__ void ssd_state_passing_f32(
    const float* states, const float* dA_cumsum, float* states_out,
    int batch, int nchunks, int nheads, int headdim, int dstate, int chunk_size
) {
    ssd_state_passing_impl<float>(states, dA_cumsum, states_out, batch, nchunks, nheads, headdim, dstate, chunk_size);
}

__global__ void ssd_state_passing_f16(
    const __half* states, const __half* dA_cumsum, __half* states_out,
    int batch, int nchunks, int nheads, int headdim, int dstate, int chunk_size
) {
    ssd_state_passing_impl<__half>(states, dA_cumsum, states_out, batch, nchunks, nheads, headdim, dstate, chunk_size);
}

__global__ void ssd_state_passing_bf16(
    const __nv_bfloat16* states, const __nv_bfloat16* dA_cumsum, __nv_bfloat16* states_out,
    int batch, int nchunks, int nheads, int headdim, int dstate, int chunk_size
) {
    ssd_state_passing_impl<__nv_bfloat16>(states, dA_cumsum, states_out, batch, nchunks, nheads, headdim, dstate, chunk_size);
}

} // extern "C"
