// Fused Logits-to-Token Kernel
//
// One block: read the last seq position as F32, apply penalties, then argmax
// (temperature 0) or the canonical filter and the id-order draw.
// `blockDim.x` must be a power of two. Dynamic shared memory holds
// `2 * blockDim.x` floats for the argmax.
// `output[0]` receives the token id on the device.

#include "sampling_filter.cuh"

extern "C" __global__ void logits_to_token_kernel(
    const void* __restrict__ logits_raw,  // [1, seq_len, vocab_size] — f32 or f16/bf16
    float* __restrict__ probs,            // [vocab_size] — scratch buffer
    const int64_t* __restrict__ token_ids,    // [num_unique] — penalty token IDs
    const int32_t* __restrict__ token_counts, // [num_unique] — penalty counts
    int64_t* __restrict__ output,         // [1] — output token ID
    int seq_len,
    int vocab_size,
    int num_unique,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float temperature,  // 0.0 = greedy argmax
    int top_k,          // 0 = disabled
    float top_p,        // 1.0 = disabled
    float min_p,        // 0.0 = disabled
    const float* __restrict__ random_val_ptr, // [1] — uniform [0, 1), read only when temperature > 0
    int dtype           // 0 = f32, 1 = f16, 2 = bf16
) {
    extern __shared__ float scratch[];  // [blockDim.x] for reductions

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Step 1: Read logits at last seq position into probs[], cast to f32, apply penalties
    // Offset into the last position: (seq_len - 1) * vocab_size
    int base_offset = (seq_len - 1) * vocab_size;

    sf_load_row(logits_raw, dtype, base_offset, probs, vocab_size);
    __syncthreads();

    // Step 2: Apply penalties (each thread checks if its vocab indices are in token_ids)
    // Linear scan over num_unique — typically small (64-256 entries)
    if (num_unique > 0) {
        for (int i = tid; i < vocab_size; i += nthreads) {
            for (int j = 0; j < num_unique; j++) {
                if (token_ids[j] == (int64_t)i) {
                    float logit = probs[i];

                    // Repetition penalty
                    if (repeat_penalty != 1.0f) {
                        if (logit > 0.0f) {
                            logit /= repeat_penalty;
                        } else {
                            logit *= repeat_penalty;
                        }
                    }

                    // Frequency penalty
                    logit -= frequency_penalty * (float)token_counts[j];

                    // Presence penalty
                    logit -= presence_penalty;

                    probs[i] = logit;
                    break;
                }
            }
        }
        __syncthreads();
    }

    // Step 3: Greedy argmax path, matching numr's `argmax`.
    // - Equal logits resolve to the lowest id. Thread `tid` scans tid, tid + nthreads, ...
    //   so each merge compares the id on a tie.
    // - NaN never wins: it fails `>` and `==`. A row of only NaN returns id 0.
    if (temperature == 0.0f) {
        const int no_id = 0x7fffffff;  // Lane holds no candidate. Every real id is lower.
        float local_max = __uint_as_float(0xff800000u);  // -inf
        int local_idx = no_id;
        for (int i = tid; i < vocab_size; i += nthreads) {
            float v = probs[i];
            if (v > local_max || (v == local_max && i < local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        // Values in scratch[0, nthreads), ids in the upper half reinterpreted as int.
        int* ids = (int*)scratch + nthreads;
        scratch[tid] = local_max;
        ids[tid] = local_idx;
        __syncthreads();

        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float v = scratch[tid + s];
                int id = ids[tid + s];
                if (v > scratch[tid] || (v == scratch[tid] && id < ids[tid])) {
                    scratch[tid] = v;
                    ids[tid] = id;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[0] = ids[0] == no_id ? 0 : (int64_t)ids[0];
        }
        return;
    }

    // Step 4: Stochastic sampling — the canonical filter, then the id-order draw.
    sf_canonical_filter(probs, vocab_size, temperature, top_k, top_p, min_p, scratch);
    int64_t token = sf_draw_id_order(probs, vocab_size, random_val_ptr[0], scratch);
    if (tid == 0) output[0] = token;
}
