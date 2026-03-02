// Fused Logits-to-Token Kernel
//
// Single-block kernel: narrow last seq position → cast F32 → apply penalties →
// argmax (greedy) or full stochastic sampling chain.
// Launched with 1 block of 1024 threads.
//
// Fuses what was previously 3-5 separate kernel launches + CPU transfers into one launch.
// Returns [1] I64 token ID on device, enabling pipelined decode.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
    const float* __restrict__ random_val_ptr, // [1] — uniform [0,1), only used if temperature > 0
    int dtype           // 0 = f32, 1 = f16, 2 = bf16
) {
    extern __shared__ float scratch[];  // [blockDim.x] for reductions

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Step 1: Read logits at last seq position into probs[], cast to f32, apply penalties
    // Offset into the last position: (seq_len - 1) * vocab_size
    int base_offset = (seq_len - 1) * vocab_size;

    if (dtype == 0) {
        // F32
        const float* logits = (const float*)logits_raw + base_offset;
        for (int i = tid; i < vocab_size; i += nthreads) {
            probs[i] = logits[i];
        }
    } else if (dtype == 1) {
        // F16
        const __half* logits = (const __half*)logits_raw + base_offset;
        for (int i = tid; i < vocab_size; i += nthreads) {
            probs[i] = __half2float(logits[i]);
        }
    } else {
        // BF16
        const __nv_bfloat16* logits = (const __nv_bfloat16*)logits_raw + base_offset;
        for (int i = tid; i < vocab_size; i += nthreads) {
            probs[i] = __bfloat162float(logits[i]);
        }
    }
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

    // Step 3: Greedy argmax path
    if (temperature == 0.0f) {
        // Parallel argmax reduction
        float local_max = -1e30f;
        int local_idx = 0;
        for (int i = tid; i < vocab_size; i += nthreads) {
            if (probs[i] > local_max) {
                local_max = probs[i];
                local_idx = i;
            }
        }

        // Store (max_val, idx) in shared mem — use scratch for vals, reinterpret upper half for indices
        scratch[tid] = local_max;
        ((int*)scratch)[nthreads + tid] = local_idx;
        __syncthreads();

        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (scratch[tid + s] > scratch[tid]) {
                    scratch[tid] = scratch[tid + s];
                    ((int*)scratch)[nthreads + tid] = ((int*)scratch)[nthreads + tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[0] = (int64_t)((int*)scratch)[nthreads];
        }
        return;
    }

    // Step 4: Stochastic sampling — temperature scaling + softmax
    float local_max = -1e30f;
    float inv_temp = 1.0f / temperature;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float val = probs[i] * inv_temp;
        probs[i] = val;
        if (val > local_max) local_max = val;
    }

    // Block-wide max reduction
    scratch[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    float global_max = scratch[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float val = expf(probs[i] - global_max);
        probs[i] = val;
        local_sum += val;
    }

    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];

    // Normalize
    for (int i = tid; i < vocab_size; i += nthreads) {
        probs[i] /= total_sum;
    }
    __syncthreads();

    // Step 5: Find max probability (for min-p)
    float local_max_prob = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        if (probs[i] > local_max_prob) local_max_prob = probs[i];
    }
    scratch[tid] = local_max_prob;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    float max_prob = scratch[0];

    // Step 6: Top-k via binary search
    float topk_threshold = 0.0f;
    if (top_k > 0 && top_k < vocab_size) {
        float lo = 0.0f;
        float hi = max_prob;

        for (int iter = 0; iter < 32; iter++) {
            float mid = (lo + hi) * 0.5f;

            int local_count = 0;
            for (int i = tid; i < vocab_size; i += nthreads) {
                if (probs[i] >= mid) local_count++;
            }

            ((int*)scratch)[tid] = local_count;
            __syncthreads();
            for (int s = nthreads / 2; s > 0; s >>= 1) {
                if (tid < s) ((int*)scratch)[tid] += ((int*)scratch)[tid + s];
                __syncthreads();
            }
            int total_count = ((int*)scratch)[0];

            if (total_count > top_k) {
                lo = mid;
            } else {
                hi = mid;
            }
            __syncthreads();
        }
        topk_threshold = lo;
    }

    // Step 7: Min-p threshold
    float minp_threshold = 0.0f;
    if (min_p > 0.0f) {
        minp_threshold = min_p * max_prob;
    }

    float threshold = topk_threshold > minp_threshold ? topk_threshold : minp_threshold;

    // Step 8: Zero out below threshold, compute filtered sum
    float local_filtered_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        if (probs[i] < threshold) {
            probs[i] = 0.0f;
        } else {
            local_filtered_sum += probs[i];
        }
    }

    scratch[tid] = local_filtered_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float filtered_sum = scratch[0];

    // Step 9: Top-p (nucleus) — binary search
    if (top_p < 1.0f && filtered_sum > 0.0f) {
        float target_sum = top_p * filtered_sum;
        float lo2 = 0.0f;
        float hi2 = max_prob;

        for (int iter = 0; iter < 32; iter++) {
            float mid = (lo2 + hi2) * 0.5f;

            float partial_sum = 0.0f;
            for (int i = tid; i < vocab_size; i += nthreads) {
                if (probs[i] >= mid) partial_sum += probs[i];
            }

            scratch[tid] = partial_sum;
            __syncthreads();
            for (int s = nthreads / 2; s > 0; s >>= 1) {
                if (tid < s) scratch[tid] += scratch[tid + s];
                __syncthreads();
            }
            float total_above = scratch[0];

            if (total_above > target_sum) {
                lo2 = mid;
            } else {
                hi2 = mid;
            }
            __syncthreads();
        }

        float new_sum = 0.0f;
        for (int i = tid; i < vocab_size; i += nthreads) {
            if (probs[i] < lo2) {
                probs[i] = 0.0f;
            } else {
                new_sum += probs[i];
            }
        }

        scratch[tid] = new_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        filtered_sum = scratch[0];
    }

    // Step 10: Multinomial sampling — thread 0 sequential scan
    if (tid == 0) {
        float target = random_val_ptr[0] * filtered_sum;
        float cumsum = 0.0f;
        int64_t sampled = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (cumsum > target) {
                sampled = (int64_t)i;
                break;
            }
            if (i == vocab_size - 1) {
                sampled = (int64_t)i;
            }
        }
        output[0] = sampled;
    }
}
