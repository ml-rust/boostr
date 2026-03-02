// Fused Token Sampling Kernel
//
// Single-block kernel: temperature → softmax → top-k → top-p → min-p → multinomial.
// Launched with 1 block of 1024 threads for vocab_size up to ~256k.
// Uses a separate global buffer for probabilities (vocab too large for shared mem).
// Returns sampled token ID in output[0].

#include <cstdint>

extern "C" __global__ void sample_token_kernel(
    const float* __restrict__ logits,  // [vocab_size]
    float* __restrict__ probs,         // [vocab_size] — scratch buffer
    int vocab_size,
    float temperature,
    int top_k,        // 0 = disabled
    float top_p,      // 1.0 = disabled
    float min_p,      // 0.0 = disabled
    const float* __restrict__ random_val_ptr, // [1] — uniform [0,1) generated on device
    int32_t* __restrict__ output // [1] — sampled token ID
) {
    extern __shared__ float scratch[];  // [blockDim.x] for reductions

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Step 1: Temperature scaling + find max for numerically stable softmax
    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float val = logits[i];
        if (temperature != 1.0f) {
            val /= temperature;
        }
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

    // Step 2: Exp and sum for softmax
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

    // Normalize to probabilities
    for (int i = tid; i < vocab_size; i += nthreads) {
        probs[i] /= total_sum;
    }
    __syncthreads();

    // Step 3: Find max probability (needed for min-p)
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

    // Step 4: Top-k via binary search for threshold
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

    // Step 5: Min-p threshold
    float minp_threshold = 0.0f;
    if (min_p > 0.0f) {
        minp_threshold = min_p * max_prob;
    }

    // Combined threshold
    float threshold = topk_threshold > minp_threshold ? topk_threshold : minp_threshold;

    // Step 6: Zero out below threshold, compute filtered sum
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

    // Step 7: Top-p (nucleus) — binary search for threshold
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

        // Zero out below lo2, recompute sum
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

    // Step 8: Multinomial sampling — thread 0 sequential scan
    if (tid == 0) {
        float target = random_val_ptr[0] * filtered_sum;
        float cumsum = 0.0f;
        int sampled = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (cumsum > target) {
                sampled = i;
                break;
            }
            if (i == vocab_size - 1) {
                sampled = i;
            }
        }
        output[0] = sampled;
    }
}
