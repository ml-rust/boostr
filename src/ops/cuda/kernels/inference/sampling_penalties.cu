// Sampling Penalty Kernels
//
// Applies repetition, frequency, and presence penalties to logits in-place.
// Used during inference to prevent degenerate repetition.

#include <cstdint>

// Apply penalties to logits for tokens that appear in the history window.
//
// Each thread handles one entry in token_counts[].
// token_ids[i] = vocabulary index, token_counts[i] = occurrence count in window.
//
// For each (token_id, count):
//   - repeat_penalty: if logit > 0, divide by penalty; else multiply
//   - frequency_penalty: logit -= freq_penalty * count
//   - presence_penalty: logit -= pres_penalty
extern "C" __global__ void apply_sampling_penalties_kernel(
    float* __restrict__ logits,              // [vocab_size] — modified in-place
    const int64_t* __restrict__ token_ids,   // [num_unique] — unique token IDs
    const int32_t* __restrict__ token_counts, // [num_unique] — count per token
    int num_unique,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;

    int64_t token_id = token_ids[idx];
    int32_t count = token_counts[idx];
    float logit = logits[token_id];

    // Repetition penalty (llama.cpp style)
    if (repeat_penalty != 1.0f) {
        if (logit > 0.0f) {
            logit /= repeat_penalty;
        } else {
            logit *= repeat_penalty;
        }
    }

    // Frequency penalty: proportional to count
    logit -= frequency_penalty * (float)count;

    // Presence penalty: flat penalty if token appeared
    logit -= presence_penalty;

    logits[token_id] = logit;
}
