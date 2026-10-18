// Grammar DFA logit mask.
//
// One thread per vocabulary token. The rules match the CPU and WebGPU backends:
// - An EOS token is allowed iff the current state accepts. Its bytes are never walked.
// - A non-EOS token is allowed iff each of its bytes has a transition.
// - A non-EOS token with zero bytes is never allowed.
// - No token is allowed from a state outside the table.
// A forbidden token gets a logit of -inf. The mask stays on the device.
//
// Every table is an F32 tensor of exact integer values, the layout all
// backends share. INVALID_STATE (-1) marks a forbidden byte.

#include <cstdint>
#include <cmath>

// True when the grammar allows `token_id` from `current_state`.
__device__ __forceinline__ bool grammar_token_allowed(
    const float* __restrict__ transition_table,
    const float* __restrict__ accepting_mask,
    const float* __restrict__ vocab_bytes,
    const float* __restrict__ vocab_offsets,
    const float* __restrict__ eos_tokens,
    int32_t token_id,
    int32_t current_state,
    int32_t num_states,
    int32_t vocab_bytes_len,
    int32_t num_eos
) {
    if (current_state < 0 || current_state >= num_states) {
        return false;
    }
    const float token_f = (float)token_id;
    for (int32_t i = 0; i < num_eos; i++) {
        if (eos_tokens[i] == token_f) {
            return accepting_mask[current_state] != 0.0f;
        }
    }
    const float start_f = vocab_offsets[token_id];
    const float end_f = vocab_offsets[token_id + 1];
    // A zero-length token makes no progress, so it is never allowed.
    if (!(start_f >= 0.0f && end_f > start_f && end_f <= (float)vocab_bytes_len)) {
        return false;
    }
    const int32_t start = (int32_t)start_f;
    const int32_t end = (int32_t)end_f;
    const float states_f = (float)num_states;
    int64_t state = current_state;
    for (int32_t i = start; i < end; i++) {
        const float byte_f = vocab_bytes[i];
        if (!(byte_f >= 0.0f && byte_f <= 255.0f)) {
            return false;
        }
        const float next = transition_table[state * 256 + (int64_t)byte_f];
        if (!(next >= 0.0f && next < states_f)) {
            return false;
        }
        state = (int64_t)next;
    }
    return true;
}

// Apply the grammar DFA mask to the last position's logits.
//
// Args:
//   logits:           the last vocab_size values of the output copy, F32, masked in place
//   transition_table: [num_states * 256] F32, next state or INVALID_STATE
//   accepting_mask:   [num_states] F32, 1 if accepting, else 0
//   vocab_bytes:      [>= vocab_bytes_len] F32, concatenated token bytes
//   vocab_offsets:    [vocab_size + 1] F32, byte range of each token
//   eos_tokens:       [>= num_eos] F32, EOS token ids
//   current_state:    the DFA state to start from
//   num_states:       number of DFA states
//   vocab_size:       number of tokens
//   vocab_bytes_len:  number of real bytes in vocab_bytes
//   num_eos:          number of EOS ids
extern "C" __global__ void grammar_dfa_mask_logits_kernel(
    float* __restrict__ logits,
    const float* __restrict__ transition_table,
    const float* __restrict__ accepting_mask,
    const float* __restrict__ vocab_bytes,
    const float* __restrict__ vocab_offsets,
    const float* __restrict__ eos_tokens,
    int32_t current_state,
    int32_t num_states,
    int32_t vocab_size,
    int32_t vocab_bytes_len,
    int32_t num_eos
) {
    const int32_t token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= vocab_size) return;

    const bool allowed = grammar_token_allowed(
        transition_table, accepting_mask, vocab_bytes, vocab_offsets, eos_tokens,
        token_id, current_state, num_states, vocab_bytes_len, num_eos);
    if (!allowed) {
        logits[token_id] = -INFINITY;
    }
}
