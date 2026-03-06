// Grammar DFA Logit Masking Kernel
//
// One thread per vocabulary token. Each thread walks the DFA from the current
// state using the token's byte representation. If any byte leads to an invalid
// transition (INVALID_STATE = -1), the token's logit is set to -inf.
//
// This eliminates the CPU<->GPU roundtrip that would otherwise be needed for
// grammar-constrained generation.

#include <cstdint>
#include <cmath>

static constexpr int INVALID_STATE = -1;

// Apply grammar DFA mask to logits.
//
// Each thread processes one vocabulary token:
// 1. Look up the token's byte range from vocab_offsets
// 2. Walk the DFA: for each byte, look up transition_table[state * 256 + byte]
// 3. If any transition is INVALID_STATE, set logit to -inf
//
// Args:
//   logits:           [vocab_size] float — modified in-place (last position logits)
//   transition_table: [num_states * 256] int32 — flattened DFA transition table
//   accepting_mask:   [num_states] int32 — 1 if accepting state, 0 otherwise (unused for masking)
//   vocab_bytes:      [total_bytes] uint8 — concatenated token byte sequences
//   vocab_offsets:    [vocab_size + 1] int32 — start/end offsets into vocab_bytes
//   current_state:    int32 — the DFA state to start from
//   num_states:       int32 — number of DFA states
//   vocab_size:       int32 — vocabulary size
extern "C" __global__ void grammar_dfa_mask_logits_kernel(
    float* __restrict__ logits,
    const int32_t* __restrict__ transition_table,
    const int32_t* __restrict__ accepting_mask,
    const uint8_t* __restrict__ vocab_bytes,
    const int32_t* __restrict__ vocab_offsets,
    int32_t current_state,
    int32_t num_states,
    int32_t vocab_size
) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= vocab_size) return;

    int byte_start = vocab_offsets[token_id];
    int byte_end = vocab_offsets[token_id + 1];
    int num_bytes = byte_end - byte_start;

    int state = current_state;
    bool valid = true;

    // Walk the DFA through each byte of this token
    for (int i = 0; i < num_bytes; i++) {
        uint8_t byte_val = vocab_bytes[byte_start + i];
        int table_idx = state * 256 + byte_val;

        int next_state = transition_table[table_idx];
        if (next_state == INVALID_STATE) {
            valid = false;
            break;
        }
        state = next_state;
    }

    // Empty tokens: only valid if current state is accepting
    if (valid && num_bytes == 0) {
        if (current_state >= 0 && current_state < num_states) {
            valid = (accepting_mask[current_state] != 0);
        } else {
            valid = false;
        }
    }

    if (!valid) {
        logits[token_id] = -INFINITY;
    }
}
