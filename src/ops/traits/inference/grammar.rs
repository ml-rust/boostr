//! Grammar DFA operations for constrained generation.
//!
//! Masks logits on-device using a compiled DFA, avoiding CPU↔GPU roundtrips.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Device-resident DFA for grammar-constrained generation.
///
/// The transition table is flattened for device-friendly access:
/// - `transition_table[state * 256 + byte]` = next_state (or INVALID_STATE if no transition)
/// - `accepting_mask[state]` = 1 if accepting, 0 otherwise
///
/// `vocab_bytes` stores the token byte representations as a flat buffer with offsets.
pub struct DeviceGrammarDfa<R: Runtime> {
    /// Flattened transition table: [num_states * 256] i32
    /// transition_table[state * 256 + byte] = next_state or -1 for invalid
    pub transition_table: Tensor<R>,
    /// Accepting state mask: [num_states] i32 (1 = accepting, 0 = not)
    pub accepting_mask: Tensor<R>,
    /// Vocabulary byte sequences, concatenated: [total_bytes] u8
    pub vocab_bytes: Tensor<R>,
    /// Offsets into vocab_bytes for each token: [vocab_size + 1] i32
    /// Token i's bytes are vocab_bytes[offsets[i]..offsets[i+1]]
    pub vocab_offsets: Tensor<R>,
    /// Current DFA state
    pub current_state: u32,
    /// Number of states in the DFA
    pub num_states: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

/// Sentinel value for invalid transitions.
pub const INVALID_STATE: i32 = -1;

/// Operations for grammar-constrained logit masking on device.
pub trait GrammarDfaOps<R: Runtime> {
    /// Apply grammar DFA mask to logits tensor in-place.
    ///
    /// For each token in the vocabulary, simulates the DFA starting from
    /// `current_state` using the token's byte representation. If the token
    /// cannot be fully consumed by valid transitions, its logit is set to -inf.
    ///
    /// # Arguments
    /// * `logits` - Logits tensor `[vocab_size]` or `[..., vocab_size]` (last dim)
    /// * `grammar` - Device-resident DFA with transition table and vocab byte data
    ///
    /// # Performance
    /// One thread per vocabulary token. Each thread walks the DFA for its token's
    /// bytes (~1-50 bytes typically). No CPU↔device transfer needed.
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<R>,
        grammar: &DeviceGrammarDfa<R>,
    ) -> Result<Tensor<R>>;
}
