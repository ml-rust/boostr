//! Grammar DFA operations for constrained generation.
//!
//! Masks logits on the device with a compiled DFA. The mask never reads data back to the host.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub use crate::inference::grammar::{DeviceGrammarDfa, INVALID_STATE};

/// Operations for grammar-constrained logit masking on device.
pub trait GrammarDfaOps<R: Runtime> {
    /// Return a copy of `logits` with every token the grammar forbids set to -inf.
    ///
    /// The mask covers the last `grammar.vocab_size()` values of `logits`.
    /// The DFA starts at `grammar.current_state()`. A token is allowed by these rules:
    /// - An EOS token is allowed iff the current state accepts. Its bytes are never walked.
    /// - A non-EOS token is allowed iff each of its bytes has a transition.
    /// - A non-EOS token with zero bytes is never allowed.
    ///
    /// Allowed logits keep their value. The input tensor stays unchanged.
    /// Logits that are not F32, not contiguous, or shorter than the vocabulary are an error.
    ///
    /// One thread per vocabulary token. Each walks its token's bytes in the table.
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<R>,
        grammar: &DeviceGrammarDfa<R>,
    ) -> Result<Tensor<R>>;
}
