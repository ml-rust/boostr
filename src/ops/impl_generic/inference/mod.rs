pub mod speculative;

pub use speculative::{
    compute_acceptance_probs_impl, compute_expected_tokens_impl, verify_speculative_tokens_impl,
};
