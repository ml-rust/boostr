pub mod sampling;
pub mod speculative;

pub use sampling::{apply_sampling_penalties_impl, sample_token_impl};
pub use speculative::{
    compute_acceptance_probs_impl, compute_expected_tokens_impl, verify_speculative_tokens_impl,
};
