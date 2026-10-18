//! Greedy-exact speculative decoding with a model-free drafter.
//!
//! A [`Drafter`] proposes the next tokens from the context alone. One target
//! forward scores the last committed token and every draft. The generator
//! keeps the longest draft prefix the target agrees with, plus the target's
//! own next token. The output equals plain greedy decoding token for token.

pub mod draft;
pub mod executor;
pub mod model;
#[cfg(test)]
mod toy_lm;
pub mod types;

pub use draft::{Drafter, PromptLookup};
pub use executor::{SpeculativeGenerator, StepOutput};
pub use model::SpeculativeModel;
pub use types::{
    DEFAULT_NUM_DRAFT_TOKENS, FinishReason, MAX_NUM_DRAFT_TOKENS, SpeculativeConfig,
    SpeculativeStats,
};
