//! Speculative decoding with a model-free drafter, greedy or sampled.
//!
//! A [`Drafter`] proposes the next tokens from the context alone. One target
//! forward scores the last committed token and every draft. The greedy path
//! keeps the longest draft prefix the target agrees with, plus the target's
//! own next token. Its output equals plain greedy decoding token for token.
//! The sampled path accepts drafts by rejection sampling. Its output follows
//! the target's filtered distribution exactly.

pub mod draft;
pub mod executor;
pub mod model;
#[cfg(test)]
mod toy_lm;
pub mod types;
pub mod verify;

pub use draft::{Drafter, PromptLookup};
pub use executor::{SpeculativeGenerator, StepOutput};
pub use model::SpeculativeModel;
pub use types::{
    COUNTERS_PER_STEP, DEFAULT_NUM_DRAFT_TOKENS, Draw, FinishReason, MAX_NUM_DRAFT_TOKENS,
    SampleRule, SpeculativeConfig, SpeculativeStats, Verdict,
};
pub use verify::verify_logits;
