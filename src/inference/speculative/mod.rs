//! Speculative decoding with a model-free drafter or a draft model, greedy or sampled.
//!
//! A [`Drafter`] proposes the next tokens from the context alone. A
//! [`ModelDrafter`] samples them from a draft model and keeps the draft
//! distributions for the verify. One target
//! forward scores the last committed token and every draft. The greedy path
//! keeps the longest draft prefix the target agrees with, plus the target's
//! own next token. Its output equals plain greedy decoding token for token,
//! penalties included. The sampled path accepts drafts by rejection sampling.
//! Its output follows the target's filtered distribution exactly.

pub mod client;
pub mod draft;
pub mod executor;
pub mod model;
mod penalty;
#[cfg(test)]
mod toy_lm;
pub mod types;
pub mod verify;

pub use client::SpeculativeClient;
pub use draft::{
    DRAFT_KEY_DOMAIN, DraftModel, Drafter, ModelDrafter, ProbDraft, ProbDrafter, PromptLookup,
};
pub use executor::{DraftSource, SpeculativeGenerator, StepOutput};
pub use model::{SpeculativeModel, SpeculativeProbModel};
pub use types::{
    COUNTERS_PER_STEP, DEFAULT_NUM_DRAFT_TOKENS, Draw, FinishReason, MAX_NUM_DRAFT_TOKENS,
    SampleRule, SpeculativeConfig, SpeculativeStats, Verdict,
};
pub use verify::verify_logits;
