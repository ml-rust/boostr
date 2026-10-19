//! Speculative generation config, sampling rule, statistics and finish reason.

pub mod config;
pub mod sample_rule;
pub mod stats;

pub use config::{DEFAULT_NUM_DRAFT_TOKENS, MAX_NUM_DRAFT_TOKENS, SpeculativeConfig};
pub use sample_rule::{COUNTERS_PER_STEP, Draw, SampleRule, Verdict};
pub use stats::{FinishReason, SpeculativeStats};
