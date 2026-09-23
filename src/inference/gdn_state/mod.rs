//! Gated DeltaNet state: per-layer recurrent state for `qwen35` inference.
//!
//! - `state`: one layer's conv window and delta-rule state
//! - `layered`: every GDN layer's state for a model
//! - `lanes`: writing and clearing a single sequence's lane of a batched
//!   state

pub mod lanes;
pub mod layered;
pub mod state;

pub use layered::LayeredGdnState;
pub use state::GdnState;
