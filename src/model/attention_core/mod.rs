//! The shared attention math that sits between the Q/K/V projections and the
//! output projection. See [`attention_core`] for the full step sequence and
//! rationale, and [`AttentionKernel`] for the two kernel choices.

mod entry;
mod mask;
mod spec;
mod stages;
#[cfg(test)]
mod tests;

pub use entry::{attention_core, attention_core_flash, attention_core_masked};
pub use mask::prefill_attention_mask;
pub use spec::{AttentionCoreSpec, AttentionKernel};
