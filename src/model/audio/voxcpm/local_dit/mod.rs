//! VoxCPM2 `feat_decoder` local DiT ("locdit"): the CFM estimator backbone,
//! reusing the shared bidirectional MiniCPM4 stack in
//! `crate::model::audio::voxcpm::bidirectional`. This module loads weights.
//! Four units live here: weight loading (`loader`), the estimator forward
//! pass (`dit`), the `Module` impl (`module`), and the CFM sampler
//! (`sampler`) that integrates it.

pub mod config;
pub mod dit;
pub mod loader;
pub mod lora;
mod module;
pub mod sampler;

pub use config::LocalDitConfig;
pub use loader::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit};
pub use sampler::{CfmOptions, cfm_time_span};

/// The tiny synthetic estimator, reachable crate-wide as `local_dit::tests`
/// from the sampler, bidirectional-layer, generate and train tests.
#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use super::loader::tests::{
        FEAT_DIM, HEAD_DIM, HIDDEN_DIM, MU_TOKENS, NUM_HEADS, NUM_KV_HEADS, PATCH_SIZE, layer,
        linear, model, norm, t,
    };
}
