//! Configuration for VoxCPM2's MiniCPM4 decoder-only transformer, resolved
//! from the checkpoint's `config.json`.
//!
//! Every architectural knob is config-driven on purpose: VoxCPM2's
//! `residual_lm` is the SAME architecture as `base_lm` with a different
//! config (8 layers, `vocab_size` 0 hence no `embed_tokens` table, and no
//! RoPE), so it becomes a second [`MiniCpm4Config`] rather than a forked
//! module.
//!
//! `residual_lm` has NO section of its own in `config.json`. The reference
//! deep-copies `lm_config` and overrides three fields, so
//! [`MiniCpm4Config::residual_lm_from_config_json`] does the same here,
//! reading the two top-level `residual_lm_*` keys.
//!
//! - `schema`: [`MiniCpm4Config`], its `Default`, the key constants, and the
//!   `residual_lm` override rule
//! - `parse`: the `config.json` entry points and the raw HuggingFace schema

mod parse;
mod schema;

pub use schema::{
    DEFAULT_CONFIG_SECTION, MiniCpm4Config, RESIDUAL_LM_NO_ROPE_KEY, RESIDUAL_LM_NUM_LAYERS_KEY,
};
