//! Single pre-norm decoder layer for VoxCPM2's MiniCPM4.
//!
//! `RmsNorm` -> causal attention -> add, `RmsNorm` -> SwiGLU MLP -> add.
//!
//! Residuals are PLAIN adds: `use_mup` is `false` on this checkpoint, so no
//! muP `scale_depth/sqrt(num_layers)` factor is applied (unlike some
//! MiniCPM-lineage ports that assume it always is). `scale_emb` (12.0) is
//! inactive for the same reason — the reference applies it only under muP —
//! and neither knob has an inert branch here;
//! [`MiniCpm4Config`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config)
//! rejects a `use_mup=true` checkpoint outright rather than letting this
//! layer compute a different model in silence.

mod forward;
mod lora_module;
mod model;

pub use model::MiniCpm4Layer;
