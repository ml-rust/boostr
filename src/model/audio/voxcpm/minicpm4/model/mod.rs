//! `MiniCpm4Model` — VoxCPM2's MiniCPM4 decoder-only transformer (`base_lm`),
//! full-sequence causal forward, inference only.
//!
//! ```text
//! inputs_embeds [B, S, hidden]
//!   -> 28x pre-norm decoder layer (causal GQA 16/2, head_dim 128, SwiGLU)
//!   -> final RmsNorm                                   [B, S, hidden]
//! ```
//!
//! `stack` documents what the model deliberately does not do, its inputs,
//! and its weights.
//!
//! - `stack`: [`MiniCpm4Model`], accessors, `embed`, and the causal `forward`
//! - `module`: the `Module` impl (parameter enumeration)
//! - `lora`: adapter attachment and the trainable toggle
//! - `lora_load`: optimizer write-back of adapter values

mod lora;
mod lora_load;
mod module;
mod stack;

pub use stack::MiniCpm4Model;

/// Shared tiny-model fixtures, reachable crate-wide as
/// `minicpm4::model::tests` from the decode, layer and generate tests.
#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use super::stack::tests::{HIDDEN, filled, tiny_model, tiny_nope_model};
}
