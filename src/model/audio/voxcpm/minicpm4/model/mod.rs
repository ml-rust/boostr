//! `MiniCpm4Model` — VoxCPM2's MiniCPM4 decoder-only transformer (`base_lm`),
//! full-sequence causal forward, inference only.
//!
//! ```text
//! inputs_embeds [B, S, hidden]
//!   -> 28x pre-norm decoder layer (causal GQA 16/2, head_dim 128, SwiGLU)
//!   -> final RmsNorm                                   [B, S, hidden]
//! ```
//!
//! Two things this model deliberately does NOT do:
//!
//! - **No `lm_head`.** The checkpoint has none. [`MiniCpm4Model::forward`]
//!   returns hidden states, never logits.
//! - **No KV cache on this path.** [`forward`](MiniCpm4Model::forward)
//!   recomputes every position on every call. The incremental (KV-cached)
//!   decode path — `new_kv_cache` / `prefill` / `decode_step` — lives in the
//!   sibling [`decode`](crate::model::audio::voxcpm::minicpm4::decode) module
//!   and leaves this one untouched.
//!
//! - **RoPE is OPTIONAL.** `residual_lm` runs NoPE (`no_rope`), so `rope` is
//!   `None` there and every attention block skips the rotation on both the
//!   full-sequence and the KV-cached path. Nothing takes its place.
//!
//! [`forward`](MiniCpm4Model::forward) takes pre-computed `inputs_embeds`
//! rather than token ids, matching the real pipeline (which feeds a combined
//! text+audio embedding). The `embed_tokens` table is exposed separately as
//! [`MiniCpm4Model::embed`] and is OPTIONAL: VoxCPM2's `residual_lm` is this
//! same architecture with `vocab_size` 0 and no table at all.
//!
//! Built from plain [`Var<R>`](numr::autograd::Var)-wrapped weights
//! (`requires_grad = false`) rather than autograd-tracked training params —
//! same inference-only posture as the `local_encoder` and AudioVAE siblings.
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
