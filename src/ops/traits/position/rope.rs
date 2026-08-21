//! Rotary Position Embedding (RoPE) operations trait

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;

/// Rotary Position Embedding (RoPE) operation
///
/// Composite op: applies rotary embeddings to query/key tensors.
/// Rotates pairs of dimensions by position-dependent angles.
///
/// # Layout contract
///
/// - `x`: `[B, H, S, D]` — input tensor (query or key)
/// - `cos_cache`: `[S, D/2]` — precomputed cosines for each position and dim pair
/// - `sin_cache`: `[S, D/2]` — precomputed sines
/// - Output: `[B, H, S, D]` — same shape as input
pub trait RoPEOps<R: Runtime> {
    /// Standard (split-half) RoPE: pairs are `(x[..., d], x[..., d+D/2])`.
    /// Used by LLaMA, Mistral, and most modern LLMs.
    fn apply_rope(&self, x: &Var<R>, cos_cache: &Var<R>, sin_cache: &Var<R>) -> Result<Var<R>>;

    /// Interleaved RoPE: pairs are `(x[..., 2d], x[..., 2d+1])`.
    /// The "mathematically pure" form, treating adjacent elements as a complex
    /// number (real + imaginary). Used by GPT-J/GPT-NeoX and RoFormer.
    ///
    /// ⚠ **Not what HuggingFace checkpoints want.** `transformers` implements
    /// RoPE as `rotate_half` — split-half — for Llama, Mistral, Qwen2 AND Qwen3
    /// alike, and the HF conversion scripts permute `q_proj`/`k_proj` so that
    /// split-half reproduces the original model. Any model loaded from HF
    /// safetensors must use [`RoPEOps::apply_rope`], not this. Reaching for
    /// this because a model is "Qwen" or "NeoX lineage" is exactly the mistake
    /// that produced silently wrong logits in the decoder (see
    /// `tests/qwen3_parity.rs`): every shape stays valid and the output still
    /// looks like text.
    ///
    /// Same layout contract as `apply_rope` for x, cos_cache, sin_cache.
    fn apply_rope_interleaved(
        &self,
        x: &Var<R>,
        cos_cache: &Var<R>,
        sin_cache: &Var<R>,
    ) -> Result<Var<R>>;

    /// YaRN (Yet another RoPE extensioN) for extended context lengths.
    /// Reference: <https://arxiv.org/abs/2309.00071>
    ///
    /// Same rotation formula as standard RoPE, but cos/sin caches are
    /// precomputed with YaRN-scaled frequencies. The `attn_scale` factor
    /// is applied to the output to compensate for longer context.
    ///
    /// - `attn_scale`: attention scaling factor (typically `0.1 * ln(s) + 1.0`
    ///   where `s` is the scale factor). Pass `1.0` for no additional scaling.
    fn apply_rope_yarn(
        &self,
        x: &Var<R>,
        cos_cache: &Var<R>,
        sin_cache: &Var<R>,
        attn_scale: f32,
    ) -> Result<Var<R>>;
}
