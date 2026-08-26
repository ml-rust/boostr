//! Kernel selector and per-block configuration for [`super::attention_core`].

use crate::nn::RmsNorm;
use numr::runtime::Runtime;

/// Which attention kernel finishes the shared sequence.
///
/// The two kernels compute the SAME function. They differ only in memory:
/// `Masked` materializes the additive mask and the score matrix, `Flash` fuses
/// both away. `tests/attention_core_kernels.rs` asserts they agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKernel {
    /// Materialized `[1, 1, sq, sk]` additive mask + `multi_head_attention_impl`.
    ///
    /// The only kernel that supports ALiBi, because the additive bias has
    /// nowhere else to go. Holds an `O(B·H·sq·sk)` score tensor for backward.
    #[default]
    Masked,
    /// Fused flash attention: GQA-native, causality and window as flags.
    ///
    /// `O(B·H·sq·D)` attention memory. Rejects ALiBi — see
    /// [`AttentionCoreSpec::use_alibi`].
    Flash,
}

/// Everything [`super::attention_core`] needs to know about the block it serves.
///
/// Borrowed rather than owned so a caller can hand over its own `RmsNorm`
/// modules without cloning weights.
pub struct AttentionCoreSpec<'a, R: Runtime> {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads. Fewer than `num_heads` selects GQA.
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Optional per-head query norm (Qwen3, Command-R, Cohere).
    pub q_norm: Option<&'a RmsNorm<R>>,
    /// Optional per-head key norm.
    pub k_norm: Option<&'a RmsNorm<R>>,
    /// Use ALiBi instead of RoPE (Falcon v1, BLOOM, MPT).
    ///
    /// Requires [`AttentionKernel::Masked`]. The flash kernel takes no bias
    /// tensor, so `use_alibi` with [`AttentionKernel::Flash`] is a hard error,
    /// never a silent downgrade: dropping the bias leaves every shape valid and
    /// still trains to fluent text while computing a different function.
    pub use_alibi: bool,
    /// Run NoPE: apply no rotary embedding at all (VoxCPM2's `residual_lm`).
    ///
    /// When set, [`apply_rotary_if_needed`](super::stages::apply_rotary_if_needed)
    /// returns Q and K untouched and `cos`/`sin` are never read. NOTHING
    /// replaces the rotation: no ALiBi bias, no learned table, no substitute
    /// of any kind, so the block carries ZERO positional signal and its only
    /// order-dependence is the causal mask.
    ///
    /// Independent of [`use_alibi`](Self::use_alibi), which also skips RoPE but
    /// additionally adds the ALiBi distance bias through
    /// `alibi_add_bias_causal`. Reusing that flag here would inject a
    /// positional bias this path must not have, so the two are separate fields
    /// and either may be set without the other.
    pub skip_rope: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited
    /// context). The window is INCLUSIVE of the current token: query `i` may
    /// attend keys `j` with `i - sliding_window < j <= i`, i.e. exactly
    /// `sliding_window` keys.
    ///
    /// Same sentinel and same inclusivity on BOTH kernels — `Masked` hands it
    /// to `causal_window_mask`, `Flash` hands it to the kernel's `window_size`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub sliding_window: usize,
    /// Which attention kernel to run. Defaults to [`AttentionKernel::Masked`].
    pub kernel: AttentionKernel,
}
