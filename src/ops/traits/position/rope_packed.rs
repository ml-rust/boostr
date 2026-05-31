//! Position-id-aware (packed/varlen) RoPE trait.

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Rotary Position Embedding for packed (varlen) sequences.
///
/// Standard `RoPEOps::apply_rope` requires a `[B, H, S, D]` input where position
/// is derived from the S-axis index. Packed (varlen) attention concatenates multiple
/// sequences into a single flat token dimension, so each token needs an explicit
/// position id that resets per sequence.
///
/// # Layout contract
///
/// - `x`: `[total_tokens, num_heads, head_dim]` — 3D packed input (query or key)
/// - `cos_cache`: `[max_seq_len, head_dim/2]` — same cache built by `RoPE` nn module
/// - `sin_cache`: `[max_seq_len, head_dim/2]` — same cache
/// - `position_ids`: `[total_tokens]` — integer tensor (I32 or I64); values reset per sequence
/// - Output: `[total_tokens, num_heads, head_dim]`
///
/// # Numerics (split-half, identical to `apply_rope`)
///
/// For token `t`, head `h`, pair index `d ∈ [0, D/2)`, with `p = position_ids[t]`:
/// ```text
/// out[t,h,d]       = x[t,h,d] * cos[p,d] - x[t,h,d+D/2] * sin[p,d]
/// out[t,h,d+D/2]   = x[t,h,d] * sin[p,d] + x[t,h,d+D/2] * cos[p,d]
/// ```
pub trait RoPEPackedOps<R: Runtime> {
    /// Apply position-id-aware (packed) split-half RoPE.
    ///
    /// Used with varlen (packed, unpadded) attention where multiple sequences are
    /// concatenated along the token dimension and per-token positions are explicit.
    fn apply_rope_packed(
        &self,
        x: &Var<R>,
        cos_cache: &Var<R>,
        sin_cache: &Var<R>,
        position_ids: &Tensor<R>,
    ) -> Result<Var<R>>;
}
