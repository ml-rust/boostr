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
    fn apply_rope(&self, x: &Var<R>, cos_cache: &Var<R>, sin_cache: &Var<R>) -> Result<Var<R>>;
}
