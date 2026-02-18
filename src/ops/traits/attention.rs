//! Attention operations trait

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;

/// Multi-head attention operation
///
/// Composite op composed from numr primitives (matmul, softmax, etc.).
/// Uses `Var<R>` for autograd compatibility — one code path for training and inference.
///
/// # Layout contract
///
/// - `q`: `[B, H, S, D]` — queries
/// - `k`: `[B, H, S_kv, D]` — keys
/// - `v`: `[B, H, S_kv, D]` — values
/// - `mask`: optional, broadcastable to `[B, H, S, S_kv]`, **additive** (e.g. -inf for masked positions)
/// - Output: `[B, H, S, D]`
pub trait AttentionOps<R: Runtime> {
    fn multi_head_attention(
        &self,
        q: &Var<R>,
        k: &Var<R>,
        v: &Var<R>,
        mask: Option<&Var<R>>,
        num_heads: usize,
    ) -> Result<Var<R>>;
}
