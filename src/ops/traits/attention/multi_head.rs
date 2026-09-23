//! Multi-head attention trait

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;

/// Multi-head attention: composed from numr primitives (matmul, softmax),
/// `Var<R>`-based for autograd — one code path for training and inference.
///
/// Layout: `q`/`k`/`v` are `[B, H, S(_kv), D]`, `mask` (optional) broadcasts
/// to `[B, H, S, S_kv]` and is additive (e.g. -inf for masked positions).
/// Output: `[B, H, S, D]`.
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
