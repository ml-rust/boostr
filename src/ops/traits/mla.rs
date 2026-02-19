//! Multi-Head Latent Attention (MLA) operations trait

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;

/// Scaled dot-product attention allowing different K and V last dimensions.
///
/// Standard attention requires K and V to have the same last dim.
/// MLA needs K dim = head_dim + rope_head_dim, V dim = head_dim_v.
///
/// # Layout contract
///
/// - `q`: `[B, H, S, D_k]` — queries
/// - `k`: `[B, H, S_kv, D_k]` — keys (D_k = head_dim + rope_head_dim)
/// - `v`: `[B, H, S_kv, D_v]` — values (D_v = head_dim_v, can differ from D_k)
/// - `scale`: scaling factor (typically 1/sqrt(D_k))
/// - `causal`: whether to apply causal masking
/// - Output: `[B, H, S, D_v]`
pub trait MlaOps<R: Runtime> {
    fn scaled_dot_product_attention(
        &self,
        q: &Var<R>,
        k: &Var<R>,
        v: &Var<R>,
        scale: f64,
        causal: bool,
    ) -> Result<Var<R>>;
}
