//! Non-causal (bidirectional) GQA attention for `feat_encoder`.
//!
//! Every other transformer block in this crate runs through
//! `crate::model::attention_core`, which hardcodes causal(+window) masking —
//! `feat_encoder` is the one bidirectional transformer in the VoxCPM2 stack,
//! attending its fixed 5-position `[CLS, p0, p1, p2, p3]` sequence with no
//! mask at all. That orchestration is written by hand here; every primitive
//! it calls (`Linear`, `RoPE`/`apply_rope`, `multi_head_attention_impl`,
//! `repeat_kv`, `var_contiguous`) is reused as-is. `LlamaAttention` itself is
//! `pub(super)` to `model::llama::model` and not reachable from here, and its
//! `forward` methods are unconditionally causal regardless.
//!
//! `head_dim` (128) is independent of `hidden_size / num_heads` (1024/16 =
//! 64) here — read from config, never derived, and passed straight to the
//! projections and the RoPE cache.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{Linear, RoPE, repeat_kv, var_contiguous};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// `q_proj`: 1024 -> 2048 (16 heads), `k_proj`/`v_proj`: 1024 -> 256 (2
/// heads, GQA group size 8), `o_proj`: 2048 -> 1024. All bias-free.
pub struct LocalEncoderAttention<R: Runtime> {
    pub(crate) q_proj: Linear<R>,
    pub(crate) k_proj: Linear<R>,
    pub(crate) v_proj: Linear<R>,
    pub(crate) o_proj: Linear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

impl<R: Runtime<DType = DType>> LocalEncoderAttention<R> {
    /// Bidirectional GQA attention over `x: [N, S, hidden]` (`N = B*T`,
    /// `S = num_positions`, fixed at 5). No mask: every position is valid
    /// and attends every other, including itself. Softmax scale is
    /// `1/sqrt(head_dim)`, derived from `q`'s actual last dimension by
    /// `multi_head_attention_impl`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
    {
        let shape = x.shape().to_vec();
        let (batch, seq_len) = (shape[0], shape[1]);

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // [N, S, H*D] -> [N, S, H, D] -> [N, H, S, D]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // Contiguous Q/K: the fused RoPE kernel assumes contiguous layout.
        // V: `repeat_kv` requires contiguous input too.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;
        let v = var_contiguous(&v)?;

        let q = client.apply_rope(&q, rope.cos_cache(), rope.sin_cache())?;
        let k = client.apply_rope(&k, rope.cos_cache(), rope.sin_cache())?;

        // GQA: repeat the 2 KV heads to 16 before the dense attention kernel.
        let repeat = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
        let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;

        // No mask: bidirectional, all 5 positions always valid.
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [N, H, S, D] -> [N, S, H, D] -> [N, S, H*D]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}
