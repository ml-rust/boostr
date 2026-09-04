//! Fused ALiBi Flash Attention forward.
//!
//! Separate trait from `FlashAttentionOps` (not a new method on it) because
//! `flash.rs` is already at its 200-line hard limit and a Rust trait cannot
//! span files.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Flash Attention forward fused with ALiBi bias: online softmax, the
/// `[B, H, S_q, S_k]` scores tensor is never materialized. F32 only,
/// `head_dim` 64 or 128 only — matches `flash_attention_alibi_64_fp32` /
/// `_128_fp32` in `alibi.cu`.
///
/// ALiBi slopes (`m_h = 2^(-8h/H)`, `h` = head index, `H` = `num_heads`) are
/// computed inside the kernel from the head index — callers pass no
/// slope/bias tensor.
///
/// Layout: `q`/`k`/`v` are `[B, num_heads, S(_kv), head_dim]`. No GQA — K/V
/// carry `num_heads` heads, same as `FlashAttentionOps::flash_attention_fwd_fp8_kv`.
///
/// Returns `(output, lse)`: `output` is `[B, num_heads, S_q, head_dim]` F32,
/// `lse` is `[B, num_heads, S_q]` F32 log-sum-exp (`max + log(sum(exp))`),
/// matching `FlashAttentionOps::flash_attention_fwd`.
///
/// Fused counterpart to `ops::autograd_biased_attention::var_attention_with_bias`,
/// which materializes the full bias-added scores tensor. This trait never does.
pub trait FlashAlibiOps<R: Runtime> {
    fn flash_attention_fwd_alibi(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;
}
