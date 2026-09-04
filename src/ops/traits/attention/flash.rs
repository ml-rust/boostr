//! Attention operations traits

use crate::error::Result;
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

/// Flash Attention v2 — fused O(N) memory attention kernel. PRIMITIVE op
/// (the fused kernel IS the algorithm); each backend has its own
/// implementation, CPU falls back to impl_generic standard attention.
///
/// Layout: `q` is `[B, num_heads, S_q, head_dim]`, `k`/`v` are
/// `[B, num_kv_heads, S_k, head_dim]` (contiguous). Output is
/// `[B, num_heads, S_q, head_dim]`, logsumexp is `[B, num_heads, S_q]` F32
/// (needed for backward).
///
/// GQA: when `num_kv_heads < num_heads` (must divide evenly), query heads
/// share KV heads — the kernel broadcasts internally, no `repeat_kv` needed.
///
/// Sliding window: when `window_size > 0`, each query attends only to the
/// most recent `window_size` key positions; out-of-window K/V tiles are
/// skipped entirely.
#[allow(clippy::too_many_arguments)]
pub trait FlashAttentionOps<R: Runtime> {
    /// Flash Attention forward pass (standard dtypes: F32, F16, BF16)
    ///
    /// Returns `(output, logsumexp)`. The logsumexp tensor is always F32
    /// and is required for the backward pass.
    ///
    /// # `kv_seq_len`
    ///
    /// When `Some(n)`, the kernel iterates over only the first `n` positions
    /// of K/V while using the tensor's dim-2 as the memory stride. This allows
    /// passing a full-capacity KV cache buffer without copying/narrowing.
    /// When `None`, `k.shape()[2]` is used for both loop bound and stride.
    fn flash_attention_fwd(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
        kv_seq_len: Option<usize>,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Flash Attention forward pass for FP8 tensors. Requires per-tensor
    /// quantization scales for numerical stability; accumulates in FP32.
    ///
    /// - `q_scale`, `k_scale`, `v_scale`: dequantization scales (FP8 → FP32)
    /// - `o_scale`: quantization scale for output (FP32 → FP8)
    fn flash_attention_fwd_fp8(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Flash Attention forward pass with an FP8-quantized KV cache
    ///
    /// `q` stays F32 — only `k_quant`/`v_quant` are FP8 (E4M3). Differs from
    /// `flash_attention_fwd_fp8`, which quantizes Q/K/V/O uniformly with one
    /// scalar scale per tensor.
    ///
    /// - `k_scales`, `v_scales`: F32, `[batch, heads, seq_len_k]` when
    ///   `per_token_scales` (one per token) else `[batch, heads]` (one per
    ///   head). A stored scale is `448 / max_abs`; dequant divides by it.
    /// - No GQA: the kernel indexes K/V with `num_heads` directly, no
    ///   `num_kv_heads` argument. `k_quant`/`v_quant` carry `num_heads`
    ///   heads, matching `q`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_fwd_fp8_kv(
        &self,
        q: &Tensor<R>,
        k_quant: &Tensor<R>,
        v_quant: &Tensor<R>,
        k_scales: &Tensor<R>,
        v_scales: &Tensor<R>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        per_token_scales: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Flash Attention forward pass with an INT4-quantized KV cache
    ///
    /// Mirrors `flash_attention_fwd_fp8_kv`: `q` stays F32, only K/V are
    /// quantized. INT4 is asymmetric per-group: each group carries a scale
    /// AND a zero point (F16, unlike FP8-KV's single F32 scale).
    ///
    /// - `k_quant`, `v_quant`: `[batch, heads, seq_len_k, head_dim/2]` packed
    ///   U8, 2 values per byte, low nibble first.
    /// - `k_scales`, `k_zeros`, `v_scales`, `v_zeros`: F16,
    ///   `[batch, heads, seq_len_k * groups_per_token]`,
    ///   `groups_per_token = head_dim / group_size`. Grouping is per-token:
    ///   group `i` of token `t` covers `k_quant[..., t, i*group_size..]`.
    ///   Requires `head_dim % group_size == 0`, rejected otherwise, since a
    ///   straddling group would disagree between backends.
    /// - No GQA: K/V carry `num_heads` heads, as in `flash_attention_fwd_fp8_kv`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_fwd_int4_kv(
        &self,
        q: &Tensor<R>,
        k_quant: &Tensor<R>,
        v_quant: &Tensor<R>,
        k_scales: &Tensor<R>,
        k_zeros: &Tensor<R>,
        v_scales: &Tensor<R>,
        v_zeros: &Tensor<R>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        group_size: Int4GroupSize,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Flash Attention backward pass
    ///
    /// Computes gradients dQ, dK, dV given output gradient dO and
    /// the forward pass outputs (O, logsumexp).
    ///
    /// # Arguments
    /// - `dout`: gradient of output `[B, num_heads, S_q, head_dim]`
    /// - `q`, `k`, `v`: original inputs from forward pass
    /// - `output`: forward pass output
    /// - `lse`: logsumexp from forward pass `[B, num_heads, S_q]`
    ///
    /// # Returns
    /// `(dq, dk, dv)` — gradients with same shapes as inputs
    fn flash_attention_bwd(
        &self,
        dout: &Tensor<R>,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        output: &Tensor<R>,
        lse: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Flash Attention backward pass for FP8 tensors
    ///
    /// FP8 backward requires per-tensor dequantization/quantization scales.
    /// - `q_scale`, `k_scale`, `v_scale`, `do_scale`: dequant scales for inputs
    /// - `o_scale`: dequant scale for forward output (used in preprocessing)
    /// - `dq_scale`, `dk_scale`, `dv_scale`: quant scales for gradient outputs
    fn flash_attention_bwd_fp8(
        &self,
        dout: &Tensor<R>,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        output: &Tensor<R>,
        lse: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        do_scale: f32,
        o_scale: f32,
        dq_scale: f32,
        dk_scale: f32,
        dv_scale: f32,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
