//! Attention operations traits

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Multi-head attention operation (generic, uses impl_generic on all backends)
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

/// Flash Attention v2 — fused O(N) memory attention kernel
///
/// This is a PRIMITIVE op (the fused kernel IS the algorithm). Each backend
/// provides its own optimized implementation. CPU falls back to impl_generic
/// standard attention.
///
/// # Layout contract
///
/// - `q`: `[B, num_heads, S_q, head_dim]` — queries (contiguous)
/// - `k`: `[B, num_kv_heads, S_k, head_dim]` — keys (contiguous)
/// - `v`: `[B, num_kv_heads, S_k, head_dim]` — values (contiguous)
/// - Output: `[B, num_heads, S_q, head_dim]`
/// - Logsumexp: `[B, num_heads, S_q]` (F32, for backward pass)
///
/// # GQA support
///
/// When `num_kv_heads < num_heads`, multiple query heads share one KV head.
/// `num_heads` must be divisible by `num_kv_heads`. The kernel handles
/// KV head broadcasting internally (no `repeat_kv` needed).
///
/// # Sliding window
///
/// When `window_size > 0`, each query position only attends to the most recent
/// `window_size` key positions. The kernel efficiently skips entire K/V tiles
/// outside the window.
#[allow(clippy::too_many_arguments)]
pub trait FlashAttentionOps<R: Runtime> {
    /// Flash Attention forward pass (standard dtypes: F32, F16, BF16)
    ///
    /// Returns `(output, logsumexp)`. The logsumexp tensor is always F32
    /// and is required for the backward pass.
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
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Flash Attention forward pass for FP8 tensors
    ///
    /// FP8 requires per-tensor quantization scales for numerical stability.
    /// All computation is done in FP32 accumulation internally.
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
