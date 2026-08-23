//! The shared attention math that sits between the Q/K/V projections and the
//! output projection.
//!
//! Every dense-attention block in the ecosystem runs the same sequence:
//! reshape/permute the projection outputs, optionally per-head-norm Q and K,
//! apply RoPE, repeat KV heads for GQA, build the additive mask, and run
//! attention. Duplicating that sequence is how a caller silently loses a step —
//! a missing per-head QK-norm keeps every shape valid and still emits fluent
//! text. [`attention_core`] owns the sequence so no caller can diverge from it.
//!
//! This lives under `model/` rather than `ops/` because it composes `nn`
//! modules ([`RmsNorm`]) and the model-level mask builder in
//! [`crate::model::attention_mask`], which the backend-agnostic `ops/` layer
//! does not depend on. It is the sibling of `model/attention_mask.rs`: one
//! rule, one file, shared by every architecture.

use crate::error::{Error, Result};
use crate::model::attention_mask::causal_window_mask;
use crate::nn::RmsNorm;
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::RoPEOps;
use crate::ops::traits::position::alibi::AlibiOps;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, LinalgOps, NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Everything [`attention_core`] needs to know about the block it serves.
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
    pub use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited
    /// context). The window is INCLUSIVE of the current token: query `i` may
    /// attend keys `j` with `i - sliding_window < j <= i`, i.e. exactly
    /// `sliding_window` keys.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub sliding_window: usize,
}

/// The prefill/training additive mask: ALiBi bias or causal(+window).
///
/// Split out so the masking rule is directly testable — a non-causal mask here
/// is invisible to shape checks and still produces fluent text.
pub fn prefill_attention_mask<R, C>(
    client: &C,
    batch: usize,
    sq: usize,
    sk: usize,
    spec: &AttentionCoreSpec<'_, R>,
    device: &R::Device,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + LinalgOps<R> + BinaryOps<R> + AlibiOps<R>,
{
    Ok(if spec.use_alibi {
        // MUST be the `_causal` kernel. `alibi_add_bias` writes a SYMMETRIC
        // `-slope * |qi - ki|` over the whole rectangle and masks nothing,
        // so using it here let every prefill position attend to FUTURE
        // tokens — the same defect the non-ALiBi path had until parity
        // testing caught it. `alibi_add_bias_causal` adds the same distance
        // bias and sets `ki > qi + position` to -inf.
        //
        // `position = 0`: this is the prefill/training path, so query `qi`
        // is at absolute position `qi`.
        let mask = Tensor::<R>::zeros(&[batch, spec.num_heads, sq, sk], DType::F32, device);
        client.alibi_add_bias_causal(&mask, batch, spec.num_heads, sq, sk, 0)?;
        Var::new(mask, false)
    } else {
        // Only called with `sq == sk` — the prefill/training path builds Q and
        // K from the same activation — so the shared builder's key offset is
        // `0`. `causal_window_mask` owns causality, the inclusive sliding
        // window, and the `f32::MIN` fill for every architecture.
        Var::new(
            causal_window_mask::<R, C>(client, sq, sk, spec.sliding_window, device)?,
            false,
        )
    })
}

/// Attention between the Q/K/V projections and the output projection.
///
/// `q`, `k`, `v` are the RAW projection outputs, shaped `[B, S, H * D]`,
/// `[B, S, H_kv * D]`, `[B, S, H_kv * D]`. The return is `[B, S, H * D]`,
/// shaped ready for `o_proj`.
///
/// `cos` / `sin` are the RoPE caches for the positions covered by `q`. They are
/// ignored when `spec.use_alibi` is set.
///
/// # Step order
///
/// 1. reshape `[B, S, H*D]` -> `[B, S, H, D]`, permute -> `[B, H, S, D]`
/// 2. make Q and K contiguous (the fused RoPE kernel assumes contiguous layout)
/// 3. **Q/K norm** (`spec.q_norm` / `spec.k_norm`)
/// 4. **RoPE**, or skip for ALiBi
/// 5. GQA: repeat K/V heads to match Q heads
/// 6. additive mask, then attention
/// 7. permute back and flatten to `[B, S, H*D]`
///
/// ⚠ Steps 3 and 4 are ORDERED: norm, THEN rope. Qwen3 normalizes each head
/// before rotating it, and HuggingFace's `Qwen3Attention` does the same.
/// Swapping them keeps every shape valid and still emits fluent text — a silent
/// corruption that only a logits-level parity test catches
/// (`tests/qwen3_parity.rs`). This function owns the order so that no two
/// callers can disagree about it.
///
/// ⚠ The mask is NOT optional. This is the prefill/training path: the whole
/// sequence arrives at once, so without a causal mask each position attends to
/// FUTURE tokens. That makes the next-token objective trivially cheatable
/// during training and corrupts every prompt position during inference, while
/// staying invisible to shape checks.
pub fn attention_core<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: &Var<R>,
    sin: &Var<R>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ScalarOps<R>
        + LinalgOps<R>
        + BinaryOps<R>
        + NormalizationOps<R>
        + RoPEOps<R>
        + AlibiOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 3D [B, S, H*D], got {}D", q_shape.len()),
        });
    }
    let batch = q_shape[0];
    let seq_len = q_shape[1];

    // Reshape to [B, S, H, D] then permute to [B, H, S, D]
    let q =
        var_reshape(q, &[batch, seq_len, spec.num_heads, spec.head_dim]).map_err(Error::Numr)?;
    let k =
        var_reshape(k, &[batch, seq_len, spec.num_kv_heads, spec.head_dim]).map_err(Error::Numr)?;
    let v =
        var_reshape(v, &[batch, seq_len, spec.num_kv_heads, spec.head_dim]).map_err(Error::Numr)?;

    let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

    // Contiguous Q/K needed because fused RoPE kernel assumes contiguous layout.
    // V skips contiguous — matmul handles strided inputs via copy_strided.
    let q = var_contiguous(&q)?;
    let k = var_contiguous(&k)?;

    // Optional Q/K layer norms (Qwen3, Command-R, Cohere) — applied before RoPE.
    // Input shape [B, H, S, D]: the norm runs over the last dimension (head_dim).
    let (q, k) = apply_qk_norms(client, &q, &k, spec)?;

    // Apply RoPE or skip for ALiBi models
    let (q, k) = apply_rotary_if_needed(client, q, k, cos, sin, spec)?;

    // GQA: repeat K/V heads to match Q heads
    let (k, v) = if spec.num_kv_heads < spec.num_heads {
        let repeat = spec.num_heads / spec.num_kv_heads;
        let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
        let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
        (k, v)
    } else {
        (k, v)
    };

    let sq = q.shape()[2];
    let sk = k.shape()[2];
    let mask = prefill_attention_mask(client, batch, sq, sk, spec, q.tensor().device())?;
    let attn_out = multi_head_attention_impl(client, &q, &k, &v, Some(&mask), spec.num_heads)?;

    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
    let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let attn_out = var_contiguous(&attn_out)?;
    var_reshape(&attn_out, &[batch, seq_len, spec.num_heads * spec.head_dim]).map_err(Error::Numr)
}

/// Apply the optional Q/K per-head norms.
///
/// Input shape `[B, H, S, D]` — the norm runs over the last dimension.
fn apply_qk_norms<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let q = match spec.q_norm {
        Some(norm) => norm.forward(client, q)?,
        None => q.clone(),
    };
    let k = match spec.k_norm {
        Some(norm) => norm.forward(client, k)?,
        None => k.clone(),
    };
    Ok((q, k))
}

/// Apply RoPE to Q/K, or skip for ALiBi models.
fn apply_rotary_if_needed<R, C>(
    client: &C,
    q: Var<R>,
    k: Var<R>,
    cos: &Var<R>,
    sin: &Var<R>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime,
    C: RoPEOps<R>,
{
    if spec.use_alibi {
        Ok((q, k))
    } else {
        let q = client.apply_rope(&q, cos, sin)?;
        let k = client.apply_rope(&k, cos, sin)?;
        Ok((q, k))
    }
}
