//! The shared attention math that sits between the Q/K/V projections and the
//! output projection.
//!
//! Every dense-attention block in the ecosystem runs the same sequence:
//! reshape/permute the projection outputs, optionally per-head-norm Q and K,
//! apply RoPE, then attend. Duplicating that sequence is how a caller silently
//! loses a step — a missing per-head QK-norm keeps every shape valid and still
//! emits fluent text. [`attention_core`] owns the sequence so no caller can
//! diverge from it.
//!
//! Two attention kernels finish the sequence, selected by
//! [`AttentionCoreSpec::kernel`]:
//!
//! - [`AttentionKernel::Masked`] repeats the KV heads, materializes a
//!   `[1, 1, sq, sk]` additive mask, and runs `multi_head_attention_impl`. It
//!   keeps the `[B, H, sq, sk]` score tensor alive for the backward pass.
//! - [`AttentionKernel::Flash`] passes `num_kv_heads` and the window straight to
//!   the fused kernel, which broadcasts the KV heads and writes causality
//!   itself. No `repeat_kv`, no materialized mask, O(N) attention memory.
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
use crate::ops::traits::position::alibi::AlibiOps;
use crate::ops::traits::{FlashAttentionOps, RoPEOps};
use crate::ops::var_flash_attention;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, LinalgOps, NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Which attention kernel finishes the shared sequence.
///
/// The two kernels compute the SAME function. They differ only in memory:
/// `Masked` materializes the additive mask and the score matrix, `Flash` fuses
/// both away. `tests/attention_core_kernels.rs` asserts they agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKernel {
    /// Materialized `[1, 1, sq, sk]` additive mask + `multi_head_attention_impl`.
    ///
    /// The only kernel that supports ALiBi, because the additive bias has
    /// nowhere else to go. Holds an `O(B·H·sq·sk)` score tensor for backward.
    #[default]
    Masked,
    /// Fused flash attention: GQA-native, causality and window as flags.
    ///
    /// `O(B·H·sq·D)` attention memory. Rejects ALiBi — see
    /// [`AttentionCoreSpec::use_alibi`].
    Flash,
}

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
    ///
    /// Requires [`AttentionKernel::Masked`]. The flash kernel takes no bias
    /// tensor, so `use_alibi` with [`AttentionKernel::Flash`] is a hard error,
    /// never a silent downgrade: dropping the bias leaves every shape valid and
    /// still trains to fluent text while computing a different function.
    pub use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited
    /// context). The window is INCLUSIVE of the current token: query `i` may
    /// attend keys `j` with `i - sliding_window < j <= i`, i.e. exactly
    /// `sliding_window` keys.
    ///
    /// Same sentinel and same inclusivity on BOTH kernels — `Masked` hands it
    /// to `causal_window_mask`, `Flash` hands it to the kernel's `window_size`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub sliding_window: usize,
    /// Which attention kernel to run. Defaults to [`AttentionKernel::Masked`].
    pub kernel: AttentionKernel,
}

/// The prefill/training additive mask: ALiBi bias or causal(+window).
///
/// Split out so the masking rule is directly testable — a non-causal mask here
/// is invisible to shape checks and still produces fluent text.
///
/// Row `i` is at ABSOLUTE position `sk - sq + i`, matching
/// `causal_window_mask` and `flash_standard::build_attention_mask`. Prefill and
/// training pass `sq == sk`, giving offset `0`.
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
        // `position` is the absolute position of query row `0`, i.e. the
        // number of keys that precede the query block. Prefill/training has
        // `sq == sk` and `position == 0`.
        let mask = Tensor::<R>::zeros(&[batch, spec.num_heads, sq, sk], DType::F32, device);
        client.alibi_add_bias_causal(
            &mask,
            batch,
            spec.num_heads,
            sq,
            sk,
            sk.saturating_sub(sq),
        )?;
        Var::new(mask, false)
    } else {
        // `causal_window_mask` owns causality, the inclusive sliding window,
        // the `sk - sq` key offset, and the `f32::MIN` fill for every
        // architecture.
        Var::new(
            causal_window_mask::<R, C>(client, sq, sk, spec.sliding_window, device)?,
            false,
        )
    })
}

/// Attention between the Q/K/V projections and the output projection.
///
/// `q`, `k`, `v` are the RAW projection outputs, shaped `[B, S_q, H * D]`,
/// `[B, S_k, H_kv * D]`, `[B, S_k, H_kv * D]`. The return is `[B, S_q, H * D]`,
/// shaped ready for `o_proj`.
///
/// `cos` / `sin` are the RoPE caches for the positions covered by `q` and `k`.
/// They are ignored when `spec.use_alibi` is set.
///
/// # Step order
///
/// 1. reshape `[B, S, H*D]` -> `[B, S, H, D]`, permute -> `[B, H, S, D]`
/// 2. make Q and K contiguous (the fused RoPE kernel assumes contiguous layout)
/// 3. **Q/K norm** (`spec.q_norm` / `spec.k_norm`)
/// 4. **RoPE**, or skip for ALiBi
/// 5. the selected kernel (see [`AttentionKernel`]) — the ONLY step that
///    differs between kernels
/// 6. permute back and flatten to `[B, S_q, H*D]`
///
/// ⚠ Steps 3 and 4 are ORDERED: norm, THEN rope. Qwen3 normalizes each head
/// before rotating it, and HuggingFace's `Qwen3Attention` does the same.
/// Swapping them keeps every shape valid and still emits fluent text — a silent
/// corruption that only a logits-level parity test catches
/// (`tests/qwen3_parity.rs`). [`attention_prologue`] owns the order for BOTH
/// kernels so that no two callers, and no two kernels, can disagree about it.
///
/// ⚠ Causality is NOT optional on either kernel. This is the prefill/training
/// path: the whole sequence arrives at once, so without causal masking each
/// position attends to FUTURE tokens. That makes the next-token objective
/// trivially cheatable during training and corrupts every prompt position
/// during inference, while staying invisible to shape checks.
///
/// # Client bounds
///
/// This entry point needs `R::Client: FlashAttentionOps<R>` because the flash
/// backward node resolves its client from the runtime. Callers that only ever
/// run [`AttentionKernel::Masked`] — notably `LlamaAttention`, which sits under
/// the [`crate::model::Model`] trait's fixed `R::Client` bound list — call
/// [`attention_core_masked`] instead. Both share the same prologue, so the step
/// order cannot drift between them.
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
    R::Client: TensorOps<R> + ScalarOps<R> + FlashAttentionOps<R>,
{
    match spec.kernel {
        AttentionKernel::Masked => attention_core_masked(client, q, k, v, cos, sin, spec),
        AttentionKernel::Flash => {
            if spec.use_alibi {
                // Hard error, never a downgrade to `Masked` and never a
                // silently dropped bias: either choice yields a model that
                // trains, runs, and emits fluent text while computing a
                // different function than the one that was asked for.
                return Err(Error::InvalidArgument {
                    arg: "spec.kernel",
                    reason: "AttentionKernel::Flash cannot serve use_alibi: the flash kernel \
                             takes no additive bias tensor and writes causality itself. Use \
                             AttentionKernel::Masked for ALiBi models."
                        .into(),
                });
            }
            let (q, k, v) = attention_prologue(client, q, k, v, cos, sin, spec)?;
            let batch = q.shape()[0];
            // Flash reads V directly; the permuted view is not contiguous.
            let v = var_contiguous(&v)?;
            let attn_out = var_flash_attention(
                &q,
                &k,
                &v,
                spec.num_heads,
                // GQA broadcast happens INSIDE the kernel. Calling `repeat_kv`
                // here would materialize the very tensor this path exists to
                // avoid.
                spec.num_kv_heads,
                spec.head_dim,
                // Causal always: prefill/training sees the whole sequence.
                true,
                // Same sentinel as `Masked`: `0` disables, and the window is
                // inclusive of the current token.
                spec.sliding_window,
            )?;
            attention_epilogue(&attn_out, batch, spec)
        }
    }
}

/// [`attention_core`] restricted to [`AttentionKernel::Masked`].
///
/// Identical math and identical step order — [`attention_core`]'s `Masked` arm
/// calls straight into here. It exists as its own entry point only because it
/// does NOT need `R::Client: FlashAttentionOps<R>`, so blocks living under the
/// [`crate::model::Model`] trait's fixed bound list can use the shared sequence.
///
/// `spec.kernel` is IGNORED here: asking for `Flash` and getting the masked
/// path would be exactly the silent downgrade [`attention_core`] refuses, so
/// pick the entry point deliberately.
pub fn attention_core_masked<R, C>(
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
    let (q, k, v) = attention_prologue(client, q, k, v, cos, sin, spec)?;
    let batch = q.shape()[0];

    // GQA: repeat K/V heads to match Q heads.
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

    attention_epilogue(&attn_out, batch, spec)
}

/// Steps 1-4: reshape/permute, contiguous Q/K, Q/K norm, then RoPE.
///
/// Returns `q [B, H, S_q, D]`, `k`/`v` `[B, H_kv, S_k, D]` — the KV heads are
/// NOT expanded here, because the flash kernel broadcasts them itself.
///
/// Both kernels go through this function, so the norm-BEFORE-rope order is
/// written once and cannot differ between them.
#[allow(clippy::type_complexity)]
fn attention_prologue<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: &Var<R>,
    sin: &Var<R>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<(Var<R>, Var<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + NormalizationOps<R> + RoPEOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 3D [B, S, H*D], got {}D", q_shape.len()),
        });
    }
    let k_shape = k.shape();
    if k_shape.len() != 3 || v.shape() != k_shape {
        return Err(Error::InvalidArgument {
            arg: "k/v",
            reason: format!(
                "expected matching 3D [B, S_k, H_kv*D], got k {:?} and v {:?}",
                k_shape,
                v.shape()
            ),
        });
    }
    let batch = q_shape[0];
    let seq_len_q = q_shape[1];
    // K/V carry their OWN length: `sk > sq` is a KV-cached or chunked block,
    // and every mask in this crate reads query row `i` as absolute position
    // `sk - sq + i`. Taking the length from `q` instead would silently attend
    // the wrong key range.
    let seq_len_k = k_shape[1];

    // Reshape to [B, S, H, D] then permute to [B, H, S, D]
    let q =
        var_reshape(q, &[batch, seq_len_q, spec.num_heads, spec.head_dim]).map_err(Error::Numr)?;
    let k = var_reshape(k, &[batch, seq_len_k, spec.num_kv_heads, spec.head_dim])
        .map_err(Error::Numr)?;
    let v = var_reshape(v, &[batch, seq_len_k, spec.num_kv_heads, spec.head_dim])
        .map_err(Error::Numr)?;

    let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

    // Contiguous Q/K needed because fused RoPE kernel assumes contiguous layout.
    let q = var_contiguous(&q)?;
    let k = var_contiguous(&k)?;

    // Optional Q/K layer norms (Qwen3, Command-R, Cohere) — applied before RoPE.
    // Input shape [B, H, S, D]: the norm runs over the last dimension (head_dim).
    let (q, k) = apply_qk_norms(client, &q, &k, spec)?;

    // Apply RoPE or skip for ALiBi models
    let (q, k) = apply_rotary_if_needed(client, q, k, cos, sin, spec)?;

    Ok((q, k, v))
}

/// Step 6: `[B, H, S_q, D]` -> `[B, S_q, H, D]` -> `[B, S_q, H*D]`.
fn attention_epilogue<R>(
    attn_out: &Var<R>,
    batch: usize,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<Var<R>>
where
    R: Runtime,
{
    let seq_len_q = attn_out.shape()[2];
    let attn_out = var_permute(attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let attn_out = var_contiguous(&attn_out)?;
    var_reshape(
        &attn_out,
        &[batch, seq_len_q, spec.num_heads * spec.head_dim],
    )
    .map_err(Error::Numr)
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
