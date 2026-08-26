//! The three attention-core entry points: the dispatcher and its two
//! kernel-restricted callees.
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
//! modules ([`RmsNorm`](crate::nn::RmsNorm)) and the model-level mask builder
//! in [`crate::model::attention_mask`], which the backend-agnostic `ops/`
//! layer does not depend on. It is the sibling of `model/attention_mask.rs`:
//! one rule, one file, shared by every architecture.

use super::mask::prefill_attention_mask;
use super::spec::{AttentionCoreSpec, AttentionKernel};
use super::stages::{attention_epilogue, attention_prologue};
use crate::error::{Error, Result};
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::position::alibi::AlibiOps;
use crate::ops::traits::{FlashAttentionOps, RoPEOps};
use crate::ops::var_flash_attention;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, LinalgOps, NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Attention between the Q/K/V projections and the output projection.
///
/// `q`, `k`, `v` are the RAW projection outputs, shaped `[B, S_q, H * D]`,
/// `[B, S_k, H_kv * D]`, `[B, S_k, H_kv * D]`. The return is `[B, S_q, H * D]`,
/// shaped ready for `o_proj`.
///
/// `cos` / `sin` are the RoPE caches for the positions covered by `q` and `k`.
/// `None` means NoPE and is only valid when `spec.skip_rope` is set (or
/// `spec.use_alibi`, which also never reads them); a rotating block
/// (`!spec.use_alibi && !spec.skip_rope`) given `None` for either table
/// returns [`Error::InvalidArgument`] rather than silently skipping the
/// rotation or dereferencing an absent table.
///
/// # Step order
///
/// 1. reshape `[B, S, H*D]` -> `[B, S, H, D]`, permute -> `[B, H, S, D]`
/// 2. make Q and K contiguous (the fused RoPE kernel assumes contiguous layout)
/// 3. **Q/K norm** (`spec.q_norm` / `spec.k_norm`)
/// 4. **RoPE**, or skip for ALiBi / NoPE
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
/// This entry point carries the UNION of both kernels' bounds, including
/// `LinalgOps<R>` and `AlibiOps<R>` (needed only by `Masked`) and
/// `R::Client: FlashAttentionOps<R>` (needed only by `Flash`, whose backward
/// node resolves its client from the runtime). A caller that only ever runs
/// one kernel should call the narrower entry point instead:
/// [`attention_core_masked`] for callers that avoid `FlashAttentionOps<R>` —
/// notably `LlamaAttention`, which sits under the [`crate::model::Model`]
/// trait's fixed `R::Client` bound list — or [`attention_core_flash`] for
/// callers that avoid `LinalgOps<R>`/`AlibiOps<R>`/`BinaryOps<R>`. All three
/// share the same prologue/epilogue, so the step order cannot drift between
/// them.
pub fn attention_core<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: Option<&Var<R>>,
    sin: Option<&Var<R>>,
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
        AttentionKernel::Flash => attention_core_flash(client, q, k, v, cos, sin, spec),
    }
}

/// [`attention_core`] restricted to [`AttentionKernel::Flash`].
///
/// Identical math and identical step order — [`attention_core`]'s `Flash` arm
/// calls straight into here. It exists as its own entry point only because it
/// does NOT need `LinalgOps`/`AlibiOps`/`BinaryOps`, so callers that only ever
/// run the flash kernel (no ALiBi arm, no additive mask) avoid threading those
/// bounds through code that never executes `Masked`.
///
/// `spec.kernel` is IGNORED here: asking for `Masked` and getting the flash
/// path would be exactly the silent divergence [`attention_core`] refuses, so
/// pick the entry point deliberately.
///
/// `cos`/`sin`: `None` means NoPE and is only valid with `spec.skip_rope` set
/// — this kernel runs the same
/// [`apply_rotary_if_needed`](super::stages) gate as `Masked`,
/// so a rotating spec (`skip_rope` false) given `None` returns
/// [`Error::InvalidArgument`] rather than silently skipping the rotation.
pub fn attention_core_flash<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: Option<&Var<R>>,
    sin: Option<&Var<R>>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + NormalizationOps<R> + RoPEOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + FlashAttentionOps<R>,
{
    if spec.use_alibi {
        // Hard error, never a downgrade to `Masked` and never a silently
        // dropped bias: either choice yields a model that trains, runs, and
        // emits fluent text while computing a different function than the
        // one that was asked for.
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
        // GQA broadcast happens INSIDE the kernel. Calling `repeat_kv` here
        // would materialize the very tensor this path exists to avoid.
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
///
/// `cos`/`sin`: `None` means NoPE and is only valid with `spec.skip_rope` set
/// (see [`super::stages`]). A rotating spec given `None` returns
/// [`Error::InvalidArgument`] rather than silently skipping the rotation.
pub fn attention_core_masked<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: Option<&Var<R>>,
    sin: Option<&Var<R>>,
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
