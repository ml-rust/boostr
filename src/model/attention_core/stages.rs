//! Shared prologue/epilogue steps: reshape/permute, Q/K norm, RoPE, and the
//! reverse reshape after the kernel runs.

use super::spec::AttentionCoreSpec;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use crate::ops::traits::RoPEOps;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Steps 1-4: reshape/permute, contiguous Q/K, Q/K norm, then RoPE.
///
/// Returns `q [B, H, S_q, D]`, `k`/`v` `[B, H_kv, S_k, D]` — the KV heads are
/// NOT expanded here, because the flash kernel broadcasts them itself.
///
/// Both kernels go through this function, so the norm-BEFORE-rope order is
/// written once and cannot differ between them.
#[allow(clippy::type_complexity)]
pub(super) fn attention_prologue<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    cos: Option<&Var<R>>,
    sin: Option<&Var<R>>,
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

    // Apply RoPE, or skip it for ALiBi and NoPE models.
    let (q, k) = apply_rotary_if_needed(client, q, k, cos, sin, spec)?;

    Ok((q, k, v))
}

/// Step 6: `[B, H, S_q, D]` -> `[B, S_q, H, D]` -> `[B, S_q, H*D]`.
pub(super) fn attention_epilogue<R>(
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

/// Apply RoPE to Q/K, or return them untouched.
///
/// Two INDEPENDENT reasons to skip the rotation:
///
/// - `spec.use_alibi` — position is carried by the ALiBi bias that
///   [`super::mask::prefill_attention_mask`] adds instead.
/// - `spec.skip_rope` — NoPE: nothing carries position at all.
///
/// `cos`/`sin` are not read in either case, so `None` is valid there.
///
/// When rotation IS required (`!spec.use_alibi && !spec.skip_rope`), a `None`
/// table is [`Error::InvalidArgument`], never a silent skip: dropping the
/// rotation keeps every shape valid and still emits fluent text while
/// computing a different model.
fn apply_rotary_if_needed<R, C>(
    client: &C,
    q: Var<R>,
    k: Var<R>,
    cos: Option<&Var<R>>,
    sin: Option<&Var<R>>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime,
    C: RoPEOps<R>,
{
    if spec.use_alibi || spec.skip_rope {
        Ok((q, k))
    } else {
        let (cos, sin) = match (cos, sin) {
            (Some(cos), Some(sin)) => (cos, sin),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "cos/sin",
                    reason: "a rotating attention block (use_alibi and skip_rope both false) \
                             requires both RoPE tables; got None for at least one. \
                             None is only valid with skip_rope set (NoPE)."
                        .into(),
                });
            }
        };
        let q = client.apply_rope(&q, cos, sin)?;
        let k = client.apply_rope(&k, cos, sin)?;
        Ok((q, k))
    }
}
