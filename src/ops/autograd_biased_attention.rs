//! Autograd attention with an additive bias/mask.
//!
//! `var_flash_attention` expresses causality as flags and takes no bias tensor,
//! so schemes that shift the pre-softmax scores — ALiBi above all — are
//! unreachable from a model built out of `var_*` primitives. This module closes
//! that gap by wiring the existing `multi_head_attention_impl` (already a pure
//! composition of numr autograd ops) to a caller-supplied additive bias.
//!
//! The bias is DATA, not a parameter: it enters as a plain `Tensor<R>` and is
//! attached as a detached leaf, exactly as `LlamaAttention::forward` attaches
//! its causal mask. Q, K and V stay on the gradient path untouched — nothing
//! here re-wraps a gradient-carrying tensor in `Var::new`.

use crate::error::{Error, Result};
use crate::model::attention_mask::causal_window_mask;
use crate::model::traits::ModelClient;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ScalarOps, UtilityOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// How causality reaches the attention scores.
///
/// There is no default and no `bool`: every caller names one of these three, so
/// a non-causal result is always something that was asked for by name and never
/// something that fell out of an omitted argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionCausality {
    /// The supplied bias ALREADY contains the causal structure.
    ///
    /// This is the `alibi_add_bias_causal` case: that kernel writes both the
    /// ALiBi slope term and the `-inf` future mask in one pass. Nothing further
    /// is added on top.
    InBias,

    /// Add a causal mask to the supplied bias.
    ///
    /// The mask comes from [`causal_window_mask`], the same builder the LLaMA
    /// blocks use. `window_size == 0` is full causal; `window_size > 0` is a
    /// sliding window that is INCLUSIVE of the current token.
    ///
    /// Use this with a plain positional bias that carries no masking of its own
    /// — `alibi_add_bias`, for instance, is bidirectional and needs this.
    Mask {
        /// `0` for full causal, otherwise the inclusive sliding-window span.
        window_size: usize,
    },

    /// No causality at all: every query attends to every key.
    ///
    /// For encoder / bidirectional attention only. Naming this on a decoder
    /// makes the next-token objective trivially cheatable.
    Bidirectional,
}

/// Multi-head attention with an additive bias, differentiable w.r.t. Q, K and V.
///
/// Computes `softmax(Q @ K^T / sqrt(d) + bias) @ V` through
/// `multi_head_attention_impl`, which is built from `var_matmul`,
/// `var_mul_scalar`, `var_add` and `var_softmax`. Backward therefore comes from
/// numr's autograd graph — there is no hand-written backward here.
///
/// # Arguments
/// - `q`, `k`, `v`: `[B, H, S, D]` / `[B, H, S_kv, D]`. GQA heads must already
///   be repeated to `H` (see `repeat_kv`), as `multi_head_attention_impl`
///   requires.
/// - `bias`: additive, broadcastable to `[B, H, S, S_kv]` — `[1, 1, S, S_kv]`
///   works. It is DATA and carries NO gradient. Masking entries are `f32::MIN`
///   (not `-inf`: `0 * -inf` is NaN, which is why the rest of this crate uses
///   `f32::MIN` too).
/// - `causality`: see [`AttentionCausality`]. Causality is NEVER inferred.
///
/// # Causality
/// With [`AttentionCausality::Mask`] the causal mask is summed onto `bias` and
/// the sum is clamped at `f32::MIN`, so a bias that already carries `f32::MIN`
/// or `-inf` entries cannot overflow to `-inf` and reintroduce the `0 * -inf`
/// NaN hazard. The masked positions stay masked either way.
pub fn var_attention_with_bias<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    bias: &Tensor<R>,
    causality: AttentionCausality,
    num_heads: usize,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + UtilityOps<R>,
    R::Client: ScalarOps<R>,
{
    let q_shape = q.tensor().shape();
    let k_shape = k.tensor().shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    if k_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("expected 4D [B, H, S_kv, D], got {}D", k_shape.len()),
        });
    }

    let combined = match causality {
        AttentionCausality::InBias | AttentionCausality::Bidirectional => bias.clone(),
        AttentionCausality::Mask { window_size } => {
            let sq = q_shape[2];
            let sk = k_shape[2];
            let mask = causal_window_mask::<R, C>(
                client,
                sq,
                sk,
                window_size,
                q.tensor().dtype(),
                q.tensor().device(),
            )?;
            let summed = client.add(bias, &mask)?;
            client.clamp(&summed, f32::MIN as f64, f32::MAX as f64)?
        }
    };

    // Detached leaf: the bias is data, so it must not — and does not — carry a
    // gradient. Q, K and V are passed through as-is, keeping their grad_fns.
    let bias_var = Var::new(combined, false);
    crate::ops::impl_generic::attention::multi_head_attention_impl(
        client,
        q,
        k,
        v,
        Some(&bias_var),
        num_heads,
    )
}
