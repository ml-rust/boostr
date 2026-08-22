//! Cross-entropy loss functions.

use super::helpers::{all_dims, batch_size, prepare_targets};
use crate::error::{Error, Result};
use numr::autograd::{
    Var, var_add, var_div_scalar, var_gather, var_log_softmax, var_mean, var_mul, var_mul_scalar,
    var_neg, var_reshape, var_sum,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Cross-entropy loss: -mean(log_softmax(logits, -1)[targets])
///
/// This is the standard loss for classification / language modeling.
///
/// - `logits`: `[..., C]` raw model output (pre-softmax)
/// - `targets`: `[...]` integer class indices in `[0, C)`
///
/// Returns scalar loss.
pub fn cross_entropy_loss<R, C>(client: &C, logits: &Var<R>, targets: &Tensor<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>,
    R::Client: ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>,
{
    let ndim = logits.shape().len();
    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected at least 2 dims, got {ndim}"),
        });
    }

    let vocab_size = logits.shape()[ndim - 1];
    let n = batch_size(logits.shape());

    let log_probs = var_log_softmax(logits, -1, client).map_err(Error::Numr)?;
    let log_probs_flat = var_reshape(&log_probs, &[n, vocab_size]).map_err(Error::Numr)?;

    let targets_expanded = prepare_targets(targets, n)?;
    let selected =
        var_gather(&log_probs_flat, 1, &targets_expanded, client).map_err(Error::Numr)?;

    let neg_selected = var_neg(&selected, client).map_err(Error::Numr)?;
    let loss = var_mean(
        &neg_selected,
        &all_dims(neg_selected.shape().len()),
        false,
        client,
    )
    .map_err(Error::Numr)?;

    Ok(loss)
}

/// Cross-entropy loss with label smoothing
///
/// Smoothed loss = `(1 - smooth) * CE(logits, targets) + smooth * uniform_loss`
/// where `uniform_loss = -mean(log_softmax(logits))` (uniform over all classes).
///
/// Used by GPT-3, PaLM. Prevents overconfident predictions and improves generalization.
///
/// - `logits`: `[..., C]` raw model output (pre-softmax)
/// - `targets`: `[...]` integer class indices in `[0, C)`
/// - `smoothing`: label smoothing factor in `[0, 1)`. 0.0 = no smoothing = standard CE.
pub fn cross_entropy_loss_smooth<R, C>(
    client: &C,
    logits: &Var<R>,
    targets: &Tensor<R>,
    smoothing: f64,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>,
    R::Client: ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>,
{
    if smoothing == 0.0 {
        return cross_entropy_loss(client, logits, targets);
    }

    let ndim = logits.shape().len();
    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected at least 2 dims, got {ndim}"),
        });
    }

    let vocab_size = logits.shape()[ndim - 1];
    let n = batch_size(logits.shape());

    let log_probs = var_log_softmax(logits, -1, client).map_err(Error::Numr)?;
    let log_probs_flat = var_reshape(&log_probs, &[n, vocab_size]).map_err(Error::Numr)?;

    // NLL component: -mean(log_probs[targets])
    let targets_expanded = prepare_targets(targets, n)?;
    let selected =
        var_gather(&log_probs_flat, 1, &targets_expanded, client).map_err(Error::Numr)?;
    let nll = var_neg(
        &var_mean(&selected, &all_dims(selected.shape().len()), false, client)
            .map_err(Error::Numr)?,
        client,
    )
    .map_err(Error::Numr)?;

    // Uniform component: -mean(log_probs) over all classes
    let uniform_loss = var_neg(
        &var_mean(
            &log_probs_flat,
            &all_dims(log_probs_flat.shape().len()),
            false,
            client,
        )
        .map_err(Error::Numr)?,
        client,
    )
    .map_err(Error::Numr)?;

    // Smoothed: (1 - smooth) * nll + smooth * uniform
    let nll_scaled = var_mul_scalar(&nll, 1.0 - smoothing, client).map_err(Error::Numr)?;
    let uni_scaled = var_mul_scalar(&uniform_loss, smoothing, client).map_err(Error::Numr)?;
    let loss = var_add(&nll_scaled, &uni_scaled, client).map_err(Error::Numr)?;

    Ok(loss)
}

/// Cross-entropy loss over masked-in positions only.
///
/// `sum(nll * mask) / sum(mask)`. The denominator is the number of masked-in
/// positions, NOT `N`: dividing by `N` would dilute the loss by every ignored
/// position (for a speech LM, by all the text that precedes the audio in a
/// packed row).
///
/// - `logits`: `[N, V]` raw model output (pre-softmax)
/// - `targets`: `[N]` integer class indices in `[0, V)`
/// - `mask`: `[N]` `1.0` = count this position, `0.0` = ignore it
///
/// Differentiable w.r.t. `logits`. `mask` is data and carries no gradient, so
/// masked-out rows receive exactly zero gradient.
///
/// Errors when the mask selects no positions, because `0 / 0` would return
/// `NaN` and silently poison the gradient.
pub fn cross_entropy_loss_masked<R, C>(
    client: &C,
    logits: &Var<R>,
    targets: &Tensor<R>,
    mask: &Tensor<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>,
    R::Client: ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>,
{
    let logits_shape = logits.shape();
    if logits_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "expected rank-2 [N, V], got shape {logits_shape:?}; flatten the batch and time dims before calling"
            ),
        });
    }
    let n = logits_shape[0];

    let targets_shape = targets.shape();
    if targets_shape.len() != 1 || targets_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "targets",
            reason: format!(
                "expected shape [{n}] to match logits {logits_shape:?}, got {targets_shape:?}; reshape targets to [N]"
            ),
        });
    }

    let mask_shape = mask.shape();
    if mask_shape.len() != 1 || mask_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "mask",
            reason: format!(
                "expected shape [{n}] to match logits {logits_shape:?}, got {mask_shape:?}; reshape mask to [N]"
            ),
        });
    }

    // Per-position NLL, identical to cross_entropy_loss: [N, 1]
    let log_probs = var_log_softmax(logits, -1, client).map_err(Error::Numr)?;
    let targets_expanded = prepare_targets(targets, n)?;
    let selected = var_gather(&log_probs, 1, &targets_expanded, client).map_err(Error::Numr)?;
    let nll = var_neg(&selected, client).map_err(Error::Numr)?;

    // Mask is data, not a parameter: requires_grad = false, so no gradient
    // flows into it and masked-out rows get exactly zero gradient.
    let mask_2d = mask.reshape(&[n, 1]).map_err(Error::Numr)?;
    let logits_dtype = logits.tensor().dtype();
    let mask_2d = if mask.dtype() == logits_dtype {
        mask_2d
    } else {
        client.cast(&mask_2d, logits_dtype).map_err(Error::Numr)?
    };

    // Denominator: number of masked-IN positions, never N.
    //
    // Reading it back to the host is a device sync point, once per loss call.
    // It is what makes the empty-mask guard below possible: without it,
    // sum(mask) == 0 yields 0/0 = NaN and poisons every gradient silently.
    // Do NOT remove this readback without replacing the guard.
    let mask_sum = client.sum(&mask_2d, &[0, 1], false).map_err(Error::Numr)?;
    let kept: f32 = client
        .cast(&mask_sum, DType::F32)
        .map_err(Error::Numr)?
        .item()
        .map_err(Error::Numr)?;
    if kept <= 0.0 || !kept.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "mask",
            reason: format!(
                "mask selected no positions: sum(mask) = {kept} over {n} positions; this usually means the mask was derived from the wrong tensor - pass a mask with at least one 1.0 entry"
            ),
        });
    }

    let mask_var = Var::new(mask_2d, false);
    let masked = var_mul(&nll, &mask_var, client).map_err(Error::Numr)?;
    let total =
        var_sum(&masked, &all_dims(masked.shape().len()), false, client).map_err(Error::Numr)?;
    let loss = var_div_scalar(&total, kept as f64, client).map_err(Error::Numr)?;

    Ok(loss)
}

#[cfg(test)]
mod tests;
