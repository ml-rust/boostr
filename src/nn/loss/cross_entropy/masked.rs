//! Cross-entropy loss over masked-in positions only.

use crate::error::{Error, Result};
use crate::nn::loss::helpers::{all_dims, prepare_targets};
use numr::autograd::{Var, var_div_scalar, var_gather, var_log_softmax, var_mul, var_neg, var_sum};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
mod tests {
    use super::super::cross_entropy_loss;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::CpuRuntime;

    /// Negative log-likelihood of one row, straight from the definition.
    fn nll_row(row: &[f32], target: usize) -> f64 {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
        let sum_exp: f64 = row.iter().map(|l| (*l as f64 - max).exp()).sum();
        max + sum_exp.ln() - row[target] as f64
    }

    #[test]
    fn test_masked_all_ones_matches_unmasked() {
        let (client, device) = cpu_setup();

        let values = [
            2.0f32, 1.0, 0.1, // row 0
            0.1, 2.0, 1.0, // row 1
            -1.0, 0.5, 3.0, // row 2
            0.7, -0.2, 0.3, // row 3
        ];
        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[4, 3], &device).unwrap(),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 1], &[4], &device).unwrap();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();

        let plain = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let masked = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();

        let p: Vec<f32> = plain.tensor().to_vec();
        let m: Vec<f32> = masked.tensor().to_vec();
        assert!(
            (p[0] - m[0]).abs() < 1e-6,
            "all-ones mask should match unmasked: {} vs {}",
            p[0],
            m[0]
        );
    }

    #[test]
    fn test_masked_out_positions_are_excluded() {
        let (client, device) = cpu_setup();

        // Rows 1 and 3 are masked out and carry deliberately terrible logits.
        let values = [
            2.0f32, 1.0, 0.1, // row 0: kept
            -20.0, 5.0, 5.0, // row 1: masked out, target 0 is hopeless
            -1.0, 0.5, 3.0, // row 2: kept
            5.0, 5.0, -20.0, // row 3: masked out, target 2 is hopeless
        ];
        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[4, 3], &device).unwrap(),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2, 2], &[4], &device).unwrap();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], &device).unwrap();

        let masked = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
        let plain = cross_entropy_loss(&client, &logits, &targets).unwrap();

        let expected = (nll_row(&values[0..3], 0) + nll_row(&values[6..9], 2)) / 2.0;
        let m: Vec<f32> = masked.tensor().to_vec();
        let p: Vec<f32> = plain.tensor().to_vec();

        assert!(
            (m[0] as f64 - expected).abs() < 1e-5,
            "masked loss {} should equal kept-only loss {expected}",
            m[0]
        );
        assert!(
            (m[0] - p[0]).abs() > 1.0,
            "masked loss {} should differ from unmasked {}",
            m[0],
            p[0]
        );
    }

    #[test]
    fn test_denominator_counts_only_kept_positions() {
        let (client, device) = cpu_setup();

        let kept = [
            2.0f32, 1.0, 0.1, // kept row 0
            -1.0, 0.5, 3.0, // kept row 1
        ];
        let kept_targets = [0i64, 2];
        let expected = (nll_row(&kept[0..3], 0) + nll_row(&kept[3..6], 2)) / 2.0;

        // Growing padding of masked-out rows must not change the loss.
        for pad in 0..5 {
            let mut values = kept.to_vec();
            let mut targets_data = kept_targets.to_vec();
            let mut mask_data = vec![1.0f32, 1.0];
            for _ in 0..pad {
                values.extend_from_slice(&[-30.0f32, 12.0, 7.5]);
                targets_data.push(0);
                mask_data.push(0.0);
            }
            let n = 2 + pad;

            let logits = Var::new(
                Tensor::<CpuRuntime>::from_slice(&values, &[n, 3], &device).unwrap(),
                false,
            );
            let targets = Tensor::<CpuRuntime>::from_slice(&targets_data, &[n], &device).unwrap();
            let mask = Tensor::<CpuRuntime>::from_slice(&mask_data, &[n], &device).unwrap();

            let loss = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
            let v: Vec<f32> = loss.tensor().to_vec();
            assert!(
                (v[0] as f64 - expected).abs() < 1e-5,
                "loss {} changed with {pad} masked-out rows, expected {expected}",
                v[0]
            );
        }
    }

    #[test]
    fn test_all_zero_mask_errors() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device)
                .unwrap(),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();

        let err = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("mask selected no positions"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let (client, device) = cpu_setup();

        let values = [2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0];
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

        // logits must be rank 2
        let logits_3d = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[1, 2, 3], &device).unwrap(),
            false,
        );
        let err = cross_entropy_loss_masked(&client, &logits_3d, &targets, &mask).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument { arg: "logits", .. }),
            "expected logits error, got {err}"
        );

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[2, 3], &device).unwrap(),
            false,
        );

        // targets length must be N
        let bad_targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0], &[3], &device).unwrap();
        let err = cross_entropy_loss_masked(&client, &logits, &bad_targets, &mask).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument { arg: "targets", .. }),
            "expected targets error, got {err}"
        );

        // mask length must be N
        let bad_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap();
        let err = cross_entropy_loss_masked(&client, &logits, &targets, &bad_mask).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument { arg: "mask", .. }),
            "expected mask error, got {err}"
        );
    }

    #[test]
    fn test_gradient_flows_only_to_kept_positions() {
        let (client, device) = cpu_setup();

        let values = [
            2.0f32, 1.0, 0.1, // row 0: kept
            0.1, 2.0, 1.0, // row 1: masked out
            -1.0, 0.5, 3.0, // row 2: kept
        ];
        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[3, 3], &device).unwrap(),
            true,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0], &[3], &device).unwrap();

        let loss = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let g: Vec<f32> = grads.get(logits.id()).unwrap().to_vec();

        for (i, v) in g.iter().enumerate().take(6).skip(3) {
            assert_eq!(*v, 0.0, "masked-out grad[{i}] = {v} should be exactly zero");
        }
        assert!(
            g[0..3].iter().any(|v| v.abs() > 1e-6),
            "kept row 0 should have gradient, got {:?}",
            &g[0..3]
        );
        assert!(
            g[6..9].iter().any(|v| v.abs() > 1e-6),
            "kept row 2 should have gradient, got {:?}",
            &g[6..9]
        );
    }
}
