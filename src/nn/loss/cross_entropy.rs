//! Cross-entropy loss functions.

use super::{all_dims, batch_size, prepare_targets};
use crate::error::{Error, Result};
use numr::autograd::{
    Var, var_add, var_gather, var_log_softmax, var_mean, var_mul_scalar, var_neg, var_reshape,
};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, UnaryOps};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_cross_entropy_basic() {
        let (client, device) = cpu_setup();

        #[rustfmt::skip]
        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[2.0f32, 1.0, 0.1,   // sample 0: class 0 is highest
                  0.1, 2.0, 1.0],     // sample 1: class 1 is highest
                &[2, 3],
                &device,
            ),
            true,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
        assert_eq!(loss.shape(), &[] as &[usize]);
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(
            val[0] < 1.0,
            "loss={} should be < 1.0 for correct predictions",
            val[0]
        );
    }

    #[test]
    fn test_cross_entropy_wrong_predictions() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    0.1f32, 0.1, 2.0, // sample 0: class 2 is highest
                    2.0, 0.1, 0.1, // sample 1: class 0 is highest
                ],
                &[2, 3],
                &device,
            ),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(
            val[0] > 1.0,
            "loss={} should be > 1.0 for wrong predictions",
            val[0]
        );
    }

    #[test]
    fn test_label_smoothing_reduces_confidence() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss_no_smooth = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let loss_smooth = cross_entropy_loss_smooth(&client, &logits, &targets, 0.1).unwrap();

        let v0: Vec<f32> = loss_no_smooth.tensor().to_vec();
        let vs: Vec<f32> = loss_smooth.tensor().to_vec();

        assert!(
            vs[0] > v0[0],
            "smoothed loss {} should be > unsmoothed {}",
            vs[0],
            v0[0]
        );
    }

    #[test]
    fn test_label_smoothing_zero_is_ce() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss_ce = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let loss_smooth = cross_entropy_loss_smooth(&client, &logits, &targets, 0.0).unwrap();

        let v0: Vec<f32> = loss_ce.tensor().to_vec();
        let vs: Vec<f32> = loss_smooth.tensor().to_vec();
        assert!(
            (v0[0] - vs[0]).abs() < 1e-6,
            "smoothing=0 should match CE: {} vs {}",
            v0[0],
            vs[0]
        );
    }
}
