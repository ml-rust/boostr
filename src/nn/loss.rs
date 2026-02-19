//! Loss functions for training.
//!
//! All loss functions operate on `Var<R>` for autograd support.

use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::autograd::{var_gather, var_log_softmax, var_mean, var_neg, var_pow_scalar, var_sub};
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

    // log_softmax along last dim
    let log_probs = var_log_softmax(logits, -1, client).map_err(Error::Numr)?;

    // Flatten to [N, C]
    let batch_size: usize = logits.shape()[..ndim - 1].iter().product();
    let log_probs_flat =
        numr::autograd::var_reshape(&log_probs, &[batch_size, vocab_size]).map_err(Error::Numr)?;

    // Flatten targets to [N], expand to [N, 1] for gather
    let targets_flat = targets.reshape(&[batch_size]).map_err(Error::Numr)?;
    let targets_expanded = targets_flat
        .unsqueeze(1)
        .map_err(Error::Numr)?
        .broadcast_to(&[batch_size, 1])
        .map_err(Error::Numr)?;

    // Gather: [N, 1] = log_probs[i, target[i]]
    let selected =
        var_gather(&log_probs_flat, 1, &targets_expanded, client).map_err(Error::Numr)?;

    // NLL = -mean(selected)
    let neg_selected = var_neg(&selected, client).map_err(Error::Numr)?;
    let all_dims: Vec<usize> = (0..neg_selected.shape().len()).collect();
    let loss = var_mean(&neg_selected, &all_dims, false, client).map_err(Error::Numr)?;

    Ok(loss)
}

/// Mean squared error loss: mean((predictions - targets)^2)
///
/// - `predictions`: `[...]` model output
/// - `targets`: `[...]` ground truth (same shape)
///
/// Returns scalar loss.
pub fn mse_loss<R, C>(client: &C, predictions: &Var<R>, targets: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
    R::Client: BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let diff = var_sub(predictions, targets, client).map_err(Error::Numr)?;
    let sq = var_pow_scalar(&diff, 2.0, client).map_err(Error::Numr)?;
    let all_dims: Vec<usize> = (0..sq.shape().len()).collect();
    let loss = var_mean(&sq, &all_dims, false, client).map_err(Error::Numr)?;
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

        // logits: [2, 3] (2 samples, 3 classes)
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
        // targets: correct classes are 0 and 1
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
        assert_eq!(loss.shape(), &[] as &[usize]); // scalar
        let val: Vec<f32> = loss.tensor().to_vec();
        // Loss should be low since predictions are correct
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
                    2.0, 0.1, 0.1,
                ], // sample 1: class 0 is highest
                &[2, 3],
                &device,
            ),
            false,
        );
        // But targets are 0 and 1 — wrong!
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        // Loss should be higher for wrong predictions
        assert!(
            val[0] > 1.0,
            "loss={} should be > 1.0 for wrong predictions",
            val[0]
        );
    }

    #[test]
    fn test_mse_basic() {
        let (client, device) = cpu_setup();

        let pred = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            true,
        );
        let target = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );

        let loss = mse_loss(&client, &pred, &target).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(
            (val[0]).abs() < 1e-6,
            "MSE of identical tensors should be ~0"
        );
    }

    #[test]
    fn test_mse_nonzero() {
        let (client, device) = cpu_setup();

        let pred = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );
        let target = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0], &[3], &device),
            false,
        );

        let loss = mse_loss(&client, &pred, &target).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        // (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2, mean = 2/3
        let expected = 2.0f32 / 3.0;
        assert!(
            (val[0] - expected).abs() < 1e-5,
            "MSE={}, expected={}",
            val[0],
            expected
        );
    }
}
