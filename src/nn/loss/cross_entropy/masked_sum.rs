//! Summed cross-entropy loss over masked-in positions.

use super::masked::{MaskedNll, masked_nll};
use crate::error::{Error, Result};
use crate::nn::loss::helpers::all_dims;
use numr::autograd::{Var, var_sum};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Cross-entropy loss summed over masked-in positions: `sum(nll * mask)`.
///
/// The caller divides by its own count of scored positions. Gradient
/// accumulation needs this form. Every scored token of an optimizer step gets
/// weight `1 / total`, where `total` counts the scored tokens of every
/// micro-batch in the step. A per-micro-batch mean gives a token in a sparse
/// micro-batch more weight than a token in a dense one.
///
/// - `logits`: `[N, V]` raw model output (pre-softmax)
/// - `targets`: `[N]` integer class indices in `[0, V)`
/// - `mask`: `[N]` `1.0` = count this position, `0.0` = ignore it
///
/// Differentiable w.r.t. `logits`. `mask` carries no gradient, so masked-out
/// rows receive exactly zero gradient. Its gradient is
/// [`super::cross_entropy_loss_masked`]'s gradient times `sum(mask)`.
///
/// This function reads nothing back from the device. An all-zero mask returns
/// `0` and a zero gradient, never an error. Count the scored positions from
/// the host copy of the mask: a device read of `sum(mask)` stalls the stream.
pub fn cross_entropy_loss_masked_sum<R, C>(
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
    let MaskedNll { masked, .. } = masked_nll(client, logits, targets, mask)?;
    var_sum(&masked, &all_dims(masked.shape().len()), false, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::cross_entropy_loss_masked;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const LOGITS: [f64; 12] = [
        2.0, 1.0, 0.1, // row 0
        0.1, 2.0, 1.0, // row 1
        -1.0, 0.5, 3.0, // row 2
        0.7, -0.2, 0.3, // row 3
    ];
    const TARGETS: [i64; 4] = [0, 1, 2, 1];

    fn logits_var(device: &CpuDevice) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&LOGITS, &[4, 3], device).unwrap(),
            true,
        )
    }

    fn inputs(mask: &[f64], device: &CpuDevice) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let targets = Tensor::<CpuRuntime>::from_slice(&TARGETS, &[4], device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(mask, &[4], device).unwrap();
        (targets, mask)
    }

    #[test]
    fn sum_equals_mean_times_count() {
        let (client, device) = cpu_setup();
        let logits = logits_var(&device);
        let (targets, mask) = inputs(&[1.0, 0.0, 1.0, 1.0], &device);

        let sum = cross_entropy_loss_masked_sum(&client, &logits, &targets, &mask).unwrap();
        let mean = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();

        let s: Vec<f64> = sum.tensor().to_vec();
        let m: Vec<f64> = mean.tensor().to_vec();
        assert!(
            (s[0] - m[0] * 3.0).abs() < 1e-12,
            "sum {} != 3 * mean {}",
            s[0],
            m[0]
        );
    }

    #[test]
    fn sum_gradient_equals_mean_gradient_times_count() {
        let (client, device) = cpu_setup();
        let mask_values = [1.0, 0.0, 1.0, 1.0];
        let count = 3.0;

        let logits_sum = logits_var(&device);
        let (targets, mask) = inputs(&mask_values, &device);
        let sum = cross_entropy_loss_masked_sum(&client, &logits_sum, &targets, &mask).unwrap();
        let sum_grads = backward(&sum, &client).unwrap();
        let g_sum: Vec<f64> = sum_grads.get(logits_sum.id()).unwrap().to_vec();

        let logits_mean = logits_var(&device);
        let mean = cross_entropy_loss_masked(&client, &logits_mean, &targets, &mask).unwrap();
        let mean_grads = backward(&mean, &client).unwrap();
        let g_mean: Vec<f64> = mean_grads.get(logits_mean.id()).unwrap().to_vec();

        assert_eq!(g_sum.len(), g_mean.len());
        for (i, (s, m)) in g_sum.iter().zip(g_mean.iter()).enumerate() {
            assert!(
                (s - m * count).abs() < 1e-12,
                "grad[{i}]: sum {s} != {count} * mean {m}"
            );
        }
    }

    #[test]
    fn masked_out_rows_get_zero_gradient() {
        let (client, device) = cpu_setup();
        let logits = logits_var(&device);
        let (targets, mask) = inputs(&[1.0, 0.0, 1.0, 0.0], &device);

        let sum = cross_entropy_loss_masked_sum(&client, &logits, &targets, &mask).unwrap();
        let grads = backward(&sum, &client).unwrap();
        let g: Vec<f64> = grads.get(logits.id()).unwrap().to_vec();

        for row in [1usize, 3] {
            for (i, v) in g.iter().enumerate().skip(row * 3).take(3) {
                assert_eq!(*v, 0.0, "masked-out grad[{i}] = {v} must be exactly zero");
            }
        }
        for row in [0usize, 2] {
            let slice = &g[row * 3..row * 3 + 3];
            assert!(
                slice.iter().any(|v| v.abs() > 1e-9),
                "kept row {row} must have a gradient, got {slice:?}"
            );
        }
    }

    #[test]
    fn all_zero_mask_returns_zero_without_error() {
        let (client, device) = cpu_setup();
        let logits = logits_var(&device);
        let (targets, mask) = inputs(&[0.0; 4], &device);

        let sum = cross_entropy_loss_masked_sum(&client, &logits, &targets, &mask).unwrap();
        let s: Vec<f64> = sum.tensor().to_vec();
        assert_eq!(s[0], 0.0);

        let grads = backward(&sum, &client).unwrap();
        let g: Vec<f64> = grads.get(logits.id()).unwrap().to_vec();
        assert!(
            g.iter().all(|v| *v == 0.0),
            "zero mask must give zero gradient: {g:?}"
        );
    }

    #[test]
    fn shape_mismatch_errors() {
        let (client, device) = cpu_setup();
        let logits = logits_var(&device);
        let targets = Tensor::<CpuRuntime>::from_slice(&TARGETS, &[4], &device).unwrap();
        let bad_mask = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 3], &[3], &device).unwrap();

        let err = cross_entropy_loss_masked_sum(&client, &logits, &targets, &bad_mask).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument { arg: "mask", .. }),
            "expected mask error, got {err}"
        );
    }
}
