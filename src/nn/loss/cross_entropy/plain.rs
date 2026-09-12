//! Standard cross-entropy loss.

use crate::error::{Error, Result};
use crate::nn::loss::helpers::{all_dims, batch_size, prepare_targets};
use numr::autograd::{Var, var_gather, var_log_softmax, var_mean, var_neg, var_reshape};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Cross-entropy loss: `-mean(log_softmax(logits, -1)[targets])`
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
            ).unwrap(),
            true,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();

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
            )
            .unwrap(),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();

        let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(
            val[0] > 1.0,
            "loss={} should be > 1.0 for wrong predictions",
            val[0]
        );
    }

    /// Cross-entropy at BF16 must match an F32 reference over the same logits.
    ///
    /// 512 tokens over a 1024-way vocabulary with all-zero logits: every per-token
    /// loss is exactly `ln(1024) = 6.9315`, so the loss is that number however the
    /// mean is computed — unless the mean saturates.
    ///
    /// It did. `cross_entropy_loss` ends in a mean over the flattened `[N, 1]`
    /// per-token losses, and numr summed those 512 BF16 values in BF16. The
    /// running sum stalled at `2048`, so the loss came back as exactly `4.0`
    /// regardless of the logits. The same defect at a 512-token, 128256-way
    /// vocabulary stalled at `4096` and reported `loss 8.0000` on every batch.
    ///
    /// This test is dark under a plain `cargo test`: `f16` is not a default
    /// boostr feature, and BF16 tensors need it.
    #[cfg(feature = "f16")]
    #[test]
    fn test_cross_entropy_bf16_matches_f32_reference() {
        use half::bf16;

        let (client, device) = cpu_setup();

        const N: usize = 512;
        const V: usize = 1024;

        let target_data: Vec<i64> = (0..N as i64).map(|i| i % V as i64).collect();
        let targets = Tensor::<CpuRuntime>::from_slice(&target_data, &[N], &device).unwrap();

        let f32_logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; N * V], &[N, V], &device).unwrap(),
            false,
        );
        let f32_loss: f32 = cross_entropy_loss(&client, &f32_logits, &targets)
            .unwrap()
            .tensor()
            .item()
            .unwrap();
        assert!(
            (f32_loss - 6.931_472).abs() < 1e-4,
            "F32 reference moved: {f32_loss}"
        );

        let bf16_logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![bf16::from_f32(0.0); N * V], &[N, V], &device)
                .unwrap(),
            false,
        );
        let bf16_loss: bf16 = cross_entropy_loss(&client, &bf16_logits, &targets)
            .unwrap()
            .tensor()
            .item()
            .unwrap();
        let bf16_loss = bf16_loss.to_f32();

        assert!(
            (bf16_loss - f32_loss).abs() < 0.06,
            "BF16 loss {bf16_loss} does not match F32 reference {f32_loss}"
        );
    }
}
