//! Focal loss for dense object detection.
//!
//! Lin et al., "Focal Loss for Dense Object Detection", 2017.

use super::{all_dims, batch_size, prepare_targets};
use crate::error::{Error, Result};
use numr::autograd::{
    Var, var_add_scalar, var_gather, var_log_softmax, var_mean, var_mul, var_mul_scalar, var_neg,
    var_pow_scalar, var_reshape, var_softmax,
};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Focal loss: `-(1 - p_t)^gamma * log(p_t)`
///
/// Down-weights easy (well-classified) examples and focuses on hard ones.
///
/// - `logits`: `[..., C]` raw model output (pre-softmax)
/// - `targets`: `[...]` integer class indices in `[0, C)`
/// - `gamma`: focusing parameter. 0 = standard CE. Typical: 2.0.
/// - `alpha`: optional class weight for the target class. `None` = uniform.
pub fn focal_loss<R, C>(
    client: &C,
    logits: &Var<R>,
    targets: &Tensor<R>,
    gamma: f64,
    alpha: Option<f64>,
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
    let ndim = logits.shape().len();
    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected at least 2 dims, got {ndim}"),
        });
    }

    let vocab_size = logits.shape()[ndim - 1];
    let n = batch_size(logits.shape());

    let probs = var_softmax(logits, -1, client).map_err(Error::Numr)?;
    let log_probs = var_log_softmax(logits, -1, client).map_err(Error::Numr)?;

    let probs_flat = var_reshape(&probs, &[n, vocab_size]).map_err(Error::Numr)?;
    let log_probs_flat = var_reshape(&log_probs, &[n, vocab_size]).map_err(Error::Numr)?;

    // Gather p_t and log(p_t)
    let targets_expanded = prepare_targets(targets, n)?;
    let p_t = var_gather(&probs_flat, 1, &targets_expanded, client).map_err(Error::Numr)?;
    let log_p_t = var_gather(&log_probs_flat, 1, &targets_expanded, client).map_err(Error::Numr)?;

    // focal_weight = (1 - p_t)^gamma
    let neg_pt = var_mul_scalar(&p_t, -1.0, client).map_err(Error::Numr)?;
    let one_minus_pt = var_add_scalar(&neg_pt, 1.0, client).map_err(Error::Numr)?;
    let focal_weight = var_pow_scalar(&one_minus_pt, gamma, client).map_err(Error::Numr)?;

    // loss = -focal_weight * log(p_t)
    let weighted = var_mul(&focal_weight, &log_p_t, client).map_err(Error::Numr)?;
    let neg_weighted = var_neg(&weighted, client).map_err(Error::Numr)?;

    // Apply alpha if provided
    let neg_weighted = if let Some(a) = alpha {
        var_mul_scalar(&neg_weighted, a, client).map_err(Error::Numr)?
    } else {
        neg_weighted
    };

    let loss = var_mean(
        &neg_weighted,
        &all_dims(neg_weighted.shape().len()),
        false,
        client,
    )
    .map_err(Error::Numr)?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::loss::cross_entropy::cross_entropy_loss;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_focal_loss_gamma_zero_is_ce() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss_ce = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let loss_focal = focal_loss(&client, &logits, &targets, 0.0, None).unwrap();

        let v_ce: Vec<f32> = loss_ce.tensor().to_vec();
        let v_fl: Vec<f32> = loss_focal.tensor().to_vec();
        assert!(
            (v_ce[0] - v_fl[0]).abs() < 1e-5,
            "focal(gamma=0) should match CE: {} vs {}",
            v_ce[0],
            v_fl[0]
        );
    }

    #[test]
    fn test_focal_loss_downweights_easy() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 0.0, 0.0, 0.0, 5.0, 0.0], &[2, 3], &device),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

        let loss_ce = cross_entropy_loss(&client, &logits, &targets).unwrap();
        let loss_focal = focal_loss(&client, &logits, &targets, 2.0, None).unwrap();

        let v_ce: Vec<f32> = loss_ce.tensor().to_vec();
        let v_fl: Vec<f32> = loss_focal.tensor().to_vec();
        assert!(
            v_fl[0] < v_ce[0],
            "focal should be < CE for easy examples: {} vs {}",
            v_fl[0],
            v_ce[0]
        );
    }
}
