//! Router z-loss for sparse MoE routing.
//!
//! Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models".

use super::helpers::all_dims;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_log_softmax, var_mean, var_pow_scalar, var_sub};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

/// ST-MoE router z-loss: `mean(logsumexp(router_logits, dim=-1)^2)`.
///
/// Penalizes large router logits in sparse MoE models while preserving a
/// differentiable path back to the router logits.
///
/// - `router_logits`: `[num_tokens, num_experts]` raw router logits
///
/// Returns scalar loss.
pub fn router_z_loss<R, C>(client: &C, router_logits: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>,
    R::Client: ActivationOps<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let ndim = router_logits.shape().len();
    if ndim != 2 {
        return Err(Error::InvalidArgument {
            arg: "router_logits",
            reason: format!("expected 2 dims [num_tokens, num_experts], got {ndim}"),
        });
    }

    // Stable logsumexp over experts using log_softmax:
    // log_softmax(x)_j = x_j - logsumexp(x), so the row mean of
    // x_j - log_softmax(x)_j recovers one logsumexp value per token.
    let log_probs = var_log_softmax(router_logits, 1, client).map_err(Error::Numr)?;
    let log_z_per_expert = var_sub(router_logits, &log_probs, client).map_err(Error::Numr)?;
    let log_z = var_mean(&log_z_per_expert, &[1], true, client).map_err(Error::Numr)?;
    let sq = var_pow_scalar(&log_z, 2.0, client).map_err(Error::Numr)?;
    let loss = var_mean(&sq, &all_dims(sq.shape().len()), false, client).map_err(Error::Numr)?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn scalar_router_z_loss(values: &[f64], num_tokens: usize, num_experts: usize) -> f64 {
        let mut total = 0.0;
        for token in 0..num_tokens {
            let row = &values[token * num_experts..(token + 1) * num_experts];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|v| (v - max).exp()).sum();
            let logsumexp = max + sum_exp.ln();
            total += logsumexp * logsumexp;
        }
        total / num_tokens as f64
    }

    #[test]
    fn test_router_z_loss_basic() {
        let (client, device) = cpu_setup();

        let logits = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(
                &[0.0f32, 3.0f32.ln(), 3.0f32.ln(), 0.0],
                &[2, 2],
                &device,
            )
            .unwrap(),
            true,
        );

        let loss = router_z_loss(&client, &logits).unwrap();
        assert_eq!(loss.shape(), &[] as &[usize]);
        let val: Vec<f32> = loss.tensor().to_vec();

        // Both rows have logsumexp = ln(4), so the mean square is ln(4)^2.
        let expected = 4.0f32.ln().powi(2);
        assert!(
            (val[0] - expected).abs() < 1e-6,
            "router z-loss={}, expected={}",
            val[0],
            expected
        );
    }

    #[test]
    fn test_router_z_loss_backward_matches_finite_difference() {
        let (client, device) = cpu_setup();

        let values = [0.25f32, -0.75, 1.25, -1.0, 0.5, 0.0];
        let logits = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&values, &[2, 3], &device).unwrap(),
            true,
        );

        let loss = router_z_loss(&client, &logits).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let d_logits: Vec<f32> = grads.get(logits.id()).unwrap().to_vec();

        let eps = 1e-3f64;
        let mut values_f64: Vec<f64> = values.iter().map(|v| *v as f64).collect();
        for idx in 0..values_f64.len() {
            let original = values_f64[idx];
            values_f64[idx] = original + eps;
            let plus = scalar_router_z_loss(&values_f64, 2, 3);
            values_f64[idx] = original - eps;
            let minus = scalar_router_z_loss(&values_f64, 2, 3);
            values_f64[idx] = original;

            let numeric_grad = (plus - minus) / (2.0 * eps);
            assert!(
                (d_logits[idx] as f64 - numeric_grad).abs() < 2e-4,
                "grad[{idx}]={}, finite_diff={}",
                d_logits[idx],
                numeric_grad
            );
        }
    }
}
