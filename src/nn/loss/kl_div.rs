//! KL divergence loss.

use super::{all_dims, batch_size};
use crate::error::{Error, Result};
use numr::autograd::{Var, var_log, var_mean, var_mul, var_mul_scalar, var_sub, var_sum};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

/// KL divergence: `D_KL(P || Q) = sum(P * (log(P) - log(Q)))`
///
/// Measures how distribution P diverges from distribution Q.
/// Used for knowledge distillation (teacher -> student).
///
/// - `log_q`: `[..., C]` log-probabilities of the model (student). Usually `log_softmax(logits)`.
/// - `p`: `[..., C]` target probability distribution (teacher). Must sum to 1 along last dim.
/// - `batchmean`: if true, sum over all dims and divide by batch size (PyTorch default).
///   If false, compute element-wise mean.
pub fn kl_div_loss<R, C>(client: &C, log_q: &Var<R>, p: &Var<R>, batchmean: bool) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
    R::Client: BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let log_p = var_log(p, client).map_err(Error::Numr)?;
    let diff = var_sub(&log_p, log_q, client).map_err(Error::Numr)?;
    let pointwise = var_mul(p, &diff, client).map_err(Error::Numr)?;
    let dims = all_dims(pointwise.shape().len());

    if batchmean {
        let total = var_sum(&pointwise, &dims, false, client).map_err(Error::Numr)?;
        let n = batch_size(pointwise.shape());
        let loss = var_mul_scalar(&total, 1.0 / n as f64, client).map_err(Error::Numr)?;
        Ok(loss)
    } else {
        let loss = var_mean(&pointwise, &dims, false, client).map_err(Error::Numr)?;
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_kl_div_identical_distributions() {
        let (client, device) = cpu_setup();

        let p = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.3, 0.5, 0.1, 0.4, 0.5], &[2, 3], &device),
            false,
        );
        let log_q = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    0.2f32.ln(),
                    0.3f32.ln(),
                    0.5f32.ln(),
                    0.1f32.ln(),
                    0.4f32.ln(),
                    0.5f32.ln(),
                ],
                &[2, 3],
                &device,
            ),
            false,
        );

        let loss = kl_div_loss(&client, &log_q, &p, false).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(val[0].abs() < 1e-5, "KL(P||P) should be ~0, got {}", val[0]);
    }

    #[test]
    fn test_kl_div_positive() {
        let (client, device) = cpu_setup();

        let p = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.9f32, 0.05, 0.05], &[1, 3], &device),
            false,
        );
        let log_q = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    (1.0f32 / 3.0).ln(),
                    (1.0f32 / 3.0).ln(),
                    (1.0f32 / 3.0).ln(),
                ],
                &[1, 3],
                &device,
            ),
            false,
        );

        let loss = kl_div_loss(&client, &log_q, &p, false).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(
            val[0] > 0.0,
            "KL divergence should be positive, got {}",
            val[0]
        );
    }
}
