//! Mean squared error loss.

use super::all_dims;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_mean, var_pow_scalar, var_sub};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

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
    let loss = var_mean(&sq, &all_dims(sq.shape().len()), false, client).map_err(Error::Numr)?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
