//! Dropout regularization layer
//!
//! During training, randomly zeroes elements with probability `p` and scales
//! survivors by `1/(1-p)` (inverted dropout). During evaluation, acts as identity.

use crate::error::Result;
use crate::nn::module::TrainMode;
use numr::autograd::{Var, var_dropout};
use numr::ops::{BinaryOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Dropout layer with training/eval mode.
///
/// Applies inverted dropout during training: each element is zeroed with
/// probability `p` and surviving elements are scaled by `1/(1-p)` so the
/// expected value is preserved.
///
/// During evaluation (default after `set_training(false)`), dropout is
/// disabled and the input passes through unchanged.
///
/// # Example
///
/// ```ignore
/// let mut dropout = Dropout::new(0.1);
/// dropout.set_training(true);
///
/// let output = dropout.forward(&client, &input)?; // applies dropout
///
/// dropout.set_training(false);
/// let output = dropout.forward(&client, &input)?; // identity
/// ```
pub struct Dropout {
    p: f64,
    training: bool,
}

impl Dropout {
    /// Create a new dropout layer with drop probability `p`.
    ///
    /// Starts in training mode.
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "dropout probability must be in [0, 1], got {p}"
        );
        Self { p, training: true }
    }

    /// Set training mode. When `true`, dropout is applied; when `false`, identity.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Returns whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the dropout probability.
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Forward pass.
    ///
    /// In training mode: applies dropout with probability `p`.
    /// In eval mode: returns input unchanged.
    ///
    /// Note: `set_training` / `is_training` are also available via the
    /// [`TrainMode`] trait for generic mode switching.
    pub fn forward<R, C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = numr::dtype::DType>,
        C: RuntimeClient<R> + TensorOps<R> + RandomOps<R> + ScalarOps<R> + BinaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R>,
    {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let (output, _mask) =
            var_dropout(input, self.p, client).map_err(crate::error::Error::Numr)?;
        Ok(output)
    }
}

impl TrainMode for Dropout {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn test_dropout_eval_mode_is_identity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device),
            false,
        );

        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let output = dropout.forward(&client, &input).unwrap();
        let data: Vec<f32> = output.tensor().to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dropout_training_mode_zeroes_elements() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 1000], &[1000], &device),
            false,
        );

        let dropout = Dropout::new(0.5);
        let output = dropout.forward(&client, &input).unwrap();
        let data: Vec<f32> = output.tensor().to_vec();

        let zeros = data.iter().filter(|&&v| v == 0.0).count();
        // Roughly half should be zero
        assert!(zeros > 300 && zeros < 700, "zeros: {zeros}");
    }

    #[test]
    fn test_dropout_zero_prob() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );

        let dropout = Dropout::new(0.0);
        let output = dropout.forward(&client, &input).unwrap();
        let data: Vec<f32> = output.tensor().to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "dropout probability must be in [0, 1]")]
    fn test_dropout_invalid_prob() {
        Dropout::new(1.5);
    }
}
