//! Stochastic depth (drop path) regularization.
//!
//! Drops entire residual block outputs per-sample during training.
//! Used in deep networks (100+ layers) to regularize and speed up training.
//!
//! Unlike dropout (which zeroes individual elements), stochastic depth
//! zeroes entire samples in the batch. Applied to the residual branch
//! output before adding to the skip connection.
//!
//! Reference: Huang et al., "Deep Networks with Stochastic Depth" (2016).

use crate::error::Result;
use crate::nn::module::TrainMode;
use numr::autograd::Var;
use numr::ops::{BinaryOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Stochastic depth (drop path) layer.
///
/// During training, each sample in the batch is independently kept
/// (with probability `1 - drop_prob`) or zeroed out. Surviving samples
/// are scaled by `1 / (1 - drop_prob)` to preserve expected values.
///
/// During evaluation, acts as identity.
///
/// # Usage
///
/// Apply to the residual branch output before the skip connection add:
///
/// ```ignore
/// // In a transformer/ResNet block:
/// let residual = x.clone();
/// let out = self.block.forward(client, &x)?;
/// let out = self.drop_path.forward(client, &out)?;
/// let out = client.add(&residual.tensor(), &out.tensor())?;
/// ```
pub struct StochasticDepth {
    drop_prob: f64,
    training: bool,
}

impl StochasticDepth {
    /// Create a new stochastic depth layer.
    ///
    /// `drop_prob` is the probability of dropping a sample's contribution.
    /// Common values: linearly increase from 0.0 (first layer) to 0.1-0.3
    /// (last layer).
    pub fn new(drop_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&drop_prob),
            "drop probability must be in [0, 1], got {drop_prob}"
        );
        Self {
            drop_prob,
            training: true,
        }
    }

    /// Returns the drop probability.
    pub fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    /// Forward pass.
    ///
    /// Training: randomly zero entire samples, scale survivors by 1/(1-p).
    /// Eval: identity.
    pub fn forward<R, C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = numr::dtype::DType>,
        C: RuntimeClient<R> + TensorOps<R> + RandomOps<R> + ScalarOps<R> + BinaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R>,
    {
        if !self.training || self.drop_prob == 0.0 {
            return Ok(input.clone());
        }

        // For drop_prob >= 1.0, everything is dropped
        if self.drop_prob >= 1.0 {
            let zeros = numr::tensor::Tensor::<R>::zeros(
                input.tensor().shape(),
                input.tensor().dtype(),
                input.tensor().device(),
            );
            return Ok(Var::new(zeros, false));
        }

        let shape = input.tensor().shape().to_vec();
        // Mask shape: [batch, 1, 1, ...] so it broadcasts across all other dims
        let mut mask_shape = vec![1usize; shape.len()];
        if !shape.is_empty() {
            mask_shape[0] = shape[0];
        }

        // Bernoulli mask: 1 with probability (1 - drop_prob), 0 with probability drop_prob
        let keep_prob = 1.0 - self.drop_prob;
        let mask = client.bernoulli(keep_prob, &mask_shape, input.tensor().dtype())?;

        // Scale by 1/(1-p) for inverted stochastic depth
        let scale = 1.0 / keep_prob;
        let scaled_mask = client.mul_scalar(&mask, scale)?;

        // Apply mask (broadcasts across non-batch dims)
        let output = client.mul(input.tensor(), &scaled_mask)?;

        Ok(Var::new(output, false))
    }
}

impl TrainMode for StochasticDepth {
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
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn test_eval_mode_is_identity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let data = vec![1.0f32; 12];
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[3, 4], &device),
            false,
        );

        let mut sd = StochasticDepth::new(0.5);
        sd.set_training(false);

        let output = sd.forward(&client, &input).unwrap();
        let out: Vec<f32> = output.tensor().to_vec();
        assert_eq!(out, data);
    }

    #[test]
    fn test_zero_drop_prob_is_identity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device),
            false,
        );

        let sd = StochasticDepth::new(0.0);
        let output = sd.forward(&client, &input).unwrap();
        let out: Vec<f32> = output.tensor().to_vec();
        assert_eq!(out, data);
    }

    #[test]
    fn test_drops_entire_samples() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // 100 samples, 4 features each
        let data = vec![1.0f32; 400];
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[100, 4], &device),
            false,
        );

        let sd = StochasticDepth::new(0.5);
        let output = sd.forward(&client, &input).unwrap();
        let out: Vec<f32> = output.tensor().to_vec();

        // Each sample should be either all-zero or all-scaled (by 1/(1-0.5)=2.0)
        let mut dropped = 0;
        let mut kept = 0;
        for sample in 0..100 {
            let row: Vec<f32> = out[sample * 4..(sample + 1) * 4].to_vec();
            if row.iter().all(|&v| v == 0.0) {
                dropped += 1;
            } else {
                // Scaled by 2.0
                assert!(
                    row.iter().all(|&v| (v - 2.0).abs() < 1e-5),
                    "sample {sample} has inconsistent values: {row:?}"
                );
                kept += 1;
            }
        }

        assert_eq!(dropped + kept, 100);
        // Roughly half should be dropped (with generous bounds for randomness)
        assert!(dropped > 20 && dropped < 80, "dropped: {dropped}");
    }

    #[test]
    #[should_panic(expected = "drop probability must be in [0, 1]")]
    fn test_invalid_prob() {
        StochasticDepth::new(1.5);
    }
}
