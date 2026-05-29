//! Adaptive Instance Normalization (AdaIN) for 1D sequences.
//!
//! AdaIN is the conditioning primitive used throughout StyleTTS2-family decoders
//! (Kokoro, StyleTTS2 proper). For an activation `x` of shape `[B, C, T]` and a
//! per-sample style vector split into `gamma` and `beta` (both `[B, C]`):
//!
//! ```text
//!     y = gamma.unsqueeze(-1) * instance_norm(x) + beta.unsqueeze(-1)
//! ```
//!
//! `instance_norm` here is group norm with `num_groups = channels` and unit scale /
//! zero shift — we reuse numr's `NormalizationOps::group_norm` rather than
//! introducing a new primitive.
//!
//! Inference-only module; no autograd wrapper. Add a `Var`-based variant when /
//! if Kokoro training lands.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, NormalizationOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Adaptive Instance Normalization for 1D sequences.
///
/// Carries only the epsilon and the channel count — weights live in the
/// style-projection layer that produces `gamma` / `beta` for each forward call.
#[derive(Debug, Clone, Copy)]
pub struct AdaIn1d {
    channels: usize,
    eps: f32,
}

impl AdaIn1d {
    pub fn new(channels: usize, eps: f32) -> Self {
        Self { channels, eps }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass.
    ///
    /// * `x` — activation of shape `[B, C, T]`.
    /// * `gamma` — per-sample scale of shape `[B, C]`.
    /// * `beta` — per-sample shift of shape `[B, C]`.
    ///
    /// Returns a tensor of the same shape as `x`.
    pub fn forward<R, C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        gamma: &Tensor<R>,
        beta: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + NormalizationOps<R> + BinaryOps<R> + UtilityOps<R>,
    {
        let x_shape = x.shape();
        if x_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("AdaIn1d expects rank-3 [B, C, T] input, got {x_shape:?}"),
            });
        }
        let (b, c, _t) = (x_shape[0], x_shape[1], x_shape[2]);
        if c != self.channels {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("AdaIn1d configured for {} channels, got {c}", self.channels),
            });
        }
        let expected_style = [b, c];
        if gamma.shape() != expected_style {
            return Err(Error::InvalidArgument {
                arg: "gamma",
                reason: format!(
                    "AdaIn1d gamma expected shape {expected_style:?}, got {:?}",
                    gamma.shape()
                ),
            });
        }
        if beta.shape() != expected_style {
            return Err(Error::InvalidArgument {
                arg: "beta",
                reason: format!(
                    "AdaIn1d beta expected shape {expected_style:?}, got {:?}",
                    beta.shape()
                ),
            });
        }

        let dtype = x.dtype();
        // Instance-normalize via group_norm(num_groups=channels) with identity scale/shift.
        // num_groups == channels collapses each group to a single channel, matching instance norm.
        let ones = client.fill(&[c], 1.0, dtype).map_err(Error::Numr)?;
        let zeros = client.fill(&[c], 0.0, dtype).map_err(Error::Numr)?;
        let normalized = client
            .group_norm(x, &ones, &zeros, c, self.eps)
            .map_err(Error::Numr)?;

        // Apply per-sample style: gamma [B,C] -> [B,C,1], same for beta. Broadcast over T.
        let gamma_expanded = gamma.reshape(&[b, c, 1]).map_err(Error::Numr)?;
        let beta_expanded = beta.reshape(&[b, c, 1]).map_err(Error::Numr)?;
        let scaled = client
            .mul(&normalized, &gamma_expanded)
            .map_err(Error::Numr)?;
        client.add(&scaled, &beta_expanded).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn forward_shape_is_preserved() {
        let (client, device) = cpu_setup();
        let adain = AdaIn1d::new(4, 1e-5);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.5f32; 2 * 4 * 6], &[2, 4, 6], &device);
        let gamma = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2 * 4], &[2, 4], &device);
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2 * 4], &[2, 4], &device);
        let out = adain.forward(&client, &x, &gamma, &beta).unwrap();
        assert_eq!(out.shape(), &[2, 4, 6]);
    }

    #[test]
    fn unit_style_preserves_normalized_stats() {
        // With gamma=1, beta=0, AdaIN output equals pure instance norm. Each (b, c)
        // slice should have mean ~= 0 across the time axis.
        let (client, device) = cpu_setup();
        let adain = AdaIn1d::new(2, 1e-5);
        let data: Vec<f32> = (0..(2 * 5)).map(|i| i as f32).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 5], &device);
        let gamma = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[1, 2], &device);
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[1, 2], &device);

        let out = adain.forward(&client, &x, &gamma, &beta).unwrap();
        let flat: Vec<f32> = out.to_vec();

        let mean_c0: f32 = flat[0..5].iter().sum::<f32>() / 5.0;
        let mean_c1: f32 = flat[5..10].iter().sum::<f32>() / 5.0;
        assert!(
            mean_c0.abs() < 1e-4,
            "channel 0 mean should be ~0, got {mean_c0}"
        );
        assert!(
            mean_c1.abs() < 1e-4,
            "channel 1 mean should be ~0, got {mean_c1}"
        );
    }

    #[test]
    fn style_rescales_and_shifts() {
        // gamma = 2, beta = 3 → normalized stats shifted accordingly.
        let (client, device) = cpu_setup();
        let adain = AdaIn1d::new(1, 1e-5);
        let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[1, 1, 4], &device);
        let gamma = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1], &device);
        let beta = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1, 1], &device);

        let out = adain.forward(&client, &x, &gamma, &beta).unwrap();
        let flat: Vec<f32> = out.to_vec();
        let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
        // After instance-norm, mean ~0; after *2+3, mean should be ~3.
        assert!((mean - 3.0).abs() < 1e-3, "mean should be ~3, got {mean}");
    }

    #[test]
    fn rejects_wrong_input_rank() {
        let (client, device) = cpu_setup();
        let adain = AdaIn1d::new(4, 1e-5);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 4], &device);
        let gamma = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device);
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 4], &device);
        assert!(adain.forward(&client, &x, &gamma, &beta).is_err());
    }

    #[test]
    fn rejects_mismatched_channel_count() {
        let (client, device) = cpu_setup();
        let adain = AdaIn1d::new(4, 1e-5);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 24], &[2, 3, 4], &device);
        let gamma = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[2, 3], &device);
        assert!(adain.forward(&client, &x, &gamma, &beta).is_err());
    }
}
