//! Dynamic loss scaling for FP16 mixed precision training
//!
//! FP16 has a narrower exponent range than FP32/BF16, so gradients can underflow.
//! GradScaler multiplies the loss by a large scale factor before backward, then
//! divides gradients by that factor. If NaN/Inf is detected, the step is skipped
//! and the scale is halved.
//!
//! Not needed for BF16 (same exponent range as FP32).

use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// Result of unscaling gradients
pub enum UnscaleResult<R: Runtime> {
    /// Gradients are valid after unscaling
    Ok(GradStore<R>),
    /// Overflow/NaN detected — skip this optimizer step
    Overflow,
}

/// Dynamic loss scaler for FP16 training
///
/// Maintains a scale factor that grows when training is stable and shrinks
/// when overflow is detected.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: u64,
    consecutive_ok: u64,
}

impl GradScaler {
    /// Create a new GradScaler with dynamic scaling parameters.
    ///
    /// # Arguments
    /// * `initial_scale` - Starting loss scale (e.g., 2^16 = 65536)
    /// * `growth_factor` - Multiply scale by this after `growth_interval` clean steps (e.g., 2.0)
    /// * `backoff_factor` - Multiply scale by this on overflow (e.g., 0.5)
    /// * `growth_interval` - Number of consecutive clean steps before growing scale
    pub fn new(
        initial_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: u64,
    ) -> Result<Self> {
        if initial_scale <= 0.0 {
            return Err(Error::TrainingError {
                reason: format!("initial_scale must be positive, got {initial_scale}"),
            });
        }
        if growth_factor <= 1.0 {
            return Err(Error::TrainingError {
                reason: format!("growth_factor must be > 1.0, got {growth_factor}"),
            });
        }
        if backoff_factor <= 0.0 || backoff_factor >= 1.0 {
            return Err(Error::TrainingError {
                reason: format!("backoff_factor must be in (0, 1), got {backoff_factor}"),
            });
        }
        if growth_interval == 0 {
            return Err(Error::TrainingError {
                reason: "growth_interval must be > 0".to_string(),
            });
        }

        Ok(Self {
            scale: initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            consecutive_ok: 0,
        })
    }

    /// Create with sensible defaults: scale=65536, grow=2x, backoff=0.5x, interval=2000
    pub fn default_fp16() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
        }
    }

    /// Get the current loss scale factor.
    ///
    /// Multiply your loss by this before calling `backward()`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Scale a loss value before backward pass.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        loss * self.scale
    }

    /// Unscale gradients and check for NaN/Inf.
    ///
    /// Divides all gradients by the current scale factor. If any gradient
    /// contains NaN or Inf after unscaling, returns `Overflow` to signal
    /// the optimizer step should be skipped.
    pub fn unscale_grads<R, C>(&self, client: &C, grads: GradStore<R>) -> Result<UnscaleResult<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ScalarOps<R> + UnaryOps<R> + ReduceOps<R>,
    {
        let inv_scale = 1.0 / self.scale;
        let ids: Vec<TensorId> = grads.keys().copied().collect();
        let mut unscaled = GradStore::new();

        for id in ids {
            let grad = grads.get(id).ok_or_else(|| Error::TrainingError {
                reason: format!("missing gradient for tensor {id:?}"),
            })?;
            let g = client.mul_scalar(grad, inv_scale)?;

            if Self::has_nan_inf(client, &g)? {
                return Ok(UnscaleResult::Overflow);
            }

            unscaled.insert(id, g);
        }

        Ok(UnscaleResult::Ok(unscaled))
    }

    /// Update the scale factor after an optimizer step.
    ///
    /// Call with `overflow=true` if `unscale_grads` returned `Overflow`.
    /// Call with `overflow=false` after a successful optimizer step.
    pub fn update_scale(&mut self, overflow: bool) {
        if overflow {
            self.scale *= self.backoff_factor;
            self.consecutive_ok = 0;
        } else {
            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_ok = 0;
            }
        }
    }

    /// Check if a tensor contains NaN or Inf values on-device.
    fn has_nan_inf<R, C>(client: &C, tensor: &numr::tensor::Tensor<R>) -> Result<bool>
    where
        R: Runtime<DType = DType>,
        C: UnaryOps<R> + ReduceOps<R>,
    {
        let nan_mask = client.isnan(tensor)?;
        let inf_mask = client.isinf(tensor)?;
        let has_nan = client.any(&nan_mask, &[], false)?;
        let has_inf = client.any(&inf_mask, &[], false)?;
        Ok(has_nan.item::<u8>()? != 0 || has_inf.item::<u8>()? != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::default_fp16();
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn test_scale_loss() {
        let scaler = GradScaler::default_fp16();
        assert_eq!(scaler.scale_loss(1.0), 65536.0);
        assert_eq!(scaler.scale_loss(0.5), 32768.0);
    }

    #[test]
    fn test_unscale_grads_ok() {
        let (client, device) = cpu_setup();
        let scaler = GradScaler::new(100.0, 2.0, 0.5, 10).unwrap();

        let id = TensorId::new();
        let grad = Tensor::<CpuRuntime>::from_slice(&[200.0f32, 400.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id, grad);

        match scaler.unscale_grads(&client, grads).unwrap() {
            UnscaleResult::Ok(unscaled) => {
                let data = unscaled.get(id).unwrap().to_vec::<f32>();
                assert!((data[0] - 2.0).abs() < 1e-5);
                assert!((data[1] - 4.0).abs() < 1e-5);
            }
            UnscaleResult::Overflow => panic!("expected Ok, got Overflow"),
        }
    }

    #[test]
    fn test_unscale_grads_overflow() {
        let (client, device) = cpu_setup();
        let scaler = GradScaler::new(100.0, 2.0, 0.5, 10).unwrap();

        let id = TensorId::new();
        let grad = Tensor::<CpuRuntime>::from_slice(&[f32::NAN, 1.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id, grad);

        match scaler.unscale_grads(&client, grads).unwrap() {
            UnscaleResult::Overflow => {} // expected
            UnscaleResult::Ok(_) => panic!("expected Overflow, got Ok"),
        }
    }

    #[test]
    fn test_update_scale_growth() {
        let mut scaler = GradScaler::new(100.0, 2.0, 0.5, 3).unwrap();
        assert_eq!(scaler.scale(), 100.0);

        // 3 clean steps → scale should double
        scaler.update_scale(false);
        scaler.update_scale(false);
        assert_eq!(scaler.scale(), 100.0); // grow_interval=3, only 2 clean steps so far
        scaler.update_scale(false);
        assert_eq!(scaler.scale(), 200.0); // grew!
    }

    #[test]
    fn test_update_scale_backoff() {
        let mut scaler = GradScaler::new(100.0, 2.0, 0.5, 3).unwrap();

        // Overflow → scale halves
        scaler.update_scale(true);
        assert_eq!(scaler.scale(), 50.0);

        // Consecutive counter reset
        scaler.update_scale(false);
        scaler.update_scale(false);
        scaler.update_scale(true); // overflow resets
        assert_eq!(scaler.scale(), 25.0);
    }

    #[test]
    fn test_invalid_params() {
        assert!(GradScaler::new(0.0, 2.0, 0.5, 10).is_err());
        assert!(GradScaler::new(100.0, 0.5, 0.5, 10).is_err());
        assert!(GradScaler::new(100.0, 2.0, 1.5, 10).is_err());
        assert!(GradScaler::new(100.0, 2.0, 0.5, 0).is_err());
    }
}
