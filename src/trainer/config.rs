//! Training configuration and metrics

use crate::error::Result;
use crate::optimizer::grad_scaler::GradScaler;
use numr::dtype::DType;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub max_grad_norm: Option<f64>,
    pub grad_accum_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
            grad_accum_steps: 1,
        }
    }
}

impl TrainingConfig {
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn with_max_grad_norm(mut self, norm: Option<f64>) -> Self {
        self.max_grad_norm = norm;
        self
    }

    pub fn with_grad_accum_steps(mut self, steps: usize) -> Self {
        self.grad_accum_steps = steps;
        self
    }
}

/// Metrics from a training step
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: u64,
    pub loss: f64,
    pub grad_norm: Option<f64>,
    pub lr: f64,
}

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Dtype for forward/backward computation (BF16 or F16)
    pub compute_dtype: DType,
    /// Dtype for master weights (always F32)
    pub master_dtype: DType,
    /// Loss scaling strategy
    pub loss_scale: LossScaleStrategy,
}

impl MixedPrecisionConfig {
    /// BF16 mixed precision (recommended).
    ///
    /// No loss scaling needed — BF16 has the same exponent range as F32.
    pub fn bf16() -> Self {
        Self {
            compute_dtype: DType::BF16,
            master_dtype: DType::F32,
            loss_scale: LossScaleStrategy::None,
        }
    }

    /// FP16 mixed precision with dynamic loss scaling.
    ///
    /// Requires loss scaling because FP16 has a narrower exponent range.
    pub fn fp16() -> Self {
        Self {
            compute_dtype: DType::F16,
            master_dtype: DType::F32,
            loss_scale: LossScaleStrategy::Dynamic {
                initial_scale: 65536.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
            },
        }
    }
}

/// Loss scaling strategy for mixed precision training
#[derive(Debug, Clone)]
pub enum LossScaleStrategy {
    /// No loss scaling (for BF16)
    None,
    /// Fixed loss scale factor
    Fixed(f64),
    /// Dynamic loss scaling (for FP16)
    Dynamic {
        initial_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: u64,
    },
}

impl LossScaleStrategy {
    /// Convert to an optional GradScaler.
    pub(crate) fn to_grad_scaler(&self) -> Result<Option<GradScaler>> {
        match self {
            LossScaleStrategy::None => Ok(None),
            LossScaleStrategy::Fixed(scale) => {
                // Fixed scale: use huge growth interval so it never changes
                Ok(Some(GradScaler::new(*scale, 2.0, 0.5, u64::MAX)?))
            }
            LossScaleStrategy::Dynamic {
                initial_scale,
                growth_factor,
                backoff_factor,
                growth_interval,
            } => Ok(Some(GradScaler::new(
                *initial_scale,
                *growth_factor,
                *backoff_factor,
                *growth_interval,
            )?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.weight_decay, 0.01);
        assert_eq!(config.max_grad_norm, Some(1.0));
        assert_eq!(config.grad_accum_steps, 1);
    }

    #[test]
    fn test_builder() {
        let config = TrainingConfig::default()
            .with_lr(0.01)
            .with_weight_decay(0.1)
            .with_max_grad_norm(None)
            .with_grad_accum_steps(4);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.weight_decay, 0.1);
        assert_eq!(config.max_grad_norm, None);
        assert_eq!(config.grad_accum_steps, 4);
    }
}
