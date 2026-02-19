//! Training configuration and metrics

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
