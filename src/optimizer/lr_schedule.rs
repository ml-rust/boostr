//! Learning rate schedulers
//!
//! Standard schedules for adjusting learning rate during training.

use std::f64::consts::PI;

/// Learning rate schedule
pub enum LrSchedule {
    /// Constant learning rate
    Constant { lr: f64 },
    /// Linear warmup then constant
    LinearWarmup { base_lr: f64, warmup_steps: u64 },
    /// Cosine annealing with optional warmup
    CosineAnnealing {
        base_lr: f64,
        min_lr: f64,
        warmup_steps: u64,
        total_steps: u64,
    },
}

impl LrSchedule {
    /// Get the learning rate for a given step
    pub fn get_lr(&self, step: u64) -> f64 {
        match self {
            LrSchedule::Constant { lr } => *lr,

            LrSchedule::LinearWarmup {
                base_lr,
                warmup_steps,
            } => {
                if *warmup_steps == 0 || step >= *warmup_steps {
                    *base_lr
                } else {
                    base_lr * (step as f64 / *warmup_steps as f64)
                }
            }

            LrSchedule::CosineAnnealing {
                base_lr,
                min_lr,
                warmup_steps,
                total_steps,
            } => {
                if step < *warmup_steps {
                    // Linear warmup phase
                    if *warmup_steps == 0 {
                        *base_lr
                    } else {
                        base_lr * (step as f64 / *warmup_steps as f64)
                    }
                } else if step >= *total_steps {
                    *min_lr
                } else {
                    // Cosine decay phase
                    let decay_steps = total_steps - warmup_steps;
                    let progress = (step - warmup_steps) as f64 / decay_steps as f64;
                    min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (PI * progress).cos())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let sched = LrSchedule::Constant { lr: 0.001 };
        assert_eq!(sched.get_lr(0), 0.001);
        assert_eq!(sched.get_lr(1000), 0.001);
    }

    #[test]
    fn test_linear_warmup() {
        let sched = LrSchedule::LinearWarmup {
            base_lr: 0.01,
            warmup_steps: 100,
        };
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-10);
        assert!((sched.get_lr(50) - 0.005).abs() < 1e-10);
        assert!((sched.get_lr(100) - 0.01).abs() < 1e-10);
        assert!((sched.get_lr(200) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let sched = LrSchedule::CosineAnnealing {
            base_lr: 0.01,
            min_lr: 0.001,
            warmup_steps: 100,
            total_steps: 1100,
        };

        // During warmup: linear
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-10);
        assert!((sched.get_lr(50) - 0.005).abs() < 1e-10);

        // At warmup end: base_lr
        assert!((sched.get_lr(100) - 0.01).abs() < 1e-6);

        // Midpoint: average of base_lr and min_lr
        assert!((sched.get_lr(600) - 0.0055).abs() < 1e-4);

        // At end: min_lr
        assert!((sched.get_lr(1100) - 0.001).abs() < 1e-6);

        // Past end: min_lr
        assert!((sched.get_lr(2000) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_no_warmup() {
        let sched = LrSchedule::CosineAnnealing {
            base_lr: 0.01,
            min_lr: 0.0,
            warmup_steps: 0,
            total_steps: 1000,
        };
        // Start at base_lr
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-6);
        // End at min_lr
        assert!((sched.get_lr(1000) - 0.0).abs() < 1e-6);
    }
}
