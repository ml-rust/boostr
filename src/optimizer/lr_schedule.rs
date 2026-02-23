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
    /// Warmup-Stable-Decay (WSD)
    ///
    /// Three phases used by Llama 3, DeepSeek-V3, and most frontier models:
    /// 1. Linear warmup from 0 to `base_lr`
    /// 2. Constant at `base_lr`
    /// 3. Decay from `base_lr` to `min_lr` using the chosen decay shape
    Wsd {
        base_lr: f64,
        min_lr: f64,
        warmup_steps: u64,
        stable_steps: u64,
        decay_steps: u64,
        decay_shape: DecayShape,
    },
    /// Step decay: `lr = initial_lr * decay_factor ^ (step / decay_steps)`
    ///
    /// Drops LR by a fixed factor every `decay_steps` steps. Standard in vision training.
    StepDecay {
        initial_lr: f64,
        decay_factor: f64,
        decay_steps: u64,
    },
    /// Exponential decay: `lr = initial_lr * exp(-decay_rate * step)`
    ExponentialDecay { initial_lr: f64, decay_rate: f64 },
    /// Cyclical LR: triangular wave between `min_lr` and `max_lr`
    ///
    /// Used in super-convergence (Smith, 2018).
    CyclicalLr {
        min_lr: f64,
        max_lr: f64,
        cycle_length: u64,
    },
    /// Warm restarts (SGDR): periodic cosine annealing with growing period
    ///
    /// Each restart doubles the period and optionally decays the amplitude.
    /// Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", 2017.
    WarmRestarts {
        base_lr: f64,
        min_lr: f64,
        first_period: u64,
        period_mult: f64,
        lr_decay: f64,
    },
    /// Custom schedule via closure
    Lambda(Box<dyn Fn(u64) -> f64 + Send + Sync>),
    /// Inverse square root decay (original Transformer schedule)
    ///
    /// `lr = base_lr * min(step^(-0.5), step * warmup_steps^(-1.5))`
    InverseSqrt { base_lr: f64, warmup_steps: u64 },
}

/// Decay shape for the WSD scheduler's decay phase
#[derive(Debug, Clone, Copy)]
pub enum DecayShape {
    /// Linear decay: `lr = base + (1 - progress) * (base - min)`
    Linear,
    /// Cosine decay: `lr = min + 0.5 * (base - min) * (1 + cos(pi * progress))`
    Cosine,
    /// Square root decay: `lr = min + (base - min) * (1 - sqrt(progress))`
    Sqrt,
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

            LrSchedule::Wsd {
                base_lr,
                min_lr,
                warmup_steps,
                stable_steps,
                decay_steps,
                decay_shape,
            } => {
                let decay_start = warmup_steps + stable_steps;
                let total = decay_start + decay_steps;

                if step < *warmup_steps {
                    // Phase 1: linear warmup
                    if *warmup_steps == 0 {
                        *base_lr
                    } else {
                        base_lr * (step as f64 / *warmup_steps as f64)
                    }
                } else if step < decay_start {
                    // Phase 2: stable
                    *base_lr
                } else if step >= total {
                    // Past end
                    *min_lr
                } else {
                    // Phase 3: decay
                    let progress = (step - decay_start) as f64 / *decay_steps as f64;
                    match decay_shape {
                        DecayShape::Linear => min_lr + (base_lr - min_lr) * (1.0 - progress),
                        DecayShape::Cosine => {
                            min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (PI * progress).cos())
                        }
                        DecayShape::Sqrt => min_lr + (base_lr - min_lr) * (1.0 - progress.sqrt()),
                    }
                }
            }

            LrSchedule::StepDecay {
                initial_lr,
                decay_factor,
                decay_steps,
            } => {
                let exponent = step / decay_steps;
                initial_lr * decay_factor.powi(exponent as i32)
            }

            LrSchedule::ExponentialDecay {
                initial_lr,
                decay_rate,
            } => initial_lr * (-decay_rate * step as f64).exp(),

            LrSchedule::CyclicalLr {
                min_lr,
                max_lr,
                cycle_length,
            } => {
                let half = *cycle_length / 2;
                if half == 0 {
                    return *max_lr;
                }
                let pos = step % cycle_length;
                let progress = if pos < half {
                    pos as f64 / half as f64
                } else {
                    1.0 - (pos - half) as f64 / (*cycle_length - half) as f64
                };
                min_lr + (max_lr - min_lr) * progress
            }

            LrSchedule::WarmRestarts {
                base_lr,
                min_lr,
                first_period,
                period_mult,
                lr_decay,
            } => {
                if *first_period == 0 {
                    return *min_lr;
                }
                // Find which restart cycle we're in
                let mut elapsed = step;
                let mut period = *first_period;
                let mut cycle = 0u32;
                loop {
                    if elapsed < period {
                        break;
                    }
                    elapsed -= period;
                    cycle += 1;
                    period = (period as f64 * period_mult) as u64;
                    if period == 0 {
                        return *min_lr;
                    }
                }
                let amplitude = base_lr * lr_decay.powi(cycle as i32);
                let progress = elapsed as f64 / period as f64;
                min_lr + (amplitude - min_lr) * 0.5 * (1.0 + (PI * progress).cos())
            }

            LrSchedule::Lambda(f) => f(step),

            LrSchedule::InverseSqrt {
                base_lr,
                warmup_steps,
            } => {
                if step == 0 {
                    return 0.0;
                }
                let s = step as f64;
                let w = *warmup_steps as f64;
                base_lr * s.powf(-0.5).min(s * w.powf(-1.5))
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
    fn test_wsd_linear_decay() {
        let sched = LrSchedule::Wsd {
            base_lr: 0.01,
            min_lr: 0.001,
            warmup_steps: 100,
            stable_steps: 400,
            decay_steps: 500,
            decay_shape: DecayShape::Linear,
        };

        // Warmup
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-10);
        assert!((sched.get_lr(50) - 0.005).abs() < 1e-10);

        // Stable phase
        assert!((sched.get_lr(100) - 0.01).abs() < 1e-10);
        assert!((sched.get_lr(300) - 0.01).abs() < 1e-10);
        assert!((sched.get_lr(499) - 0.01).abs() < 1e-10);

        // Decay start
        assert!((sched.get_lr(500) - 0.01).abs() < 1e-6);

        // Decay midpoint: min + (base - min) * 0.5 = 0.001 + 0.009 * 0.5 = 0.0055
        assert!((sched.get_lr(750) - 0.0055).abs() < 1e-6);

        // At end
        assert!((sched.get_lr(1000) - 0.001).abs() < 1e-6);

        // Past end
        assert!((sched.get_lr(2000) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_wsd_cosine_decay() {
        let sched = LrSchedule::Wsd {
            base_lr: 0.01,
            min_lr: 0.0,
            warmup_steps: 0,
            stable_steps: 500,
            decay_steps: 500,
            decay_shape: DecayShape::Cosine,
        };

        // Stable
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-10);
        assert!((sched.get_lr(499) - 0.01).abs() < 1e-10);

        // Decay midpoint: cosine midpoint = 0.5 * base = 0.005
        assert!((sched.get_lr(750) - 0.005).abs() < 1e-4);

        // End
        assert!((sched.get_lr(1000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wsd_sqrt_decay() {
        let sched = LrSchedule::Wsd {
            base_lr: 0.01,
            min_lr: 0.0,
            warmup_steps: 0,
            stable_steps: 0,
            decay_steps: 100,
            decay_shape: DecayShape::Sqrt,
        };

        // Start: progress=0, sqrt(0)=0, lr = 0 + 0.01 * (1 - 0) = 0.01
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-10);

        // progress=0.25, sqrt(0.25)=0.5, lr = 0.01 * (1 - 0.5) = 0.005
        assert!((sched.get_lr(25) - 0.005).abs() < 1e-6);

        // End
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_decay() {
        let sched = LrSchedule::StepDecay {
            initial_lr: 0.1,
            decay_factor: 0.1,
            decay_steps: 30,
        };
        // step 0-29: 0.1 * 0.1^0 = 0.1
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((sched.get_lr(29) - 0.1).abs() < 1e-10);
        // step 30-59: 0.1 * 0.1^1 = 0.01
        assert!((sched.get_lr(30) - 0.01).abs() < 1e-10);
        // step 60-89: 0.1 * 0.1^2 = 0.001
        assert!((sched.get_lr(60) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay() {
        let sched = LrSchedule::ExponentialDecay {
            initial_lr: 0.1,
            decay_rate: 0.01,
        };
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
        // step 100: 0.1 * exp(-1.0) ≈ 0.03679
        assert!((sched.get_lr(100) - 0.1 * (-1.0f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_cyclical_lr() {
        let sched = LrSchedule::CyclicalLr {
            min_lr: 0.001,
            max_lr: 0.01,
            cycle_length: 100,
        };
        // Start: min
        assert!((sched.get_lr(0) - 0.001).abs() < 1e-10);
        // Midpoint of ascent (step 25 of half=50): progress=0.5
        assert!((sched.get_lr(25) - 0.0055).abs() < 1e-6);
        // Peak (step 50): max
        assert!((sched.get_lr(50) - 0.01).abs() < 1e-6);
        // Back to min at cycle end
        assert!((sched.get_lr(100) - 0.001).abs() < 1e-6);
        // Periodic: step 200 = step 0
        assert!((sched.get_lr(200) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_warm_restarts() {
        let sched = LrSchedule::WarmRestarts {
            base_lr: 0.01,
            min_lr: 0.0,
            first_period: 10,
            period_mult: 2.0,
            lr_decay: 1.0,
        };
        // Start of first cycle: base_lr (cos(0)=1 -> min + base * 0.5 * 2 = base)
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-6);
        // End of first cycle (step 10): start of second cycle, resets to base
        assert!((sched.get_lr(10) - 0.01).abs() < 1e-6);
        // Second cycle has period 20, so step 30 = start of third cycle
        assert!((sched.get_lr(30) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_warm_restarts_lr_decay() {
        let sched = LrSchedule::WarmRestarts {
            base_lr: 0.01,
            min_lr: 0.0,
            first_period: 10,
            period_mult: 1.0,
            lr_decay: 0.5,
        };
        // Cycle 0: amplitude = 0.01
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-6);
        // Cycle 1: amplitude = 0.005
        assert!((sched.get_lr(10) - 0.005).abs() < 1e-6);
        // Cycle 2: amplitude = 0.0025
        assert!((sched.get_lr(20) - 0.0025).abs() < 1e-6);
    }

    #[test]
    fn test_lambda() {
        let sched = LrSchedule::Lambda(Box::new(|step| 0.1 / (1.0 + step as f64)));
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((sched.get_lr(9) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_sqrt() {
        let sched = LrSchedule::InverseSqrt {
            base_lr: 0.01,
            warmup_steps: 4000,
        };
        // step 0: returns 0
        assert_eq!(sched.get_lr(0), 0.0);
        // During warmup (step < warmup): step * warmup^(-1.5) < step^(-0.5)
        // step 1000: 0.01 * 1000 * 4000^(-1.5) = 0.01 * 1000 / 252982.2... ≈ 3.95e-5
        let lr_1000 = sched.get_lr(1000);
        let lr_4000 = sched.get_lr(4000);
        // At warmup_steps, both terms equal: step^(-0.5) = step * warmup^(-1.5)
        // 4000^(-0.5) = 4000 * 4000^(-1.5) = 4000^(-0.5) ✓
        assert!(lr_1000 < lr_4000, "LR should increase during warmup");
        // After warmup: step^(-0.5) decay
        let lr_16000 = sched.get_lr(16000);
        assert!(lr_16000 < lr_4000, "LR should decay after warmup");
        // 16000^(-0.5) / 4000^(-0.5) = sqrt(4000/16000) = 0.5
        assert!((lr_16000 / lr_4000 - 0.5).abs() < 1e-6);
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
