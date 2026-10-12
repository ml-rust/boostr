//! Fitting one answer temperature to labelled logits by temperature scaling.
//!
//! The fit minimises the mean negative log-likelihood over `beta = 1 / T` with
//! projected, backtracking Newton steps. The objective is convex in `beta`
//! (see [`super::objective`]). Only `T > 0` is searched, so the fit never
//! changes an argmax.

use super::objective::{CalibrationSample, evaluate, is_flat, validate_samples};
use crate::error::{Error, Result};

/// The most Newton iterations a fit runs.
pub const MAX_ITERATIONS: usize = 100;
/// The most step halvings per iteration.
const MAX_HALVINGS: usize = 60;
/// Relative NLL rise treated as rounding error in the line search.
const NLL_ROUNDING: f64 = 1e-12;

/// The temperature range a fit searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureBounds {
    /// Lowest temperature. Must be finite and > 0.
    pub min: f64,
    /// Highest temperature. Must be finite and > `min`.
    pub max: f64,
}

impl Default for TemperatureBounds {
    /// `[0.05, 100]`. The ceiling gives near-uniform distributions for
    /// logit gaps of order 1.
    fn default() -> Self {
        Self {
            min: 0.05,
            max: 100.0,
        }
    }
}

/// How a fit ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitOutcome {
    /// The step or the gradient fell below tolerance.
    Converged,
    /// The constrained minimum lies on a bound.
    Bound,
    /// Every sample has equal logits, so every temperature gives the same
    /// likelihood. The start temperature is returned.
    Flat,
    /// [`MAX_ITERATIONS`] ran out.
    IterationLimit,
}

/// The result of a temperature fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureFit {
    /// The fitted temperature, inside the bounds.
    pub t: f64,
    /// Mean NLL at T = 1.
    pub nll_before: f64,
    /// Mean NLL at `t`.
    pub nll_after: f64,
    /// Newton iterations run.
    pub iterations: usize,
    /// `t` equals a bound.
    pub at_bound: bool,
    /// How the fit ended.
    pub outcome: FitOutcome,
}

fn check_bounds(b: TemperatureBounds) -> Result<()> {
    if b.min.is_finite() && b.max.is_finite() && b.min > 0.0 && b.max > b.min {
        Ok(())
    } else {
        Err(Error::InvalidArgument {
            arg: "bounds",
            reason: format!("need finite 0 < min < max, got min {} max {}", b.min, b.max),
        })
    }
}

/// The Newton step `-g / h`, with an infinite step toward the descent side when `h` is 0.
fn newton_step(grad: f64, hess: f64) -> f64 {
    if hess > 0.0 {
        -grad / hess
    } else if grad < 0.0 {
        f64::INFINITY
    } else if grad > 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Fits one temperature to `samples` within `bounds`. The search starts at T = 1,
/// clamped into the bounds.
pub fn fit_temperature(
    samples: &[CalibrationSample],
    bounds: TemperatureBounds,
) -> Result<TemperatureFit> {
    validate_samples(samples)?;
    check_bounds(bounds)?;
    let (beta_lo, beta_hi) = (1.0 / bounds.max, 1.0 / bounds.min);
    let nll_before = evaluate(samples, 1.0).nll;
    let mut beta = 1.0_f64.clamp(beta_lo, beta_hi);
    let mut point = evaluate(samples, beta);

    if is_flat(samples) {
        return Ok(finish(
            beta,
            point.nll,
            nll_before,
            0,
            bounds,
            FitOutcome::Flat,
        ));
    }

    for iteration in 1..=MAX_ITERATIONS {
        let target = (beta + newton_step(point.grad, point.hess)).clamp(beta_lo, beta_hi);
        if target == beta {
            let outcome = if beta == beta_lo || beta == beta_hi {
                FitOutcome::Bound
            } else {
                FitOutcome::Converged
            };
            return Ok(finish(
                beta,
                point.nll,
                nll_before,
                iteration - 1,
                bounds,
                outcome,
            ));
        }
        let mut step = target - beta;
        let mut accepted = None;
        // Near the minimum the NLL change falls below its rounding error. A step
        // that shrinks |grad| and raises NLL by no more than that error is kept.
        let noise = NLL_ROUNDING * point.nll.abs().max(1.0);
        for _ in 0..=MAX_HALVINGS {
            let candidate = beta + step;
            if candidate > 0.0 {
                let next = evaluate(samples, candidate);
                let descends = next.nll <= point.nll;
                let settles = next.grad.abs() < point.grad.abs() && next.nll <= point.nll + noise;
                if descends || settles {
                    accepted = Some((candidate, next));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((next_beta, next)) = accepted else {
            return Ok(finish(
                beta,
                point.nll,
                nll_before,
                iteration,
                bounds,
                FitOutcome::Converged,
            ));
        };
        let moved = (next_beta - beta).abs();
        beta = next_beta;
        point = next;
        let on_bound = beta == beta_lo || beta == beta_hi;
        if moved <= 1e-12 * beta.max(1.0) || point.grad.abs() <= 1e-14 {
            let outcome = if on_bound {
                FitOutcome::Bound
            } else {
                FitOutcome::Converged
            };
            return Ok(finish(
                beta, point.nll, nll_before, iteration, bounds, outcome,
            ));
        }
    }
    Ok(finish(
        beta,
        point.nll,
        nll_before,
        MAX_ITERATIONS,
        bounds,
        FitOutcome::IterationLimit,
    ))
}

fn finish(
    beta: f64,
    nll_after: f64,
    nll_before: f64,
    iterations: usize,
    bounds: TemperatureBounds,
    outcome: FitOutcome,
) -> TemperatureFit {
    // `beta` sits exactly on `1 / min` or `1 / max` after a clamp; report the bound itself.
    let t = if beta >= 1.0 / bounds.min {
        bounds.min
    } else if beta <= 1.0 / bounds.max {
        bounds.max
    } else {
        (1.0 / beta).clamp(bounds.min, bounds.max)
    };
    let at_bound = outcome == FitOutcome::Bound || t == bounds.min || t == bounds.max;
    TemperatureFit {
        t,
        nll_before,
        nll_after,
        iterations,
        at_bound,
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::super::objective::Target;
    use super::*;

    /// 64-bit LCG (Knuth MMIX constants); `uniform` gives a uniform in `[0, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn reference_probs(z: &[f64], t: f64) -> Vec<f64> {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = z.iter().map(|v| ((v - m) / t).exp()).collect();
        let s: f64 = w.iter().sum();
        w.iter().map(|v| v / s).collect()
    }

    /// Logits uniform in [-4, 4); labels drawn from softmax(z / t_star).
    fn drawn(n: usize, k: usize, t_star: f64, seed: u64) -> Vec<CalibrationSample> {
        let mut g = Lcg(seed);
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..k).map(|_| g.uniform() * 8.0 - 4.0).collect();
                let p = reference_probs(&z, t_star);
                let r = g.uniform();
                let mut c = 0.0;
                let mut label = k - 1;
                for (i, pi) in p.iter().enumerate() {
                    c += pi;
                    if r < c {
                        label = i;
                        break;
                    }
                }
                CalibrationSample {
                    logits: z,
                    target: Target::Hard(label),
                }
            })
            .collect()
    }

    #[test]
    fn recovers_a_known_temperature() {
        let samples = drawn(4000, 4, 2.0, 1);
        let fit = fit_temperature(&samples, TemperatureBounds::default()).unwrap();
        assert!((fit.t - 2.0).abs() < 0.1, "t = {}", fit.t);
        assert!(fit.nll_after < fit.nll_before);
        assert!(!fit.at_bound);
        assert_eq!(fit.outcome, FitOutcome::Converged);
        // The gradient vanishes at the fit.
        assert!(evaluate(&samples, 1.0 / fit.t).grad.abs() < 1e-9);
    }

    #[test]
    fn matches_a_log_grid_minimiser() {
        let samples = drawn(500, 3, 0.7, 7);
        let fit = fit_temperature(&samples, TemperatureBounds::default()).unwrap();
        let (lo, hi) = (0.05f64.ln(), 100f64.ln());
        let steps = 4000;
        let (mut best_t, mut best_nll) = (1.0, f64::INFINITY);
        for i in 0..=steps {
            let t = (lo + (hi - lo) * i as f64 / steps as f64).exp();
            let nll = evaluate(&samples, 1.0 / t).nll;
            if nll < best_nll {
                best_t = t;
                best_nll = nll;
            }
        }
        let grid_ratio = ((hi - lo) / steps as f64).exp();
        assert!(fit.t / best_t < grid_ratio && best_t / fit.t < grid_ratio);
        assert!(fit.nll_after <= best_nll + 1e-12);
    }

    #[test]
    fn soft_targets_recover_exactly() {
        let mut g = Lcg(42);
        let samples: Vec<CalibrationSample> = (0..50)
            .map(|_| {
                let z: Vec<f64> = (0..5).map(|_| g.uniform() * 6.0 - 3.0).collect();
                let y = reference_probs(&z, 1.7);
                CalibrationSample {
                    logits: z,
                    target: Target::Soft(y),
                }
            })
            .collect();
        let fit = fit_temperature(&samples, TemperatureBounds::default()).unwrap();
        assert!((fit.t - 1.7).abs() < 1e-8, "t = {}", fit.t);
    }

    #[test]
    fn ragged_option_counts() {
        let mut g = Lcg(3);
        let samples: Vec<CalibrationSample> = (0..60)
            .map(|i| {
                let k = [2, 3, 5][i % 3];
                let z: Vec<f64> = (0..k).map(|_| g.uniform() * 6.0 - 3.0).collect();
                let y = reference_probs(&z, 0.6);
                CalibrationSample {
                    logits: z,
                    target: Target::Soft(y),
                }
            })
            .collect();
        let fit = fit_temperature(&samples, TemperatureBounds::default()).unwrap();
        assert!((fit.t - 0.6).abs() < 1e-8, "t = {}", fit.t);
    }

    #[test]
    fn separable_data_hits_the_lower_bound() {
        let mut g = Lcg(11);
        let samples: Vec<CalibrationSample> = (0..40)
            .map(|_| {
                let z: Vec<f64> = (0..4).map(|_| g.uniform() * 4.0 - 2.0).collect();
                let label = (0..4).fold(0, |b, i| if z[i] > z[b] { i } else { b });
                CalibrationSample {
                    logits: z,
                    target: Target::Hard(label),
                }
            })
            .collect();
        let bounds = TemperatureBounds {
            min: 0.05,
            max: 20.0,
        };
        let fit = fit_temperature(&samples, bounds).unwrap();
        assert_eq!(fit.t, 0.05);
        assert!(fit.at_bound);
        assert_eq!(fit.outcome, FitOutcome::Bound);
        assert!(fit.nll_after < fit.nll_before);
    }

    #[test]
    fn identical_logits_are_flat() {
        let samples = vec![
            CalibrationSample {
                logits: vec![0.5; 3],
                target: Target::Hard(1),
            },
            CalibrationSample {
                logits: vec![-2.0; 2],
                target: Target::Soft(vec![0.3, 0.7]),
            },
        ];
        let fit = fit_temperature(&samples, TemperatureBounds::default()).unwrap();
        assert_eq!(fit.t, 1.0);
        assert_eq!(fit.outcome, FitOutcome::Flat);
        assert!(!fit.at_bound);
        assert!(fit.nll_after.is_finite());
        assert_eq!(fit.nll_after, fit.nll_before);
        // A start outside the bounds is clamped in.
        let fit = fit_temperature(&samples, TemperatureBounds { min: 2.0, max: 5.0 }).unwrap();
        assert_eq!(fit.t, 2.0);
        assert!(fit.at_bound);
    }

    #[test]
    fn invalid_input_errors() {
        let ok = vec![CalibrationSample {
            logits: vec![0.0, 1.0],
            target: Target::Hard(1),
        }];
        assert!(fit_temperature(&[], TemperatureBounds::default()).is_err());
        for (min, max) in [
            (0.0, 1.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (f64::NAN, 1.0),
            (0.1, f64::INFINITY),
        ] {
            assert!(fit_temperature(&ok, TemperatureBounds { min, max }).is_err());
        }
        let one = vec![CalibrationSample {
            logits: vec![1.0],
            target: Target::Hard(0),
        }];
        assert!(fit_temperature(&one, TemperatureBounds::default()).is_err());
        let out_of_range = vec![CalibrationSample {
            logits: vec![0.0, 1.0],
            target: Target::Hard(5),
        }];
        assert!(fit_temperature(&out_of_range, TemperatureBounds::default()).is_err());
        let nan = vec![CalibrationSample {
            logits: vec![0.0, f64::NAN],
            target: Target::Hard(0),
        }];
        assert!(fit_temperature(&nan, TemperatureBounds::default()).is_err());
        let bad_soft = vec![CalibrationSample {
            logits: vec![0.0, 1.0],
            target: Target::Soft(vec![0.2, 0.2]),
        }];
        assert!(fit_temperature(&bad_soft, TemperatureBounds::default()).is_err());
    }
}
