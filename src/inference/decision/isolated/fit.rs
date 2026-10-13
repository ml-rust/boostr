//! Fitting one temperature to isolated Score samples, and the bracketed
//! search it runs, which the mixed Score fit shares.
//!
//! The NLL of the combined level distribution is not convex in `beta = 1
//! / T` (see [`super::objective`]), so a Newton search from one start can
//! stop in a local minimum. The fit runs in three steps:
//!
//! 1. Scan: the NLL at [`SCAN_POINTS`] log-spaced temperatures across the
//!    bounds, bounds included. The best point and its two neighbours
//!    bracket a minimum.
//! 2. Golden-section search on `ln beta` inside the bracket. It needs only
//!    NLL values, so it holds where the NLL is locally concave.
//! 3. Newton polish: projected, backtracking Newton steps inside the
//!    bracket while the second derivative is positive, until the step or
//!    the gradient vanishes.
//!
//! The result is the global minimum up to the scan's resolution: a
//! minimum narrower than one scan step can be missed.

use super::objective::{IsolatedSample, evaluate_isolated, is_flat_isolated, validate_isolated};
use crate::error::Result;
use crate::inference::decision::calibrate::{
    FitOutcome, MAX_ITERATIONS, TemperatureBounds, TemperatureFit, backtrack, check_bounds, finish,
};
use crate::inference::decision::objective::ObjectivePoint;

/// The temperatures the bracketing scan evaluates, bounds included.
pub const SCAN_POINTS: usize = 129;
/// The golden-section search stops when its bracket is this narrow in `ln beta`.
const GOLDEN_WIDTH: f64 = 1e-10;
/// The most golden-section steps.
const MAX_GOLDEN: usize = 200;

/// `beta` at scan point `i` of `[lo, hi]`, endpoints exact.
fn scan_beta(i: usize, lo: f64, hi: f64) -> f64 {
    let last = SCAN_POINTS - 1;
    if i == 0 {
        lo
    } else if i >= last {
        hi
    } else {
        (lo.ln() + (hi.ln() - lo.ln()) * i as f64 / last as f64).exp()
    }
}

/// The minimiser of `nll` over `[a, b]` by golden-section search on
/// `ln beta`, and the steps it took.
fn golden(nll: impl Fn(f64) -> f64, a: f64, b: f64) -> (f64, usize) {
    let shrink = (5f64.sqrt() - 1.0) / 2.0;
    let (mut lo, mut hi) = (a.ln(), b.ln());
    let mut x1 = hi - shrink * (hi - lo);
    let mut x2 = lo + shrink * (hi - lo);
    let (mut f1, mut f2) = (nll(x1.exp()), nll(x2.exp()));
    let mut steps = 0;
    while hi - lo > GOLDEN_WIDTH && steps < MAX_GOLDEN {
        if f1 <= f2 {
            hi = x2;
            (x2, f2) = (x1, f1);
            x1 = hi - shrink * (hi - lo);
            f1 = nll(x1.exp());
        } else {
            lo = x1;
            (x1, f1) = (x2, f2);
            x2 = lo + shrink * (hi - lo);
            f2 = nll(x2.exp());
        }
        steps += 1;
    }
    let best = if f1 <= f2 { x1 } else { x2 };
    (best.exp(), steps)
}

/// Fits one temperature to isolated Score `samples` within `bounds` by
/// minimising the mean NLL of the gold level under the combined level
/// distribution. `iterations` counts golden-section and Newton steps.
///
/// # Errors
///
/// [`crate::error::Error::InvalidArgument`] for samples that fail
/// [`validate_isolated`] or bounds that are not finite `0 < min < max`.
pub fn fit_temperature_isolated(
    samples: &[IsolatedSample],
    bounds: TemperatureBounds,
) -> Result<TemperatureFit> {
    validate_isolated(samples)?;
    check_bounds(bounds)?;
    Ok(fit_bracketed(
        |beta| evaluate_isolated(samples, beta),
        is_flat_isolated(samples),
        bounds,
    ))
}

/// The scan, golden-section and Newton fit of an objective `eval` in
/// `beta` that need not be convex. `flat` says the objective does not
/// depend on `beta`. The samples and `bounds` must be checked already.
pub(crate) fn fit_bracketed(
    eval: impl Fn(f64) -> ObjectivePoint + Copy,
    flat: bool,
    bounds: TemperatureBounds,
) -> TemperatureFit {
    let (lo, hi) = (1.0 / bounds.max, 1.0 / bounds.min);
    let nll_before = eval(1.0).nll;
    if flat {
        let beta = 1.0_f64.clamp(lo, hi);
        let nll = eval(beta).nll;
        return finish(beta, nll, nll_before, 0, bounds, FitOutcome::Flat);
    }

    let (mut best, mut best_nll) = (0, f64::INFINITY);
    for i in 0..SCAN_POINTS {
        let nll = eval(scan_beta(i, lo, hi)).nll;
        if nll < best_nll {
            (best, best_nll) = (i, nll);
        }
    }
    let a = scan_beta(best.saturating_sub(1), lo, hi);
    let b = scan_beta(best + 1, lo, hi);
    let (found, mut iterations) = golden(|beta| eval(beta).nll, a, b);

    // The bracket's ends and the scan's best point stay candidates: the
    // golden search never evaluates the ends, where a bound minimum sits.
    let mut beta = found;
    let mut point = eval(beta);
    for candidate in [scan_beta(best, lo, hi), a, b] {
        let at = eval(candidate);
        if at.nll < point.nll {
            (beta, point) = (candidate, at);
        }
    }
    let outward = (beta == lo && point.grad >= 0.0) || (beta == hi && point.grad <= 0.0);
    if outward {
        return finish(
            beta,
            point.nll,
            nll_before,
            iterations,
            bounds,
            FitOutcome::Bound,
        );
    }

    let mut polish = 0;
    while polish < MAX_ITERATIONS && point.grad.abs() > 1e-14 && point.hess > 0.0 {
        let target = (beta - point.grad / point.hess).clamp(a, b);
        if target == beta {
            break;
        }
        let Some((next_beta, next)) = backtrack(eval, beta, &point, target) else {
            break;
        };
        polish += 1;
        let moved = (next_beta - beta).abs();
        (beta, point) = (next_beta, next);
        if moved <= 1e-12 * beta.max(1.0) {
            break;
        }
    }
    iterations += polish;
    let outcome = if polish == MAX_ITERATIONS {
        FitOutcome::IterationLimit
    } else if beta == lo || beta == hi {
        FitOutcome::Bound
    } else {
        FitOutcome::Converged
    };
    finish(beta, point.nll, nll_before, iterations, bounds, outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::decision::calibrate::tests::Lcg;
    use crate::inference::decision::objective::Target;

    /// The combined level distribution of `levels` at `t`, in f64.
    fn combined(levels: &[[f64; 2]], t: f64) -> Vec<f64> {
        let q: Vec<f64> = levels
            .iter()
            .map(|l| 1.0 / (1.0 + (-(l[1] - l[0]) / t).exp()))
            .collect();
        let s: f64 = q.iter().sum();
        q.iter().map(|v| v / s).collect()
    }

    fn levels(g: &mut Lcg, k: usize) -> Vec<[f64; 2]> {
        (0..k)
            .map(|_| [g.uniform() * 8.0 - 4.0, g.uniform() * 8.0 - 4.0])
            .collect()
    }

    /// Level logits uniform in [-4, 4); gold levels drawn from the
    /// combined distribution at `t_star`.
    fn drawn(n: usize, k: usize, t_star: f64, seed: u64) -> Vec<IsolatedSample> {
        let mut g = Lcg(seed);
        (0..n)
            .map(|_| {
                let lv = levels(&mut g, k);
                let p = combined(&lv, t_star);
                let r = g.uniform();
                let mut c = 0.0;
                let mut gold = k - 1;
                for (i, pi) in p.iter().enumerate() {
                    c += pi;
                    if r < c {
                        gold = i;
                        break;
                    }
                }
                IsolatedSample {
                    levels: lv,
                    target: Target::Hard(gold),
                }
            })
            .collect()
    }

    #[test]
    fn recovers_a_known_temperature() {
        let samples = drawn(4000, 4, 1.8, 1);
        let fit = fit_temperature_isolated(&samples, TemperatureBounds::default()).unwrap();
        assert!((fit.t - 1.8).abs() < 0.1, "t = {}", fit.t);
        assert!(fit.nll_after < fit.nll_before);
        assert!(!fit.at_bound);
        assert_eq!(fit.outcome, FitOutcome::Converged);
        // The gradient vanishes at the fit.
        assert!(evaluate_isolated(&samples, 1.0 / fit.t).grad.abs() < 1e-9);
    }

    /// Soft targets equal to the combined distribution at `t_star` put
    /// the minimum exactly at `t_star`.
    #[test]
    fn soft_targets_recover_exactly() {
        let mut g = Lcg(42);
        let samples: Vec<IsolatedSample> = (0..50)
            .map(|i| {
                let lv = levels(&mut g, 2 + i % 4);
                let y = combined(&lv, 0.6);
                IsolatedSample {
                    levels: lv,
                    target: Target::Soft(y),
                }
            })
            .collect();
        let fit = fit_temperature_isolated(&samples, TemperatureBounds::default()).unwrap();
        assert!((fit.t - 0.6).abs() < 1e-7, "t = {}", fit.t);
        assert_eq!(fit.outcome, FitOutcome::Converged);
    }

    #[test]
    fn matches_a_log_grid_minimiser() {
        let samples = drawn(500, 3, 0.7, 7);
        let fit = fit_temperature_isolated(&samples, TemperatureBounds::default()).unwrap();
        let (lo, hi) = (0.05f64.ln(), 100f64.ln());
        let steps = 4000;
        let (mut best_t, mut best_nll) = (1.0, f64::INFINITY);
        for i in 0..=steps {
            let t = (lo + (hi - lo) * i as f64 / steps as f64).exp();
            let nll = evaluate_isolated(&samples, 1.0 / t).nll;
            if nll < best_nll {
                (best_t, best_nll) = (t, nll);
            }
        }
        let grid_ratio = ((hi - lo) / steps as f64).exp();
        assert!(fit.t / best_t < grid_ratio && best_t / fit.t < grid_ratio);
        assert!(fit.nll_after <= best_nll + 1e-12);
    }

    /// Only the gold level has a positive gap, so a colder temperature
    /// always helps and the fit stops at the lower bound.
    #[test]
    fn separable_data_hits_the_lower_bound() {
        let mut g = Lcg(11);
        let samples: Vec<IsolatedSample> = (0..40)
            .map(|i| {
                let gold = i % 3;
                let lv = (0..3)
                    .map(|j| {
                        let d = 0.5 + 2.5 * g.uniform();
                        if j == gold { [0.0, d] } else { [d, 0.0] }
                    })
                    .collect();
                IsolatedSample {
                    levels: lv,
                    target: Target::Hard(gold),
                }
            })
            .collect();
        let bounds = TemperatureBounds {
            min: 0.05,
            max: 20.0,
        };
        let fit = fit_temperature_isolated(&samples, bounds).unwrap();
        assert_eq!(fit.t, 0.05);
        assert!(fit.at_bound);
        assert_eq!(fit.outcome, FitOutcome::Bound);
        assert!(fit.nll_after < fit.nll_before);
    }

    #[test]
    fn equal_gaps_are_flat() {
        let samples = vec![IsolatedSample {
            levels: vec![[0.0, 1.0], [2.0, 3.0]],
            target: Target::Hard(1),
        }];
        let fit = fit_temperature_isolated(&samples, TemperatureBounds::default()).unwrap();
        assert_eq!(fit.t, 1.0);
        assert_eq!(fit.outcome, FitOutcome::Flat);
        assert_eq!(fit.nll_after, fit.nll_before);
        let fit =
            fit_temperature_isolated(&samples, TemperatureBounds { min: 2.0, max: 5.0 }).unwrap();
        assert_eq!(fit.t, 2.0);
        assert!(fit.at_bound);
    }

    #[test]
    fn invalid_input_errors() {
        let ok = vec![IsolatedSample {
            levels: vec![[0.0, 1.0], [0.0, -1.0]],
            target: Target::Hard(0),
        }];
        assert!(fit_temperature_isolated(&ok, TemperatureBounds::default()).is_ok());
        assert!(fit_temperature_isolated(&[], TemperatureBounds::default()).is_err());
        for (min, max) in [
            (0.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (0.1, f64::INFINITY),
        ] {
            assert!(fit_temperature_isolated(&ok, TemperatureBounds { min, max }).is_err());
        }
        let out_of_range = vec![IsolatedSample {
            levels: vec![[0.0, 1.0], [0.0, -1.0]],
            target: Target::Hard(2),
        }];
        assert!(fit_temperature_isolated(&out_of_range, TemperatureBounds::default()).is_err());
    }
}
