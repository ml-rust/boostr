//! Fitting one temperature to Score answers read both ways: listed
//! samples (one row over every level) and isolated samples (one yes/no row
//! per level).
//!
//! The objective is the mean NLL over all samples together:
//! `NLL = (n_l * NLL_listed + n_i * NLL_isolated) / (n_l + n_i)`, with
//! each part as [`super::super::objective`] and [`super::objective`] define
//! it. The isolated part is not convex in `beta = 1 / T`, so a fit with
//! isolated samples brackets before it refines, as
//! [`super::fit_temperature_isolated`] does. A fit with listed samples only
//! is [`fit_temperature`].

use super::fit::{fit_bracketed, fit_temperature_isolated};
use super::objective::{IsolatedSample, evaluate_isolated, is_flat_isolated, validate_isolated};
use crate::error::{Error, Result};
use crate::inference::decision::calibrate::{
    TemperatureBounds, TemperatureFit, check_bounds, fit_temperature,
};
use crate::inference::decision::objective::{
    CalibrationSample, ObjectivePoint, evaluate, is_flat, validate_samples,
};

/// The mean objective over `listed` and `isolated` together at `beta`.
/// Both must have passed validation. An empty side adds nothing; both
/// empty give a zero point.
pub fn evaluate_mixed(
    listed: &[CalibrationSample],
    isolated: &[IsolatedSample],
    beta: f64,
) -> ObjectivePoint {
    let (nl, ni) = (listed.len() as f64, isolated.len() as f64);
    let n = nl + ni;
    let zero = ObjectivePoint {
        nll: 0.0,
        grad: 0.0,
        hess: 0.0,
    };
    if n == 0.0 {
        return zero;
    }
    let l = if listed.is_empty() {
        zero
    } else {
        evaluate(listed, beta)
    };
    let i = if isolated.is_empty() {
        zero
    } else {
        evaluate_isolated(isolated, beta)
    };
    ObjectivePoint {
        nll: (nl * l.nll + ni * i.nll) / n,
        grad: (nl * l.grad + ni * i.grad) / n,
        hess: (nl * l.hess + ni * i.hess) / n,
    }
}

/// Fits one temperature to listed and isolated Score samples together,
/// within `bounds`, by minimising their mean NLL. With isolated samples
/// only it is [`fit_temperature_isolated`]; with listed samples only it is
/// [`fit_temperature`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when both sides are empty, a sample fails
/// its side's validation, or the bounds are not finite `0 < min < max`.
pub fn fit_temperature_mixed(
    listed: &[CalibrationSample],
    isolated: &[IsolatedSample],
    bounds: TemperatureBounds,
) -> Result<TemperatureFit> {
    match (listed.is_empty(), isolated.is_empty()) {
        (true, true) => Err(Error::InvalidArgument {
            arg: "samples",
            reason: "no calibration samples".into(),
        }),
        (false, true) => fit_temperature(listed, bounds),
        (true, false) => fit_temperature_isolated(isolated, bounds),
        (false, false) => {
            validate_samples(listed)?;
            validate_isolated(isolated)?;
            check_bounds(bounds)?;
            Ok(fit_bracketed(
                |beta| evaluate_mixed(listed, isolated, beta),
                is_flat(listed) && is_flat_isolated(isolated),
                bounds,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::decision::calibrate::FitOutcome;
    use crate::inference::decision::calibrate::tests::Lcg;
    use crate::inference::decision::objective::Target;

    fn listed(g: &mut Lcg, n: usize) -> Vec<CalibrationSample> {
        (0..n)
            .map(|i| CalibrationSample {
                logits: (0..3).map(|_| g.uniform() * 6.0 - 3.0).collect(),
                target: Target::Hard(i % 3),
            })
            .collect()
    }

    fn isolated(g: &mut Lcg, n: usize) -> Vec<IsolatedSample> {
        (0..n)
            .map(|i| IsolatedSample {
                levels: (0..4)
                    .map(|_| [g.uniform() * 6.0 - 3.0, g.uniform() * 6.0 - 3.0])
                    .collect(),
                target: Target::Hard(i % 4),
            })
            .collect()
    }

    #[test]
    fn objective_is_the_sample_weighted_mean() {
        let mut g = Lcg(9);
        let (l, i) = (listed(&mut g, 7), isolated(&mut g, 3));
        for beta in [0.3, 1.0, 2.2] {
            let m = evaluate_mixed(&l, &i, beta);
            let (a, b) = (evaluate(&l, beta), evaluate_isolated(&i, beta));
            assert!((m.nll - (7.0 * a.nll + 3.0 * b.nll) / 10.0).abs() < 1e-12);
            assert!((m.grad - (7.0 * a.grad + 3.0 * b.grad) / 10.0).abs() < 1e-12);
            assert!((m.hess - (7.0 * a.hess + 3.0 * b.hess) / 10.0).abs() < 1e-12);
        }
        assert_eq!(evaluate_mixed(&l, &[], 1.0), evaluate(&l, 1.0));
        assert_eq!(evaluate_mixed(&[], &i, 1.0), evaluate_isolated(&i, 1.0));
    }

    #[test]
    fn matches_a_log_grid_minimiser() {
        let mut g = Lcg(21);
        let (l, i) = (listed(&mut g, 120), isolated(&mut g, 80));
        let fit = fit_temperature_mixed(&l, &i, TemperatureBounds::default()).unwrap();
        let (lo, hi) = (0.05f64.ln(), 100f64.ln());
        let steps = 4000;
        let (mut best_t, mut best_nll) = (1.0, f64::INFINITY);
        for s in 0..=steps {
            let t = (lo + (hi - lo) * s as f64 / steps as f64).exp();
            let nll = evaluate_mixed(&l, &i, 1.0 / t).nll;
            if nll < best_nll {
                (best_t, best_nll) = (t, nll);
            }
        }
        let grid_ratio = ((hi - lo) / steps as f64).exp();
        assert!(fit.t / best_t < grid_ratio && best_t / fit.t < grid_ratio);
        assert!(fit.nll_after <= best_nll + 1e-12);
        assert!((fit.nll_before - evaluate_mixed(&l, &i, 1.0).nll).abs() < 1e-15);
    }

    #[test]
    fn one_sided_input_is_the_one_sided_fit() {
        let mut g = Lcg(4);
        let (l, i) = (listed(&mut g, 40), isolated(&mut g, 40));
        let b = TemperatureBounds::default();
        assert_eq!(
            fit_temperature_mixed(&l, &[], b).unwrap(),
            fit_temperature(&l, b).unwrap()
        );
        assert_eq!(
            fit_temperature_mixed(&[], &i, b).unwrap(),
            fit_temperature_isolated(&i, b).unwrap()
        );
    }

    #[test]
    fn flat_and_invalid_input() {
        let l = vec![CalibrationSample {
            logits: vec![1.0, 1.0],
            target: Target::Hard(0),
        }];
        let i = vec![IsolatedSample {
            levels: vec![[0.0, 1.0], [2.0, 3.0]],
            target: Target::Hard(1),
        }];
        let fit = fit_temperature_mixed(&l, &i, TemperatureBounds::default()).unwrap();
        assert_eq!((fit.t, fit.outcome), (1.0, FitOutcome::Flat));
        let b = TemperatureBounds::default();
        assert!(fit_temperature_mixed(&[], &[], b).is_err());
        let bad = vec![CalibrationSample {
            logits: vec![1.0, 2.0],
            target: Target::Hard(4),
        }];
        assert!(fit_temperature_mixed(&bad, &i, b).is_err());
        let bounds = TemperatureBounds { min: 2.0, max: 1.0 };
        assert!(fit_temperature_mixed(&l, &i, bounds).is_err());
    }
}
