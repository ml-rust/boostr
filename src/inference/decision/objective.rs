//! Calibration samples and the temperature-scaling objective over them.
//!
//! The objective is written in `beta = 1 / T`:
//! `NLL(beta) = mean_s [logsumexp(beta * z_s) - beta * E_y[z_s]]`,
//! with gradient `mean_s (E_p[z_s] - E_y[z_s])` and second derivative
//! `mean_s Var_p[z_s] >= 0`, where `p = softmax(beta * z_s)`.

use crate::error::{Error, Result};

use super::answer::SUM_TOLERANCE;

/// The label of one calibration sample.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Target {
    /// The index of the correct option.
    Hard(usize),
    /// A target distribution over the options.
    Soft(Vec<f64>),
}

/// One row of option logits and its label. Rows can have different option counts.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSample {
    /// The option logits at T = 1.
    pub logits: Vec<f64>,
    /// The label.
    pub target: Target,
}

/// The objective value and its first two derivatives in `beta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectivePoint {
    /// `NLL(beta)`.
    pub nll: f64,
    /// `dNLL / dbeta`.
    pub grad: f64,
    /// `d2NLL / dbeta2`.
    pub hess: f64,
}

pub(crate) fn invalid(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "samples",
        reason,
    }
}

/// Checks every sample: at least 2 finite logits, a Hard index in range, a
/// Soft target that is a distribution of matching length.
pub fn validate_samples(samples: &[CalibrationSample]) -> Result<()> {
    if samples.is_empty() {
        return Err(invalid("no calibration samples".into()));
    }
    for (s, sample) in samples.iter().enumerate() {
        let k = sample.logits.len();
        if k < 2 {
            return Err(invalid(format!(
                "sample {s} has {k} options; at least 2 are needed"
            )));
        }
        if let Some(i) = sample.logits.iter().position(|v| !v.is_finite()) {
            return Err(invalid(format!("sample {s} logit {i} is not finite")));
        }
        check_target(s, &sample.target, k)?;
    }
    Ok(())
}

/// Checks sample `s`'s target over `k` options: a Hard index below `k`, or
/// a Soft distribution of `k` entries.
pub(crate) fn check_target(s: usize, target: &Target, k: usize) -> Result<()> {
    match target {
        Target::Hard(j) if *j >= k => Err(invalid(format!(
            "sample {s} hard target {j} is out of range for {k} options"
        ))),
        Target::Hard(_) => Ok(()),
        Target::Soft(y) => {
            if y.len() != k {
                return Err(invalid(format!(
                    "sample {s} soft target has {} entries for {k} options",
                    y.len()
                )));
            }
            if y.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(invalid(format!(
                    "sample {s} soft target has an entry that is negative or not finite"
                )));
            }
            let total: f64 = y.iter().sum();
            if (total - 1.0).abs() > SUM_TOLERANCE {
                return Err(invalid(format!(
                    "sample {s} soft target sums to {total}, not 1"
                )));
            }
            Ok(())
        }
    }
}

/// True when every sample's logits are all equal, so the objective does not depend on `beta`.
pub fn is_flat(samples: &[CalibrationSample]) -> bool {
    samples
        .iter()
        .all(|s| s.logits.iter().all(|&v| v == s.logits[0]))
}

fn target_mean(sample: &CalibrationSample) -> f64 {
    match &sample.target {
        Target::Hard(j) => sample.logits[*j],
        Target::Soft(y) => y.iter().zip(&sample.logits).map(|(a, b)| a * b).sum(),
    }
}

/// The objective and its derivatives at `beta`. Samples must have passed [`validate_samples`].
pub fn evaluate(samples: &[CalibrationSample], beta: f64) -> ObjectivePoint {
    let (mut nll, mut grad, mut hess) = (0.0, 0.0, 0.0);
    let mut w: Vec<f64> = Vec::new();
    for sample in samples {
        let z = &sample.logits;
        let m = z
            .iter()
            .map(|&v| beta * v)
            .fold(f64::NEG_INFINITY, f64::max);
        w.clear();
        w.extend(z.iter().map(|&v| (beta * v - m).exp()));
        let total: f64 = w.iter().sum();
        let lse = m + total.ln();
        let mean_p: f64 = w.iter().zip(z).map(|(wi, zi)| wi * zi).sum::<f64>() / total;
        let var_p: f64 = w
            .iter()
            .zip(z)
            .map(|(wi, zi)| wi * (zi - mean_p) * (zi - mean_p))
            .sum::<f64>()
            / total;
        let mean_y = target_mean(sample);
        nll += lse - beta * mean_y;
        grad += mean_p - mean_y;
        hess += var_p;
    }
    let n = samples.len() as f64;
    ObjectivePoint {
        nll: nll / n,
        grad: grad / n,
        hess: hess / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hard(logits: Vec<f64>, j: usize) -> CalibrationSample {
        CalibrationSample {
            logits,
            target: Target::Hard(j),
        }
    }

    #[test]
    fn hand_values_two_options() {
        // z = [0, 1], target 1, beta = 1: p1 = e / (1 + e).
        let s = [hard(vec![0.0, 1.0], 1)];
        let pt = evaluate(&s, 1.0);
        let e = 1f64.exp();
        let p1 = e / (1.0 + e);
        assert!((pt.nll - ((1.0 + e).ln() - 1.0)).abs() < 1e-12);
        assert!((pt.grad - (p1 - 1.0)).abs() < 1e-12);
        assert!((pt.hess - p1 * (1.0 - p1)).abs() < 1e-12);
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let s = [
            hard(vec![0.3, -1.0, 2.0], 0),
            CalibrationSample {
                logits: vec![1.0, 0.5],
                target: Target::Soft(vec![0.3, 0.7]),
            },
        ];
        let b = 0.8;
        let h = 1e-5;
        let (lo, mid, hi) = (evaluate(&s, b - h), evaluate(&s, b), evaluate(&s, b + h));
        assert!(((hi.nll - lo.nll) / (2.0 * h) - mid.grad).abs() < 1e-7);
        assert!(((hi.grad - lo.grad) / (2.0 * h) - mid.hess).abs() < 1e-7);
    }

    #[test]
    fn large_logits_stay_finite() {
        let s = [hard(vec![1.0e4, -1.0e4, 0.0], 1)];
        let pt = evaluate(&s, 5.0);
        assert!(pt.nll.is_finite() && pt.grad.is_finite() && pt.hess.is_finite());
        assert!((pt.nll - 1.0e5).abs() < 1e-6);
    }

    #[test]
    fn validation() {
        assert!(validate_samples(&[]).is_err());
        assert!(validate_samples(&[hard(vec![1.0], 0)]).is_err());
        assert!(validate_samples(&[hard(vec![1.0, f64::NAN], 0)]).is_err());
        assert!(validate_samples(&[hard(vec![1.0, 2.0], 2)]).is_err());
        let soft = |y: Vec<f64>| CalibrationSample {
            logits: vec![0.0, 1.0],
            target: Target::Soft(y),
        };
        assert!(validate_samples(&[soft(vec![0.5, 0.6])]).is_err());
        assert!(validate_samples(&[soft(vec![1.5, -0.5])]).is_err());
        assert!(validate_samples(&[soft(vec![1.0])]).is_err());
        assert!(validate_samples(&[soft(vec![0.4, 0.6]), hard(vec![0.0, 1.0, 2.0], 2)]).is_ok());
    }

    #[test]
    fn flat_detection() {
        assert!(is_flat(&[hard(vec![2.0, 2.0], 0), hard(vec![-1.0; 3], 2)]));
        assert!(!is_flat(&[
            hard(vec![2.0, 2.0], 0),
            hard(vec![0.0, 1.0], 1)
        ]));
    }
}
