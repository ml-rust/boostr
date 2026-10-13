//! Isolated Score calibration samples and the NLL of their combined level
//! distribution as a function of `beta = 1 / T`.
//!
//! Level `i` of a sample holds `[no, yes]` logits with gap `d_i = yes -
//! no`. At `beta` its yes probability is the tempered softmax's yes entry,
//! `q_i = sigmoid(beta * d_i)`, and the level distribution is `p_i = q_i /
//! S` with `S = sum_j q_j`, as `combine_isolated` forms it. With `w = p`
//! and `r = 1 - q`:
//!
//! - `NLL = mean_s [-E_y[ln q] + ln S]`
//! - `grad = mean_s [-E_y[d r] + E_w[d r]]`
//! - `hess = mean_s [E_y[d^2 q r] + E_w[d^2 r (1 - 2q)] - E_w[d r]^2]`
//!
//! The first term is convex in `beta`. `ln S`, a log-sum of sigmoids, is
//! not: for a gold level with `d = 0` and one other level with `d > 0`,
//! `hess < 0` at every `beta > 0`. The NLL can have more than one local
//! minimum, so the fit brackets before it refines.

use super::super::objective::{ObjectivePoint, Target, check_target, invalid};
use super::super::run::finish::LEVEL_YES;
use crate::error::Result;

/// The level rows of one isolated Score question and its label.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedSample {
    /// The logits of each level row at T = 1, `[no, yes]` as the run's
    /// level rows hold them.
    pub levels: Vec<[f64; 2]>,
    /// The gold level, or a target distribution over the levels.
    pub target: Target,
}

/// `ln(1 + e^x)` without overflow.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The yes-minus-no gap of a level row.
fn gap(level: &[f64; 2]) -> f64 {
    level[LEVEL_YES] - level[1 - LEVEL_YES]
}

/// Checks every sample: at least 2 levels, finite logits, a Hard level in
/// range, a Soft target that is a distribution over the levels.
///
/// # Errors
///
/// [`Error::InvalidArgument`] naming the first sample and value that fail.
pub fn validate_isolated(samples: &[IsolatedSample]) -> Result<()> {
    if samples.is_empty() {
        return Err(invalid("no calibration samples".into()));
    }
    for (s, sample) in samples.iter().enumerate() {
        let k = sample.levels.len();
        if k < 2 {
            return Err(invalid(format!(
                "sample {s} has {k} levels; at least 2 are needed"
            )));
        }
        if let Some(i) = sample
            .levels
            .iter()
            .position(|l| !l.iter().all(|v| v.is_finite()))
        {
            return Err(invalid(format!(
                "sample {s} level {i} has a logit that is not finite"
            )));
        }
        check_target(s, &sample.target, k)?;
    }
    Ok(())
}

/// True when every sample's levels share one gap, so every temperature
/// gives the same level distribution.
pub fn is_flat_isolated(samples: &[IsolatedSample]) -> bool {
    samples.iter().all(|s| {
        let first = gap(&s.levels[0]);
        s.levels.iter().all(|l| gap(l) == first)
    })
}

/// The objective and its derivatives at `beta`. Samples must have passed
/// [`validate_isolated`].
pub fn evaluate_isolated(samples: &[IsolatedSample], beta: f64) -> ObjectivePoint {
    let (mut nll, mut grad, mut hess) = (0.0, 0.0, 0.0);
    let mut d: Vec<f64> = Vec::new();
    let mut ln_q: Vec<f64> = Vec::new();
    for sample in samples {
        d.clear();
        d.extend(sample.levels.iter().map(gap));
        ln_q.clear();
        ln_q.extend(d.iter().map(|&di| -softplus(-beta * di)));
        let m = ln_q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + ln_q.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        // `q`, `r = 1 - q` and the level weight `w` of level `j`.
        let at = |j: usize| {
            let q = ln_q[j].exp();
            let r = (-softplus(beta * d[j])).exp();
            (q, r, (ln_q[j] - lse).exp())
        };
        let (mut s1, mut s2) = (0.0, 0.0);
        for (j, &dj) in d.iter().enumerate() {
            let (q, r, w) = at(j);
            s1 += w * dj * r;
            s2 += w * dj * dj * r * (r - q);
        }
        // The target's `-ln q`, its derivative and its second derivative.
        let term = |g: usize| {
            let (q, r, _) = at(g);
            (-ln_q[g], -d[g] * r, d[g] * d[g] * q * r)
        };
        let (t_nll, t_grad, t_hess) = match &sample.target {
            Target::Hard(g) => term(*g),
            Target::Soft(y) => y.iter().enumerate().fold((0.0, 0.0, 0.0), |acc, (g, &yg)| {
                if yg == 0.0 {
                    return acc;
                }
                let (a, b, c) = term(g);
                (acc.0 + yg * a, acc.1 + yg * b, acc.2 + yg * c)
            }),
        };
        nll += t_nll + lse;
        grad += t_grad + s1;
        hess += t_hess + s2 - s1 * s1;
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
    use crate::inference::decision::answer::combine_isolated;
    use crate::inference::decision::calibrate::tests::Lcg;
    use crate::inference::decision::temperature::tempered_softmax;

    fn hard(levels: Vec<[f64; 2]>, g: usize) -> IsolatedSample {
        IsolatedSample {
            levels,
            target: Target::Hard(g),
        }
    }

    /// Gaps 1 and 0, gold level 0, beta 1: `p_0 = sigmoid(1) / (sigmoid(1) + 1/2)`.
    #[test]
    fn hand_value() {
        let s = [hard(vec![[0.0, 1.0], [0.0, 0.0]], 0)];
        let sig = 1.0 / (1.0 + (-1f64).exp());
        let pt = evaluate_isolated(&s, 1.0);
        assert!((pt.nll + (sig / (sig + 0.5)).ln()).abs() < 1e-12);
        // grad = -(1 - sig) + w_0 (1 - sig), w_0 = p_0.
        let p0 = sig / (sig + 0.5);
        assert!((pt.grad - (p0 - 1.0) * (1.0 - sig)).abs() < 1e-12);
    }

    /// The NLL equals `-ln p_gold` of the run's own answer math: tempered
    /// level rows, their yes entries, `combine_isolated`.
    #[test]
    fn matches_the_answer_stage() {
        let mut g = Lcg(5);
        for t in [0.3, 1.0, 2.5] {
            let levels: Vec<[f64; 2]> = (0..4)
                .map(|_| [g.uniform() * 6.0 - 3.0, g.uniform() * 6.0 - 3.0])
                .collect();
            let yes: Vec<f64> = levels
                .iter()
                .map(|l| tempered_softmax(&[l[0] as f32, l[1] as f32], t).unwrap()[LEVEL_YES])
                .collect();
            let (p, _) = combine_isolated(&yes).unwrap();
            let levels: Vec<[f64; 2]> = levels
                .iter()
                .map(|l| [f64::from(l[0] as f32), f64::from(l[1] as f32)])
                .collect();
            let pt = evaluate_isolated(&[hard(levels, 2)], 1.0 / t);
            assert!((pt.nll + p[2].ln()).abs() < 1e-9, "t {t}");
        }
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let s = [
            hard(vec![[0.3, -1.0], [2.0, 0.5], [-0.4, 1.2]], 1),
            IsolatedSample {
                levels: vec![[1.0, 0.5], [0.0, 2.0]],
                target: Target::Soft(vec![0.3, 0.7]),
            },
        ];
        let h = 1e-5;
        for b in [0.2, 0.8, 3.0] {
            let (lo, mid, hi) = (
                evaluate_isolated(&s, b - h),
                evaluate_isolated(&s, b),
                evaluate_isolated(&s, b + h),
            );
            assert!(
                ((hi.nll - lo.nll) / (2.0 * h) - mid.grad).abs() < 1e-7,
                "beta {b}"
            );
            assert!(
                ((hi.grad - lo.grad) / (2.0 * h) - mid.hess).abs() < 1e-7,
                "beta {b}"
            );
        }
    }

    /// A gold level with gap 0 beside a level with gap 2: the NLL is
    /// concave in `beta`.
    #[test]
    fn objective_is_not_convex() {
        let s = [hard(vec![[0.0, 0.0], [0.0, 2.0]], 0)];
        for b in [0.1, 1.0, 5.0] {
            assert!(evaluate_isolated(&s, b).hess < 0.0, "beta {b}");
        }
    }

    #[test]
    fn large_gaps_stay_finite() {
        let s = [hard(vec![[0.0, 1.0e4], [1.0e4, 0.0]], 1)];
        let pt = evaluate_isolated(&s, 5.0);
        assert!(pt.nll.is_finite() && pt.grad.is_finite() && pt.hess.is_finite());
        // ln q_1 = -5e4 and ln S = 0 up to e^-5e4.
        assert!((pt.nll - 5.0e4).abs() < 1e-6);
    }

    #[test]
    fn validation_and_flatness() {
        assert!(validate_isolated(&[]).is_err());
        let e = validate_isolated(&[hard(vec![[0.0, 1.0]], 0)]).unwrap_err();
        assert!(e.to_string().contains("sample 0 has 1 levels"), "{e}");
        let e = validate_isolated(&[hard(vec![[0.0, 1.0], [f64::NAN, 0.0]], 0)]).unwrap_err();
        assert!(e.to_string().contains("sample 0 level 1"), "{e}");
        assert!(validate_isolated(&[hard(vec![[0.0, 1.0]; 2], 2)]).is_err());
        let soft = IsolatedSample {
            levels: vec![[0.0, 1.0]; 2],
            target: Target::Soft(vec![0.5, 0.6]),
        };
        assert!(validate_isolated(&[soft]).is_err());
        assert!(is_flat_isolated(&[
            hard(vec![[0.0, 1.0], [2.0, 3.0]], 0),
            hard(vec![[1.0, 1.0]; 3], 2)
        ]));
        assert!(!is_flat_isolated(&[hard(vec![[0.0, 1.0], [0.0, 2.0]], 0)]));
    }
}
