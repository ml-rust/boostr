//! Answer temperatures: the policy that picks T per question, and the tempered
//! softmax over one row of option logits.

use super::types::{QuestionKind, QuestionMeta};
use crate::error::{Error, Result};

/// How the temperature of a question is chosen.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TemperaturePolicy {
    /// One temperature for every question.
    Fixed(f64),
    /// One temperature per question kind.
    ByKind {
        /// Temperature of Choice questions.
        choice: f64,
        /// Temperature of Noul questions.
        noul: f64,
        /// Temperature of Score questions, including every level row of an
        /// isolated Score question.
        score: f64,
    },
    /// `T(n) = max(min, a + b * ln(max(n, 2)))` for a question with `n` options.
    /// `n` below 2 is read as 2.
    ByOptions {
        /// Intercept.
        a: f64,
        /// Slope on `ln n`.
        b: f64,
        /// Floor. Must be finite and > 0.
        min: f64,
    },
}

fn checked(t: f64, what: &str) -> Result<f64> {
    if t.is_finite() && t > 0.0 {
        Ok(t)
    } else {
        Err(Error::InvalidArgument {
            arg: "temperature",
            reason: format!("{what} must be finite and > 0, got {t}"),
        })
    }
}

fn finite(v: f64, what: &str) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::InvalidArgument {
            arg: "temperature",
            reason: format!("{what} must be finite, got {v}"),
        })
    }
}

impl TemperaturePolicy {
    /// The temperature of a question. Errors when the result is not finite and > 0.
    pub fn resolve(&self, meta: &QuestionMeta) -> Result<f64> {
        match self {
            Self::Fixed(t) => checked(*t, "Fixed temperature"),
            Self::ByKind {
                choice,
                noul,
                score,
            } => match meta.kind {
                QuestionKind::Choice => checked(*choice, "ByKind choice temperature"),
                QuestionKind::Noul { .. } => checked(*noul, "ByKind noul temperature"),
                QuestionKind::Score { .. } => checked(*score, "ByKind score temperature"),
            },
            Self::ByOptions { a, b, min } => {
                let a = finite(*a, "ByOptions a")?;
                let b = finite(*b, "ByOptions b")?;
                let min = checked(*min, "ByOptions min")?;
                let n = meta.options.max(2) as f64;
                checked((a + b * n.ln()).max(min), "ByOptions temperature")
            }
        }
    }
}

/// `softmax(logits / t)` computed in f64 with the maximum subtracted.
/// Errors on empty input, a non-finite logit, or `t` not finite and > 0.
pub fn tempered_softmax(logits: &[f32], t: f64) -> Result<Vec<f64>> {
    let t = checked(t, "softmax temperature")?;
    if logits.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: "empty logit row".into(),
        });
    }
    if let Some(i) = logits.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("logit {i} is not finite: {}", logits[i]),
        });
    }
    let mut out: Vec<f64> = logits.iter().map(|&v| f64::from(v) / t).collect();
    let m = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for v in &mut out {
        *v = (*v - m).exp();
    }
    let s: f64 = out.iter().sum();
    for v in &mut out {
        *v /= s;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(kind: QuestionKind, options: usize) -> QuestionMeta {
        QuestionMeta { kind, options }
    }

    fn score() -> QuestionKind {
        QuestionKind::Score {
            values: None,
            isolated: false,
        }
    }

    #[test]
    fn fixed_and_invalid_fixed() {
        let m = meta(QuestionKind::Choice, 3);
        assert_eq!(TemperaturePolicy::Fixed(1.3).resolve(&m).unwrap(), 1.3);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(TemperaturePolicy::Fixed(bad).resolve(&m).is_err());
        }
    }

    #[test]
    fn by_kind_picks_the_kind() {
        let p = TemperaturePolicy::ByKind {
            choice: 1.48,
            noul: 2.22,
            score: 1.38,
        };
        assert_eq!(p.resolve(&meta(QuestionKind::Choice, 4)).unwrap(), 1.48);
        assert_eq!(
            p.resolve(&meta(QuestionKind::Noul { yes: 1 }, 2)).unwrap(),
            2.22
        );
        assert_eq!(p.resolve(&meta(score(), 5)).unwrap(), 1.38);
        let bad = TemperaturePolicy::ByKind {
            choice: 1.0,
            noul: 0.0,
            score: 1.0,
        };
        assert!(bad.resolve(&meta(QuestionKind::Choice, 4)).is_ok());
        assert!(
            bad.resolve(&meta(QuestionKind::Noul { yes: 1 }, 2))
                .is_err()
        );
    }

    #[test]
    fn by_options_matches_the_decider_rule() {
        let p = TemperaturePolicy::ByOptions {
            a: 10.124,
            b: -1.633,
            min: 0.05,
        };
        let t2 = p.resolve(&meta(QuestionKind::Choice, 2)).unwrap();
        assert!((t2 - (10.124 - 1.633 * 2f64.ln())).abs() < 1e-12);
        let t10 = p.resolve(&meta(QuestionKind::Choice, 10)).unwrap();
        assert!((t10 - (10.124 - 1.633 * 10f64.ln())).abs() < 1e-12);
        let t151 = p.resolve(&meta(QuestionKind::Choice, 151)).unwrap();
        assert!((t151 - (10.124 - 1.633 * 151f64.ln())).abs() < 1e-12);
        let floor = TemperaturePolicy::ByOptions {
            a: 1.0,
            b: -1.0,
            min: 0.2,
        };
        let t1 = floor.resolve(&meta(QuestionKind::Choice, 1)).unwrap();
        assert_eq!(t1, floor.resolve(&meta(QuestionKind::Choice, 2)).unwrap());
        assert_eq!(floor.resolve(&meta(QuestionKind::Choice, 50)).unwrap(), 0.2);
        for (a, b, min) in [
            (1.0, f64::NAN, 0.05),
            (1.0, 1.0, 0.0),
            (f64::INFINITY, 0.0, 0.05),
        ] {
            let bad = TemperaturePolicy::ByOptions { a, b, min };
            assert!(bad.resolve(&meta(QuestionKind::Choice, 3)).is_err());
        }
    }

    #[test]
    fn softmax_hand_values() {
        let p = tempered_softmax(&[0.0, 2f32.ln()], 1.0).unwrap();
        assert!((p[0] - 1.0 / 3.0).abs() < 1e-7);
        assert!((p[1] - 2.0 / 3.0).abs() < 1e-7);
        // T = 2 halves the logits: weights 1 and sqrt(2).
        let q = tempered_softmax(&[0.0, 2f32.ln()], 2.0).unwrap();
        let s2 = 2f64.sqrt();
        assert!((q[1] - s2 / (1.0 + s2)).abs() < 1e-7);
    }

    #[test]
    fn softmax_is_stable_on_large_logits() {
        let p = tempered_softmax(&[1.0e4, 1.0e4 - 1.0, -1.0e4], 1.0).unwrap();
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let e = (-1f64).exp();
        assert!((p[0] - 1.0 / (1.0 + e)).abs() < 1e-12);
        assert_eq!(p[2], 0.0);
    }

    #[test]
    fn softmax_sums_to_one_flattens_and_keeps_argmax() {
        let logits = [0.3f32, -1.2, 2.5, 0.9, 2.4];
        let mut prev_max = 1.0;
        for t in [0.1, 0.5, 1.0, 3.0, 50.0, 1.0e6] {
            let p = tempered_softmax(&logits, t).unwrap();
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let argmax = (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
            assert_eq!(argmax, 2);
            assert!(p[2] <= prev_max + 1e-15);
            prev_max = p[2];
        }
        let flat = tempered_softmax(&logits, 1.0e6).unwrap();
        assert!(flat.iter().all(|v| (v - 0.2).abs() < 1e-5));
    }

    #[test]
    fn softmax_rejects_bad_input() {
        assert!(tempered_softmax(&[], 1.0).is_err());
        assert!(tempered_softmax(&[0.0, f32::NAN], 1.0).is_err());
        assert!(tempered_softmax(&[0.0, f32::INFINITY], 1.0).is_err());
        assert!(tempered_softmax(&[0.0, 1.0], 0.0).is_err());
        assert!(tempered_softmax(&[0.0, 1.0], f64::NAN).is_err());
    }
}
