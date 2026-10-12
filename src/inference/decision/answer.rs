//! Turning one question's option distribution into its answer.

use super::confidence::{ScoreConfidence, choice_confidence, first_argmax, score_confidence};
use super::noul::{NoulBand, NoulMap, decide};
use super::types::{Answer, QuestionKind, QuestionMeta};
use crate::error::{Error, Result};

/// The largest allowed distance of a distribution's sum from 1.
pub const SUM_TOLERANCE: f64 = 1e-6;

/// How answers are formed from distributions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AnswerOptions {
    /// Noul abstention band. `None` decides at 0.5 (`p_yes >= 0.5` is "yes").
    pub noul_band: Option<NoulBand>,
    /// Map applied to the Noul yes probability before the band.
    pub noul_map: NoulMap,
    /// Score confidence rule.
    pub score_confidence: ScoreConfidence,
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

fn check_entries(probs: &[f64], arg: &'static str) -> Result<()> {
    if let Some(i) = probs.iter().position(|p| !p.is_finite() || *p < 0.0) {
        return Err(invalid(
            arg,
            format!("entry {i} must be finite and >= 0, got {}", probs[i]),
        ));
    }
    Ok(())
}

fn check_distribution(probs: &[f64]) -> Result<()> {
    check_entries(probs, "probs")?;
    let s: f64 = probs.iter().sum();
    if (s - 1.0).abs() > SUM_TOLERANCE {
        return Err(invalid(
            "probs",
            format!("must sum to 1 within {SUM_TOLERANCE}, sum is {s}"),
        ));
    }
    Ok(())
}

/// Combines per-level yes probabilities read in isolation into one level
/// distribution `p_i = y_i / sum(y)`, and returns it with the fit mass `sum(y)`.
/// Each `y_i` must be in `[0, 1]`. A zero sum is an error.
pub fn combine_isolated(yes_probs: &[f64]) -> Result<(Vec<f64>, f64)> {
    if yes_probs.is_empty() {
        return Err(invalid("yes_probs", "no levels".into()));
    }
    check_entries(yes_probs, "yes_probs")?;
    if let Some(i) = yes_probs.iter().position(|&p| p > 1.0) {
        return Err(invalid(
            "yes_probs",
            format!("entry {i} is a probability above 1: {}", yes_probs[i]),
        ));
    }
    let mass: f64 = yes_probs.iter().sum();
    if mass <= 0.0 {
        return Err(invalid(
            "yes_probs",
            "every level has yes probability 0, so no level distribution exists".into(),
        ));
    }
    Ok((yes_probs.iter().map(|&y| y / mass).collect(), mass))
}

fn level_values(values: Option<&[f64]>, levels: usize) -> Result<(Vec<f64>, f64, f64)> {
    let v: Vec<f64> = match values {
        Some(v) => {
            if v.len() != levels {
                return Err(invalid(
                    "values",
                    format!("{} values for {levels} levels", v.len()),
                ));
            }
            if let Some(i) = v.iter().position(|x| !x.is_finite()) {
                return Err(invalid(
                    "values",
                    format!("value {i} is not finite: {}", v[i]),
                ));
            }
            v.to_vec()
        }
        None => (0..levels).map(|i| i as f64).collect(),
    };
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo {
        return Err(invalid(
            "values",
            format!("value range is empty: min {lo}, max {hi}"),
        ));
    }
    Ok((v, lo, hi))
}

/// The answer to one question from its option distribution.
///
/// `probs` is the option distribution (sum 1 within [`SUM_TOLERANCE`]), except
/// for an isolated Score question, where it holds the per-level yes
/// probabilities and [`combine_isolated`] forms the distribution.
pub fn answer_from_probs(
    meta: &QuestionMeta,
    probs: Vec<f64>,
    opts: &AnswerOptions,
) -> Result<Answer> {
    if meta.options == 0 {
        return Err(invalid(
            "options",
            "a question needs at least 1 option".into(),
        ));
    }
    if probs.len() != meta.options {
        return Err(invalid(
            "probs",
            format!("{} probabilities for {} options", probs.len(), meta.options),
        ));
    }
    match &meta.kind {
        QuestionKind::Choice => {
            check_distribution(&probs)?;
            let index = first_argmax(&probs);
            let confidence = choice_confidence(&probs);
            Ok(Answer::Choice {
                index,
                confidence,
                probs,
            })
        }
        QuestionKind::Noul { yes } => {
            if meta.options != 2 {
                return Err(invalid(
                    "options",
                    format!("a Noul question has 2 options, got {}", meta.options),
                ));
            }
            if *yes > 1 {
                return Err(invalid(
                    "yes",
                    format!("yes index must be 0 or 1, got {yes}"),
                ));
            }
            check_distribution(&probs)?;
            let p_yes = opts.noul_map.apply(probs[*yes].clamp(0.0, 1.0));
            let decided = decide(p_yes, opts.noul_band)?;
            Ok(Answer::Noul { p_yes, decided })
        }
        QuestionKind::Score { values, isolated } => {
            if meta.options < 2 {
                return Err(invalid(
                    "options",
                    format!(
                        "a Score question has at least 2 levels, got {}",
                        meta.options
                    ),
                ));
            }
            let (v, lo, hi) = level_values(values.as_deref(), meta.options)?;
            let (probs, fit_mass) = if *isolated {
                let (p, mass) = combine_isolated(&probs)?;
                (p, Some(mass))
            } else {
                check_distribution(&probs)?;
                (probs, None)
            };
            let expected: f64 = probs.iter().zip(&v).map(|(p, x)| p * x).sum();
            let normalized = ((expected - lo) / (hi - lo)).clamp(0.0, 1.0);
            let confidence = score_confidence(&probs, opts.score_confidence);
            Ok(Answer::Score {
                expected,
                normalized,
                confidence,
                probs,
                fit_mass,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(kind: QuestionKind, options: usize) -> QuestionMeta {
        QuestionMeta { kind, options }
    }

    fn score(values: Option<Vec<f64>>, isolated: bool) -> QuestionKind {
        QuestionKind::Score { values, isolated }
    }

    fn noul(p_yes: f64, opts: &AnswerOptions) -> (f64, Option<bool>) {
        let m = meta(QuestionKind::Noul { yes: 1 }, 2);
        match answer_from_probs(&m, vec![1.0 - p_yes, p_yes], opts).unwrap() {
            Answer::Noul { p_yes, decided } => (p_yes, decided),
            other => panic!("expected Noul, got {other:?}"),
        }
    }

    #[test]
    fn choice_confidence_and_ties() {
        let d = AnswerOptions::default();
        let a = answer_from_probs(&meta(QuestionKind::Choice, 4), vec![0.25; 4], &d).unwrap();
        assert_eq!(
            a,
            Answer::Choice {
                index: 0,
                confidence: 0.0,
                probs: vec![0.25; 4]
            }
        );
        let one_hot = vec![0.0, 0.0, 1.0];
        let b = answer_from_probs(&meta(QuestionKind::Choice, 3), one_hot, &d).unwrap();
        assert!(matches!(b, Answer::Choice { index: 2, confidence, .. } if confidence == 1.0));
        let c = answer_from_probs(&meta(QuestionKind::Choice, 1), vec![1.0], &d).unwrap();
        assert!(matches!(c, Answer::Choice { index: 0, confidence, .. } if confidence == 1.0));
        let tie = vec![0.1, 0.45, 0.45];
        let t = answer_from_probs(&meta(QuestionKind::Choice, 3), tie, &d).unwrap();
        assert!(matches!(t, Answer::Choice { index: 1, .. }));
    }

    #[test]
    fn noul_without_band() {
        let d = AnswerOptions::default();
        assert_eq!(noul(0.7, &d), (0.7, Some(true)));
        assert_eq!(noul(0.5, &d), (0.5, Some(true)));
        assert_eq!(noul(0.3, &d), (0.3, Some(false)));
        // The yes index selects the column.
        let m = meta(QuestionKind::Noul { yes: 0 }, 2);
        let a = answer_from_probs(&m, vec![0.9, 0.1], &d).unwrap();
        assert_eq!(
            a,
            Answer::Noul {
                p_yes: 0.9,
                decided: Some(true)
            }
        );
    }

    #[test]
    fn noul_band_edges_are_inclusive() {
        let o = AnswerOptions {
            noul_band: Some(NoulBand::default()),
            ..AnswerOptions::default()
        };
        assert_eq!(noul(0.2, &o), (0.2, Some(false)));
        assert_eq!(noul(0.8, &o), (0.8, Some(true)));
        assert_eq!(noul(0.5, &o), (0.5, None));
        assert_eq!(noul(0.21, &o).1, None);
        assert_eq!(noul(0.79, &o).1, None);
    }

    #[test]
    fn noul_commit_map_reports_the_mapped_probability() {
        let o = AnswerOptions {
            noul_band: Some(NoulBand::default()),
            noul_map: NoulMap::Commit,
            ..AnswerOptions::default()
        };
        assert_eq!(noul(0.5, &o), (0.8, Some(true)));
        let (p, d) = noul(0.49, &o);
        assert!((p - 0.199).abs() < 1e-12);
        assert_eq!(d, Some(false));
    }

    #[test]
    fn noul_errors() {
        let d = AnswerOptions::default();
        let three = meta(QuestionKind::Noul { yes: 1 }, 3);
        assert!(answer_from_probs(&three, vec![0.2, 0.3, 0.5], &d).is_err());
        let bad_yes = meta(QuestionKind::Noul { yes: 2 }, 2);
        assert!(answer_from_probs(&bad_yes, vec![0.5, 0.5], &d).is_err());
        let inverted = AnswerOptions {
            noul_band: Some(NoulBand {
                low: 0.8,
                high: 0.2,
            }),
            ..d
        };
        let m = meta(QuestionKind::Noul { yes: 1 }, 2);
        assert!(answer_from_probs(&m, vec![0.5, 0.5], &inverted).is_err());
    }

    #[test]
    fn score_default_values() {
        let d = AnswerOptions::default();
        let a = answer_from_probs(&meta(score(None, false), 3), vec![1.0 / 3.0; 3], &d).unwrap();
        let Answer::Score {
            expected,
            normalized,
            confidence,
            fit_mass,
            ..
        } = a
        else {
            panic!("expected Score");
        };
        assert!((expected - 1.0).abs() < 1e-12);
        assert!((normalized - 0.5).abs() < 1e-12);
        assert_eq!(confidence, 0.0);
        assert_eq!(fit_mass, None);
        // L = 5, p = [0, 0, 0.5, 0.5, 0]: expected 2.5, normalized 2.5 / 4.
        let p = vec![0.0, 0.0, 0.5, 0.5, 0.0];
        let b = answer_from_probs(&meta(score(None, false), 5), p, &d).unwrap();
        let Answer::Score {
            expected,
            normalized,
            confidence,
            ..
        } = b
        else {
            panic!("expected Score");
        };
        assert!((expected - 2.5).abs() < 1e-12);
        assert!((normalized - 0.625).abs() < 1e-12);
        assert!((confidence - (1.0 - 0.5 / 1.2)).abs() < 1e-12);
    }

    #[test]
    fn score_custom_values() {
        let d = AnswerOptions::default();
        let kind = score(Some(vec![1.0, 3.0, 5.0, 10.0]), false);
        let a = answer_from_probs(&meta(kind, 4), vec![0.1, 0.2, 0.3, 0.4], &d).unwrap();
        let Answer::Score {
            expected,
            normalized,
            ..
        } = a
        else {
            panic!("expected Score");
        };
        // 0.1 + 0.6 + 1.5 + 4.0 = 6.2; (6.2 - 1) / 9.
        assert!((expected - 6.2).abs() < 1e-12);
        assert!((normalized - 5.2 / 9.0).abs() < 1e-12);
        let flat = score(Some(vec![2.0, 2.0]), false);
        assert!(answer_from_probs(&meta(flat, 2), vec![0.5, 0.5], &d).is_err());
        let short = score(Some(vec![0.0, 1.0]), false);
        assert!(answer_from_probs(&meta(short, 3), vec![0.2, 0.3, 0.5], &d).is_err());
        let nan = score(Some(vec![0.0, f64::NAN]), false);
        assert!(answer_from_probs(&meta(nan, 2), vec![0.5, 0.5], &d).is_err());
    }

    #[test]
    fn score_chance_corrected_rule() {
        let o = AnswerOptions {
            score_confidence: ScoreConfidence::ChanceCorrected,
            ..AnswerOptions::default()
        };
        let a = answer_from_probs(&meta(score(None, false), 3), vec![0.1, 0.8, 0.1], &o).unwrap();
        let Answer::Score { confidence, .. } = a else {
            panic!("expected Score");
        };
        assert!((confidence - 0.7).abs() < 1e-12);
        let b = answer_from_probs(&meta(score(None, false), 3), vec![0.7, 0.0, 0.3], &o).unwrap();
        let Answer::Score { confidence, .. } = b else {
            panic!("expected Score");
        };
        // Chance-corrected: (0.7 - 1/3) / (2/3) = 0.55. Distance gives 1 - 0.6 / (2/3) = 0.1.
        assert!((confidence - 0.55).abs() < 1e-12);
    }

    #[test]
    fn score_isolated_combines() {
        let d = AnswerOptions::default();
        let a = answer_from_probs(&meta(score(None, true), 3), vec![0.2, 0.6, 0.2], &d).unwrap();
        let Answer::Score {
            expected,
            probs,
            fit_mass,
            ..
        } = a
        else {
            panic!("expected Score");
        };
        assert!((fit_mass.unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert!((expected - 1.0).abs() < 1e-12);
        assert!((probs[1] - 0.6).abs() < 1e-12);
        // Mass 0.5: levels 0.1, 0.3, 0.1 -> 0.2, 0.6, 0.2.
        let b = answer_from_probs(&meta(score(None, true), 3), vec![0.1, 0.3, 0.1], &d).unwrap();
        let Answer::Score {
            probs, fit_mass, ..
        } = b
        else {
            panic!("expected Score");
        };
        assert!((fit_mass.unwrap_or(0.0) - 0.5).abs() < 1e-12);
        assert!((probs[0] - 0.2).abs() < 1e-12 && (probs[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn combine_isolated_cases() {
        let (p, m) = combine_isolated(&[0.9, 0.9, 0.2]).unwrap();
        assert!((m - 2.0).abs() < 1e-12);
        assert!((p[0] - 0.45).abs() < 1e-12 && (p[2] - 0.1).abs() < 1e-12);
        assert!(combine_isolated(&[0.0, 0.0, 0.0]).is_err());
        assert!(combine_isolated(&[]).is_err());
        assert!(combine_isolated(&[0.5, 1.5]).is_err());
        assert!(combine_isolated(&[0.5, -0.1]).is_err());
        assert!(combine_isolated(&[0.5, f64::NAN]).is_err());
    }

    #[test]
    fn distribution_errors() {
        let d = AnswerOptions::default();
        let c3 = meta(QuestionKind::Choice, 3);
        assert!(answer_from_probs(&c3, vec![0.5, 0.5], &d).is_err());
        assert!(answer_from_probs(&c3, vec![0.5, 0.5, 0.5], &d).is_err());
        assert!(answer_from_probs(&c3, vec![1.2, -0.2, 0.0], &d).is_err());
        assert!(answer_from_probs(&c3, vec![f64::NAN, 0.5, 0.5], &d).is_err());
        assert!(answer_from_probs(&meta(QuestionKind::Choice, 0), vec![], &d).is_err());
        assert!(answer_from_probs(&meta(score(None, false), 1), vec![1.0], &d).is_err());
        // Within tolerance passes.
        assert!(answer_from_probs(&c3, vec![0.5, 0.25, 0.25 + 5e-7], &d).is_ok());
    }
}
