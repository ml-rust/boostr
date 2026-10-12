//! The last stage of a decision run: option logits to answers.
//!
//! The ragged rows are padded on the device to `[rows, K_max]` with a
//! `-inf` sentinel, divided by each row's temperature, and put through one
//! softmax. The raw and tempered rows come back to the host in one read,
//! and each question's answer is formed there.

use super::head::OptionLogits;
use crate::error::{Error, Result};
use crate::inference::decision::answer::{AnswerOptions, answer_from_probs};
use crate::inference::decision::temperature::TemperaturePolicy;
use crate::inference::decision::types::{DecisionResult, QuestionKind, QuestionMeta};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The index of the "yes" logit in an isolated Score level row.
pub const LEVEL_YES: usize = 1;

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// The scoring rows of each question, `(first row, row count)`, and the
/// temperature of every row.
struct Layout {
    spans: Vec<(usize, usize)>,
    temperatures: Vec<f64>,
}

fn layout(
    logits_counts: &[usize],
    meta: &[QuestionMeta],
    temp: &TemperaturePolicy,
) -> Result<Layout> {
    let mut spans = Vec::with_capacity(meta.len());
    let mut temperatures = Vec::with_capacity(logits_counts.len());
    let mut want: Vec<usize> = Vec::with_capacity(logits_counts.len());
    for m in meta {
        let first = want.len();
        if matches!(m.kind, QuestionKind::Score { isolated: true, .. }) {
            let level = QuestionMeta {
                kind: m.kind.clone(),
                options: 2,
            };
            let t = temp.resolve(&level)?;
            for _ in 0..m.options {
                want.push(2);
                temperatures.push(t);
            }
        } else {
            want.push(m.options);
            temperatures.push(temp.resolve(m)?);
        }
        spans.push((first, want.len() - first));
    }
    if want.len() != logits_counts.len() {
        return Err(invalid(
            "counts",
            format!(
                "the questions need {} scoring rows, the head gave {}",
                want.len(),
                logits_counts.len()
            ),
        ));
    }
    if let Some(r) = want.iter().position(|&k| k == 0) {
        return Err(invalid(
            "meta",
            format!("scoring row {r} belongs to a question with no options"),
        ));
    }
    if let Some(r) = (0..want.len()).find(|&r| want[r] != logits_counts[r]) {
        return Err(invalid(
            "counts",
            format!(
                "scoring row {r} holds {} logits, its question needs {}",
                logits_counts[r], want[r]
            ),
        ));
    }
    Ok(Layout {
        spans,
        temperatures,
    })
}

/// The answers to `meta` from the head's option logits.
///
/// Each scoring row is tempered by the policy's temperature for its
/// question; an isolated Score question's level rows each resolve as a
/// two-option row of that question's kind. A level's yes probability is
/// index [`LEVEL_YES`] of its tempered row, and `answer_from_probs`
/// combines the levels. The device softmax runs in F32 and each row is
/// renormalized in f64 on the host before the answer stage checks its sum.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `logits.flat` is not `[sum(counts)]`
/// F32, when the counts disagree with `meta`, or when a raw logit is not
/// finite, naming the row. Temperature and answer errors propagate.
pub fn finish<R, C>(
    client: &C,
    logits: &OptionLogits<R>,
    meta: &[QuestionMeta],
    temp: &TemperaturePolicy,
    opts: &AnswerOptions,
) -> Result<DecisionResult>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let counts = &logits.counts;
    let Layout {
        spans,
        temperatures,
    } = layout(counts, meta, temp)?;
    let total: usize = counts.iter().sum();
    let flat = &logits.flat;
    if flat.shape() != [total].as_slice() || flat.dtype() != DType::F32 {
        return Err(invalid(
            "flat",
            format!(
                "must be [{total}] F32 for counts {counts:?}, got {:?} {:?}",
                flat.shape(),
                flat.dtype()
            ),
        ));
    }
    let rows = counts.len();
    if rows == 0 {
        return Ok(DecisionResult {
            answers: Vec::new(),
            raw_logits: Vec::new(),
            temperatures,
        });
    }
    let k_max = counts.iter().copied().max().unwrap_or(1);
    let device = flat.device();

    // Index `total` is the -inf sentinel appended after the real logits.
    let mut index = Vec::with_capacity(rows * k_max);
    let mut start = 0usize;
    for &c in counts {
        index.extend((start..start + c).map(|i| i as i64));
        index.extend(std::iter::repeat_n(total as i64, k_max - c));
        start += c;
    }
    let sentinel = Tensor::<R>::full_scalar(&[1], DType::F32, f64::NEG_INFINITY, device)?;
    let extended = client.cat(&[flat, &sentinel], 0)?;
    let index = Tensor::<R>::from_slice(&index, &[rows * k_max], device)?;
    let padded = client
        .index_select(&extended, 0, &index)?
        .reshape(&[rows, k_max])?;
    let t: Vec<f32> = temperatures.iter().map(|&t| t as f32).collect();
    let t = Tensor::<R>::from_slice(&t, &[rows, 1], device)?;
    let probs = client.softmax(&client.div(&padded, &t)?, -1)?;
    let both = client.cat(&[&padded, &probs], 0)?;
    let host = both.contiguous()?.try_to_vec::<f32>()?;
    let (raw_host, prob_host) = host.split_at(rows * k_max);

    let mut raw_logits = Vec::with_capacity(rows);
    let mut dists: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for (r, &c) in counts.iter().enumerate() {
        let raw = &raw_host[r * k_max..r * k_max + c];
        if let Some(i) = raw.iter().position(|v| !v.is_finite()) {
            return Err(invalid(
                "flat",
                format!("scoring row {r} logit {i} is not finite: {}", raw[i]),
            ));
        }
        raw_logits.push(raw.to_vec());
        let mut p: Vec<f64> = prob_host[r * k_max..r * k_max + c]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let s: f64 = p.iter().sum();
        if !(s.is_finite() && s > 0.0) {
            return Err(invalid(
                "flat",
                format!("scoring row {r} has no probability mass: sum {s}"),
            ));
        }
        p.iter_mut().for_each(|v| *v /= s);
        dists.push(p);
    }

    let mut answers = Vec::with_capacity(meta.len());
    for (m, &(first, n)) in meta.iter().zip(&spans) {
        let probs = if matches!(m.kind, QuestionKind::Score { isolated: true, .. }) {
            (first..first + n).map(|r| dists[r][LEVEL_YES]).collect()
        } else {
            std::mem::take(&mut dists[first])
        };
        answers.push(answer_from_probs(m, probs, opts)?);
    }
    Ok(DecisionResult {
        answers,
        raw_logits,
        temperatures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::decision::temperature::tempered_softmax;
    use crate::inference::decision::types::Answer;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn meta(kind: QuestionKind, options: usize) -> QuestionMeta {
        QuestionMeta { kind, options }
    }

    fn logits(device: &CpuDevice, rows: &[&[f32]]) -> OptionLogits<CpuRuntime> {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        OptionLogits {
            flat: Tensor::<CpuRuntime>::from_slice(&flat, &[flat.len()], device).unwrap(),
            counts: rows.iter().map(|r| r.len()).collect(),
        }
    }

    fn by_kind() -> TemperaturePolicy {
        TemperaturePolicy::ByKind {
            choice: 0.5,
            noul: 2.0,
            score: 1.5,
        }
    }

    /// Ragged K with a one-option row: every row sums to 1, matches the
    /// host tempered softmax at its own temperature, and the pads add
    /// nothing.
    #[test]
    fn ragged_rows_temper_per_question() {
        let (client, device) = cpu_setup();
        let rows: [&[f32]; 4] = [
            &[0.3, -1.0, 2.0],
            &[4.0],
            &[0.1, 0.2, -3.0, 1.0, 0.5],
            &[1.0, -1.0],
        ];
        let metas = [
            meta(QuestionKind::Choice, 3),
            meta(QuestionKind::Choice, 1),
            meta(
                QuestionKind::Score {
                    values: None,
                    isolated: false,
                },
                5,
            ),
            meta(QuestionKind::Noul { yes: 1 }, 2),
        ];
        let opts = AnswerOptions::default();
        let out = finish(&client, &logits(&device, &rows), &metas, &by_kind(), &opts).unwrap();
        assert_eq!(out.temperatures, vec![0.5, 0.5, 1.5, 2.0]);
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(out.raw_logits[r], row.to_vec());
        }
        let probs = |a: &Answer| match a {
            Answer::Choice { probs, .. } | Answer::Score { probs, .. } => probs.clone(),
            other => panic!("unexpected {other:?}"),
        };
        for (r, (answer, row)) in out.answers.iter().zip(&rows).enumerate().take(3) {
            let got = probs(answer);
            assert_eq!(got.len(), row.len());
            assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let want = tempered_softmax(row, out.temperatures[r]).unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-6, "row {r}: {g} vs {w}");
            }
        }
        let Answer::Noul { p_yes, .. } = out.answers[3] else {
            panic!("expected Noul");
        };
        let want = tempered_softmax(rows[3], 2.0).unwrap()[1];
        assert!((p_yes - want).abs() < 1e-6);
    }

    /// Three level rows `[no, yes]` form one isolated Score answer.
    #[test]
    fn isolated_score_combines_level_rows() {
        let (client, device) = cpu_setup();
        let rows: [&[f32]; 4] = [&[0.0, 1.0], &[2.0, -1.0], &[0.5, 0.5], &[1.0, 2.0]];
        let kind = QuestionKind::Score {
            values: None,
            isolated: true,
        };
        let metas = [meta(kind, 3), meta(QuestionKind::Choice, 2)];
        let opts = AnswerOptions::default();
        let out = finish(&client, &logits(&device, &rows), &metas, &by_kind(), &opts).unwrap();
        assert_eq!(out.temperatures, vec![1.5, 1.5, 1.5, 0.5]);
        let yes: Vec<f64> = rows[..3]
            .iter()
            .map(|r| tempered_softmax(r, 1.5).unwrap()[LEVEL_YES])
            .collect();
        let want = answer_from_probs(&metas[0], yes, &opts).unwrap();
        let (
            Answer::Score {
                expected: e1,
                fit_mass: Some(m1),
                ..
            },
            Answer::Score {
                expected: e2,
                fit_mass: Some(m2),
                ..
            },
        ) = (&out.answers[0], &want)
        else {
            panic!("expected isolated Score answers");
        };
        assert!((e1 - e2).abs() < 1e-6 && (m1 - m2).abs() < 1e-6);
    }

    #[test]
    fn wrong_counts_and_bad_logits_are_refused() {
        let (client, device) = cpu_setup();
        let opts = AnswerOptions::default();
        let t = TemperaturePolicy::Fixed(1.0);
        let metas = [meta(QuestionKind::Choice, 3)];
        let e = finish(&client, &logits(&device, &[&[1.0, 2.0]]), &metas, &t, &opts)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("scoring row 0 holds 2 logits, its question needs 3"),
            "{e}"
        );
        let iso = [meta(
            QuestionKind::Score {
                values: None,
                isolated: true,
            },
            2,
        )];
        let e = finish(&client, &logits(&device, &[&[1.0, 2.0]]), &iso, &t, &opts)
            .unwrap_err()
            .to_string();
        assert!(e.contains("need 2 scoring rows, the head gave 1"), "{e}");
        let e = finish(
            &client,
            &logits(&device, &[&[1.0, f32::NAN, 0.0]]),
            &metas,
            &t,
            &opts,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("scoring row 0 logit 1 is not finite"), "{e}");
    }
}
