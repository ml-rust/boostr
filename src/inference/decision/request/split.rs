//! [`split_result`]: the result of a merged request cut back into each
//! request's result.

use super::merge::MergedPart;
use crate::error::{Error, Result};
use crate::inference::decision::types::DecisionResult;

/// Cuts the result of a merged request into each request's result, in
/// the order the requests were merged. Answers follow each part's
/// questions; raw logit rows and temperatures follow its scoring rows.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `result` does not hold the answers,
/// raw logit rows and temperatures `parts` describe, naming the counts.
pub fn split_result(result: DecisionResult, parts: &[MergedPart]) -> Result<Vec<DecisionResult>> {
    let questions = parts.last().map_or(0, |p| p.questions.end);
    let scoring = parts.last().map_or(0, |p| p.scoring_rows.end);
    if result.answers.len() != questions
        || result.raw_logits.len() != scoring
        || result.temperatures.len() != scoring
    {
        return Err(Error::InvalidArgument {
            arg: "result",
            reason: format!(
                "holds {} answers, {} raw logit rows and {} temperatures; the parts need {} \
                 answers and {} of each row",
                result.answers.len(),
                result.raw_logits.len(),
                result.temperatures.len(),
                questions,
                scoring
            ),
        });
    }
    let DecisionResult {
        answers,
        raw_logits,
        temperatures,
    } = result;
    let mut answers = answers.into_iter();
    let mut raw_logits = raw_logits.into_iter();
    let mut temperatures = temperatures.into_iter();
    Ok(parts
        .iter()
        .map(|p| DecisionResult {
            answers: answers.by_ref().take(p.questions.len()).collect(),
            raw_logits: raw_logits.by_ref().take(p.scoring_rows.len()).collect(),
            temperatures: temperatures.by_ref().take(p.scoring_rows.len()).collect(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::merge::tests::{choice, isolated, two};
    use super::super::merge::{DecisionRequest, merge_requests};
    use super::super::rows::{RowSlot, plan_from_rows};
    use super::*;
    use crate::inference::decision::answer::AnswerOptions;
    use crate::inference::decision::run::RunnerConfig;
    use crate::inference::decision::run::runner::run_decision;
    use crate::inference::decision::run::runner::tests::{close, model, tokens};
    use crate::inference::decision::temperature::TemperaturePolicy;
    use crate::inference::decision::types::{Answer, QuestionMeta};
    use crate::model::decision::{LabelTokenHead, LabelTokenItem};
    use crate::test_utils::cpu_setup;

    fn answer(n: usize) -> Answer {
        Answer::Choice {
            index: n,
            confidence: n as f64,
            probs: vec![n as f64],
        }
    }

    /// Splitting a result shaped like the merge gives each request its own
    /// answers, raw logit rows and temperatures, ragged K included.
    #[test]
    fn split_round_trips() {
        let merged = merge_requests(two()).unwrap();
        let raw: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3],
            vec![1.1, 1.2, 1.3],
            vec![2.1, 2.2],
            vec![3.0, 3.5],
            vec![4.0, 4.5],
            vec![5.0, 5.5],
        ];
        let temps = vec![0.5, 0.5, 0.5, 2.0, 2.0, 2.0];
        let whole = DecisionResult {
            answers: (0..4).map(answer).collect(),
            raw_logits: raw.clone(),
            temperatures: temps.clone(),
        };
        let split = split_result(whole, &merged.parts).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].answers, (0..3).map(answer).collect::<Vec<_>>());
        assert_eq!(split[0].raw_logits, raw[..3].to_vec());
        assert_eq!(split[0].temperatures, temps[..3].to_vec());
        assert_eq!(split[1].answers, vec![answer(3)]);
        assert_eq!(split[1].raw_logits, raw[3..].to_vec());
        assert_eq!(split[1].temperatures, vec![2.0; 3]);

        let short = DecisionResult {
            answers: (0..4).map(answer).collect(),
            raw_logits: raw[..5].to_vec(),
            temperatures: temps,
        };
        let e = split_result(short, &merged.parts).unwrap_err().to_string();
        assert!(
            e.contains("holds 4 answers, 5 raw logit rows and 6 temperatures"),
            "{e}"
        );
        assert!(e.contains("need 4 answers and 6 of each row"), "{e}");
    }

    fn labelled(
        rows: Vec<Vec<u32>>,
        slots: &[RowSlot],
        labels: &[&[u32]],
        meta: Vec<QuestionMeta>,
        temp: TemperaturePolicy,
    ) -> DecisionRequest<LabelTokenItem> {
        let built = plan_from_rows(rows, slots, usize::MAX).unwrap();
        let items = built
            .slot_gathers
            .iter()
            .zip(labels)
            .map(|(&slot, l)| LabelTokenItem {
                slot,
                labels: l.to_vec(),
            })
            .collect();
        DecisionRequest {
            plan: built.plan,
            items,
            meta,
            temp,
        }
    }

    fn run(req: &DecisionRequest<LabelTokenItem>) -> DecisionResult {
        let (client, device) = cpu_setup();
        run_decision(
            &client,
            &model(&device),
            &LabelTokenHead,
            &req.plan,
            None,
            &req.items,
            &req.meta,
            &req.temp,
            &AnswerOptions::default(),
            &RunnerConfig::default(),
        )
        .unwrap()
    }

    fn probs(a: &Answer) -> Vec<f32> {
        match a {
            Answer::Choice { probs, .. } | Answer::Score { probs, .. } => {
                probs.iter().map(|&p| p as f32).collect()
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// Two requests with different temperatures, K and kinds, run merged,
    /// give each request the result it gets alone.
    #[test]
    fn merged_run_matches_runs_alone() {
        let a = labelled(
            vec![tokens(7, 1), tokens(5, 2)],
            &[
                RowSlot { row: 0, pos: 6 },
                RowSlot { row: 1, pos: 4 },
                RowSlot { row: 0, pos: 3 },
            ],
            &[&[1, 2, 3], &[4, 5], &[6, 7]],
            vec![choice(3), choice(2), choice(2)],
            TemperaturePolicy::Fixed(0.7),
        );
        let b = labelled(
            vec![tokens(4, 3)],
            &[
                RowSlot { row: 0, pos: 1 },
                RowSlot { row: 0, pos: 2 },
                RowSlot { row: 0, pos: 3 },
            ],
            &[&[1, 2], &[3, 4], &[5, 6]],
            vec![isolated(3)],
            TemperaturePolicy::Fixed(1.4),
        );
        let merged = merge_requests(vec![a.clone(), b.clone()]).unwrap();
        let split = split_result(run(&merged.request), &merged.parts).unwrap();
        for (alone, got) in [run(&a), run(&b)].iter().zip(&split) {
            assert_eq!(got.temperatures, alone.temperatures);
            assert_eq!(got.answers.len(), alone.answers.len());
            for (g, w) in got.raw_logits.iter().zip(&alone.raw_logits) {
                close(g, w, 1e-5, "raw logits");
            }
            for (g, w) in got.answers.iter().zip(&alone.answers) {
                close(&probs(g), &probs(w), 1e-5, "probabilities");
            }
        }
    }
}
