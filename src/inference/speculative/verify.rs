//! Verification of one step from the target logits, greedy or sampled.

use super::client::SpeculativeClient;
use super::executor::accept::greedy::match_drafts;
use super::penalty::penalized_logits;
use super::types::{Draw, SampleRule, Verdict};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Verify `drafts` against the target logits of one step and pick the next token.
///
/// - `logits`: `[1, M, V]` in any float dtype, `M = drafts.len() + 1`.
///   Row `i` scores the prefix `context ++ drafts[..i]`.
/// - `context`: the prefix row 0 scores. It ends with the last committed token.
/// - `drafts`: the draft ids. The drafter is deterministic, so each draft
///   distribution is one-hot.
///
/// Row `i` takes the penalties of `rule` over the penalty window of its
/// prefix: its last `rule.penalty_last_n` tokens, or all of it for 0. The
/// penalties run per row on the device through
/// [`SamplingOps::apply_sampling_penalties`](crate::ops::traits::inference::SamplingOps::apply_sampling_penalties).
///
/// A greedy rule takes the argmax of each row on the device. Ties break to
/// the lowest id. It keeps the longest draft prefix equal to those ids. Only
/// the `M` ids cross to the host. It draws no uniform and ignores `draw`.
///
/// A sampled rule filters through
/// [`SamplingProbsOps::sampling_probs`](crate::ops::traits::inference::SamplingProbsOps::sampling_probs).
/// The `M` uniforms come from `philox_uniform(draw.key, draw.counter)`. Only the
/// `[1, 2]` result of
/// [`SpeculativeOps::speculative_verify`](crate::ops::traits::inference::SpeculativeOps::speculative_verify)
/// crosses to the host.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a wrong logits shape or a bad `rule`.
/// [`Error::InferenceError`] when the verify result is out of range. A
/// device op error returns as is.
pub fn verify_logits<R, C>(
    client: &C,
    logits: &Tensor<R>,
    context: &[u32],
    drafts: &[u32],
    rule: &SampleRule,
    draw: Draw,
) -> Result<Verdict>
where
    R: Runtime<DType = DType>,
    C: SpeculativeClient<R>,
{
    rule.check()?;
    let rows = drafts.len() + 1;
    let shape = logits.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != rows || shape[2] == 0 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "verify_logits: expected [1, {rows}, vocab] for {} drafts, got {shape:?}",
                drafts.len()
            ),
        });
    }
    let vocab = shape[2];
    let device = logits.device();
    let penalized = if rule.has_penalties() {
        Some(penalized_logits(
            client, logits, context, drafts, rule, vocab,
        )?)
    } else {
        None
    };
    let scores = penalized.as_ref().unwrap_or(logits);

    if rule.is_greedy() {
        // [1, M, V] -> [1, M] ids on the device; M ids cross to the host.
        let ids: Vec<u32> = client
            .argmax(scores, 2, false)?
            .contiguous()?
            .to_vec::<i64>()
            .into_iter()
            .map(|id| u32::try_from(id).unwrap_or(u32::MAX))
            .collect();
        return match_drafts(drafts, &ids, vocab, "argmax");
    }

    let probs = client.sampling_probs(scores, &rule.filter)?;

    let ids: Vec<i32> = drafts
        .iter()
        .map(|&d| i32::try_from(d).unwrap_or(-1))
        .collect();
    let draft_tokens = Tensor::<R>::from_slice(&ids, &[1, drafts.len()], device)?;
    let uniforms = client.philox_uniform(&[1, rows], draw.key, draw.counter, DType::F32)?;
    let result = client.speculative_verify(&probs, None, &draft_tokens, &uniforms)?;

    let pair = result.contiguous()?.to_vec::<i32>();
    let (m, next) = match pair.as_slice() {
        &[m, next] => (m, next),
        other => {
            return Err(Error::InferenceError {
                reason: format!(
                    "speculative_verify returned {other:?}; expected one (m, next) pair"
                ),
            });
        }
    };
    let accepted = usize::try_from(m).ok().filter(|&m| m <= drafts.len());
    let next_id = u32::try_from(next).ok().filter(|&t| (t as usize) < vocab);
    match (accepted, next_id) {
        (Some(accepted), Some(next)) => Ok(Verdict { accepted, next }),
        _ => Err(Error::InferenceError {
            reason: format!(
                "speculative_verify returned (m {m}, next {next}) for {} drafts over {vocab} \
                 tokens; m must be at most the draft count and next below the vocabulary",
                drafts.len()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::inference::FilterRule;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    const V: usize = 5;

    fn logits(rows: &[[f32; V]]) -> Tensor<CpuRuntime> {
        let (_, device) = cpu_setup();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<CpuRuntime>::from_slice(&flat, &[1, rows.len(), V], &device).expect("logits")
    }

    fn top1() -> SampleRule {
        SampleRule::new(FilterRule {
            top_k: 1,
            ..FilterRule::temperature_only(1.0)
        })
    }

    #[test]
    fn top_k_one_accepts_argmax_drafts_and_emits_the_argmax() {
        let (client, _) = cpu_setup();
        let l = logits(&[
            [0.0, 2.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0],
        ]);
        for counter in 0..8 {
            let draw = Draw::for_step(0, counter);
            let all = verify_logits(&client, &l, &[4], &[1, 0], &top1(), draw).expect("verify");
            assert_eq!(
                all,
                Verdict {
                    accepted: 2,
                    next: 3
                }
            );
            let one = verify_logits(&client, &l, &[4], &[1, 2], &top1(), draw).expect("verify");
            assert_eq!(
                one,
                Verdict {
                    accepted: 1,
                    next: 0
                }
            );
        }
    }

    #[test]
    fn a_penalty_moves_the_argmax_and_leaves_the_input_unchanged() {
        let (client, _) = cpu_setup();
        let l = logits(&[[0.0, 2.0, 1.5, 0.0, 0.0], [0.0, 2.0, 1.5, 0.0, 0.0]]);
        let rule = SampleRule {
            presence_penalty: 1.0,
            ..top1()
        };
        let draw = Draw::for_step(7, 0);
        // Row 0 counts token 1 from the context, so token 2 wins.
        let got = verify_logits(&client, &l, &[1], &[2], &rule, draw).expect("verify");
        // Row 1 counts tokens 1 and 2: 1.0 vs 0.5, token 1 wins.
        assert_eq!(
            got,
            Verdict {
                accepted: 1,
                next: 1
            }
        );
        assert_eq!(l.to_vec::<f32>()[..V], [0.0, 2.0, 1.5, 0.0, 0.0]);
    }

    #[test]
    fn greedy_argmax_counts_only_the_penalty_window() {
        let (client, _) = cpu_setup();
        let l = logits(&[[0.0, 2.0, 1.5, 0.0, 0.0], [0.0, 2.0, 1.5, 0.0, 0.0]]);
        let mut rule = SampleRule {
            frequency_penalty: 0.3,
            ..top1()
        };
        rule.filter.temperature = 0.0;
        // Window 1 counts token 1 once: 1.7 beats 1.5. Windows 0 and 2 count it twice: 1.4.
        for (n, accepted, next) in [(1, 1, 1), (0, 0, 2), (2, 0, 2)] {
            let rule = SampleRule {
                penalty_last_n: n,
                ..rule
            };
            let draw = Draw::for_step(0, n as u64);
            let got = verify_logits(&client, &l, &[1, 1], &[1], &rule, draw).expect("verify");
            assert_eq!(got, Verdict { accepted, next }, "window {n}");
        }
    }

    #[test]
    fn no_drafts_draws_from_row_zero() {
        let (client, _) = cpu_setup();
        let l = logits(&[[0.0, 0.0, 0.0, 4.0, 0.0]]);
        let got = verify_logits(&client, &l, &[0], &[], &top1(), Draw::for_step(1, 2));
        assert_eq!(
            got.expect("verify"),
            Verdict {
                accepted: 0,
                next: 3
            }
        );
    }

    #[test]
    fn a_wrong_row_count_names_the_logits() {
        let (client, _) = cpu_setup();
        let l = logits(&[[0.0; V]]);
        match verify_logits(&client, &l, &[0], &[1], &top1(), Draw::for_step(0, 0)) {
            Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, "logits"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
