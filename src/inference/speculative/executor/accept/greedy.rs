//! Greedy acceptance: keep the drafts equal to the target's argmax ids.

use super::contract::check_extend;
use crate::error::{Error, Result};
use crate::inference::speculative::model::SpeculativeModel;
use crate::inference::speculative::types::Verdict;

/// Feed `feed` through `extend_greedy` and keep the longest matching draft prefix.
///
/// `feed` is the last committed token, then the drafts. The forward returns
/// the target ids `a_0..a_k`. The verdict keeps the longest prefix with
/// `d_{i+1} == a_i`, of length `m`, and `next = a_m`.
///
/// # Errors
///
/// [`Error::InferenceError`] when `feed` is empty or the model breaks its
/// contract. A model error returns as is.
pub(crate) fn accept<M: SpeculativeModel>(
    model: &mut M,
    feed: &[u32],
    len_before: usize,
) -> Result<Verdict> {
    let vocab_size = model.vocab_size();
    let drafts = feed.get(1..).ok_or_else(|| Error::InferenceError {
        reason: "a verify forward needs the last committed token".into(),
    })?;
    let ids = model.extend_greedy(feed)?;
    check_extend(feed.len(), ids.len(), len_before, model.len())?;
    match_drafts(drafts, &ids, vocab_size, "extend_greedy")
}

/// The verdict of `drafts` against the target argmax ids, one per verify row.
///
/// `ids[i]` is the argmax after `context ++ drafts[..i]`. The verdict keeps
/// the longest prefix with `drafts[i] == ids[i]`, of length `m`, and
/// `next = ids[m]`. `source` names the producer of `ids` in an error.
///
/// # Errors
///
/// [`Error::InferenceError`] when `ids` does not hold `drafts.len() + 1`
/// ids or holds an id outside `0..vocab_size`.
pub(crate) fn match_drafts(
    drafts: &[u32],
    ids: &[u32],
    vocab_size: usize,
    source: &str,
) -> Result<Verdict> {
    if ids.len() != drafts.len() + 1 {
        return Err(Error::InferenceError {
            reason: format!(
                "{source} returned {} ids for {} drafts; it must return one id per verify row",
                ids.len(),
                drafts.len()
            ),
        });
    }
    if let Some((j, &id)) = ids
        .iter()
        .enumerate()
        .find(|&(_, &id)| id as usize >= vocab_size)
    {
        return Err(Error::InferenceError {
            reason: format!(
                "{source} returned id {id} at row {j}, outside the model vocabulary of \
                 {vocab_size} tokens"
            ),
        });
    }
    let accepted = drafts.iter().zip(ids).take_while(|(d, a)| d == a).count();
    Ok(Verdict {
        accepted,
        next: ids[accepted],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::draft::{Drafter, PromptLookup};
    use crate::inference::speculative::executor::generate::SpeculativeGenerator;
    use crate::inference::speculative::toy_lm::{
        EmptyDrafter, HashDrafter, RandomDrafter, ToyLm, reference_penalized_argmax,
    };
    use crate::inference::speculative::types::{FinishReason, SampleRule, SpeculativeConfig};
    use crate::ops::traits::inference::FilterRule;
    use crate::test_utils::Lcg;

    #[test]
    fn match_drafts_keeps_the_longest_equal_prefix() {
        let got = match_drafts(&[4, 2, 9], &[4, 2, 7, 1], 10, "t").expect("valid");
        assert_eq!(
            got,
            Verdict {
                accepted: 2,
                next: 7
            }
        );
        let all = match_drafts(&[4, 2], &[4, 2, 3], 10, "t").expect("valid");
        assert_eq!(
            all,
            Verdict {
                accepted: 2,
                next: 3
            }
        );
        let none = match_drafts(&[], &[5], 10, "t").expect("valid");
        assert_eq!(
            none,
            Verdict {
                accepted: 0,
                next: 5
            }
        );
    }

    #[test]
    fn match_drafts_names_a_bad_id_list() {
        let err = match_drafts(&[1], &[1], 10, "argmax").expect_err("short");
        assert!(
            err.to_string()
                .contains("argmax returned 1 ids for 1 drafts"),
            "{err}"
        );
        let err = match_drafts(&[1], &[1, 10], 10, "argmax").expect_err("id 10");
        assert!(err.to_string().contains("id 10 at row 1"), "{err}");
    }

    /// Plain penalized greedy decoding: one token per step from the test oracle.
    fn plain_penalized(
        seed: u64,
        vocab: usize,
        prompt: &[u32],
        rule: &SampleRule,
        config: &SpeculativeConfig,
    ) -> (Vec<u32>, FinishReason) {
        let mut seq = prompt.to_vec();
        let mut out = Vec::new();
        loop {
            let t = reference_penalized_argmax(seed, vocab, &seq, rule);
            seq.push(t);
            out.push(t);
            if config.is_eos(t) {
                return (out, FinishReason::Eos);
            }
            if out.len() >= config.max_tokens {
                return (out, FinishReason::Length);
            }
        }
    }

    fn run<D: Drafter>(
        vocab: usize,
        seed: u64,
        drafter: D,
        config: &SpeculativeConfig,
        prompt: &[u32],
    ) -> (Vec<u32>, FinishReason, usize) {
        let mut g = SpeculativeGenerator::new(ToyLm::new(vocab, seed), drafter, config.clone())
            .expect("valid");
        let (out, finish, stats) = g.generate(prompt).expect("generate");
        let tokens = g.tokens();
        assert_eq!(g.model().fed.as_slice(), &tokens[..tokens.len() - 1]);
        assert!(stats.accepted <= stats.drafted, "{stats:?}");
        (out, finish, stats.accepted)
    }

    /// Proposes the oracle's own penalized greedy continuation.
    struct PenalizedOracle {
        seed: u64,
        vocab: usize,
        rule: SampleRule,
    }

    impl Drafter for PenalizedOracle {
        fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
            let mut seq = context.to_vec();
            for _ in 0..k {
                let t = reference_penalized_argmax(self.seed, self.vocab, &seq, &self.rule);
                seq.push(t);
            }
            seq.split_off(context.len())
        }
    }

    /// Greedy speculative output with penalties equals plain penalized greedy
    /// output for every drafter, window, draft count and EOS set.
    #[test]
    fn penalized_greedy_output_equals_plain_penalized_greedy() {
        const CASES: usize = 120;
        let mut rng = Lcg(0x9e17_a1ed);
        let mut oracle_accepted = 0usize;
        let mut differs_from_plain = 0usize;
        let mut eos_endings = 0usize;
        for case in 0..CASES {
            let vocab = 8 + rng.below(120);
            let seed = rng.below(1 << 30) as u64;
            let prompt: Vec<u32> = if rng.below(2) == 0 {
                let period = 1 + rng.below(3);
                (0..1 + rng.below(30))
                    .map(|i| ((i % period) * 3 % vocab) as u32)
                    .collect()
            } else {
                (0..1 + rng.below(30))
                    .map(|_| rng.below(vocab) as u32)
                    .collect()
            };
            let k = 1 + rng.below(8);
            let max_tokens = 1 + rng.below(40);
            let rule = SampleRule {
                repeat_penalty: [1.0, 1.1, 1.5, 0.8][rng.below(4)],
                frequency_penalty: [0.0, 0.3, -0.2][rng.below(3)],
                presence_penalty: [0.0, 0.6][rng.below(2)],
                penalty_last_n: [0, 2, 5, 64][rng.below(4)],
                ..SampleRule::new(FilterRule {
                    temperature: 0.0,
                    ..FilterRule::temperature_only(1.0)
                })
            };
            let rule = if rule.has_penalties() {
                rule
            } else {
                SampleRule {
                    presence_penalty: 1.0,
                    ..rule
                }
            };
            let base = SpeculativeConfig {
                num_draft_tokens: k,
                max_tokens,
                eos_token_ids: Vec::new(),
                sampling: Some(rule),
            };
            let (free, _) = plain_penalized(seed, vocab, &prompt, &rule, &base);
            let unpenalized = SpeculativeConfig {
                sampling: None,
                ..base.clone()
            };
            let (plain_out, _, _) = run(vocab, seed, EmptyDrafter, &unpenalized, &prompt);
            if plain_out != free {
                differs_from_plain += 1;
            }
            let eos = SpeculativeConfig {
                eos_token_ids: vec![free[rng.below(free.len())]],
                ..base.clone()
            };
            for config in [base, eos] {
                let want = plain_penalized(seed, vocab, &prompt, &rule, &config);
                if want.1 == FinishReason::Eos {
                    eos_endings += 1;
                }
                let what = format!("case {case} vocab {vocab} seed {seed} config {config:?}");
                let (out, finish, _) = run(vocab, seed, PromptLookup::default(), &config, &prompt);
                assert_eq!((out, finish), want.clone(), "{what}: prompt lookup");
                let oracle = PenalizedOracle { seed, vocab, rule };
                let (out, finish, accepted) = run(vocab, seed, oracle, &config, &prompt);
                assert_eq!((out, finish), want.clone(), "{what}: oracle");
                oracle_accepted += accepted;
                let random = RandomDrafter {
                    state: seed ^ case as u64,
                    vocab,
                };
                let (out, finish, _) = run(vocab, seed, random, &config, &prompt);
                assert_eq!((out, finish), want.clone(), "{what}: random");
                let hash = HashDrafter { salt: seed, vocab };
                let (out, finish, _) = run(vocab, seed, hash, &config, &prompt);
                assert_eq!((out, finish), want.clone(), "{what}: hash");
                let (out, finish, _) = run(vocab, seed, EmptyDrafter, &config, &prompt);
                assert_eq!((out, finish), want, "{what}: empty");
            }
        }
        assert!(
            oracle_accepted > 0,
            "the oracle never matched; the test is vacuous"
        );
        assert!(
            differs_from_plain > 0,
            "penalties changed no output of {CASES} cases; the test is vacuous"
        );
        assert!(eos_endings > 0, "no run ended at EOS; the test is vacuous");
    }
}
