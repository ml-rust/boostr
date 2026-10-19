//! Sampled acceptance: rejection sampling of the drafts by the target model.

use super::contract::check_growth;
use crate::error::{Error, Result};
use crate::inference::speculative::model::{SpeculativeModel, SpeculativeProbModel};
use crate::inference::speculative::types::{Draw, SampleRule, Verdict};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Feed `feed` through `extend_sampled` and check the verdict.
///
/// `feed` is the last committed token, then the drafts.
///
/// # Errors
///
/// [`Error::InferenceError`] when `feed` is empty, the cache does not grow by
/// `feed.len()`, or the verdict accepts more drafts than were fed or names
/// a token outside the vocabulary. A model error returns as is.
pub(crate) fn accept<M: SpeculativeModel>(
    model: &mut M,
    feed: &[u32],
    rule: &SampleRule,
    draw: Draw,
    len_before: usize,
) -> Result<Verdict> {
    let drafts = draft_count(feed)?;
    let verdict = model.extend_sampled(feed, rule, draw)?;
    check_verdict(model, feed.len(), drafts, verdict, len_before)
}

/// [`accept`] through `extend_sampled_with_probs`, with the draft distributions `q`.
///
/// `q` is `[1, feed.len() - 1, V]` F32, or `None` for one-hot drafts.
///
/// # Errors
///
/// Every error of [`accept`].
pub(crate) fn accept_with_probs<R: Runtime, M: SpeculativeProbModel<R>>(
    model: &mut M,
    feed: &[u32],
    rule: &SampleRule,
    draw: Draw,
    q: Option<&Tensor<R>>,
    len_before: usize,
) -> Result<Verdict> {
    let drafts = draft_count(feed)?;
    let verdict = model.extend_sampled_with_probs(feed, rule, draw, q)?;
    check_verdict(model, feed.len(), drafts, verdict, len_before)
}

/// The drafts in `feed`: every token after the last committed one.
fn draft_count(feed: &[u32]) -> Result<usize> {
    feed.len()
        .checked_sub(1)
        .ok_or_else(|| Error::InferenceError {
            reason: "a verify forward needs the last committed token".into(),
        })
}

/// Check the cache growth and the verdict range of one sampled verify forward.
fn check_verdict<M: SpeculativeModel>(
    model: &M,
    fed: usize,
    drafts: usize,
    verdict: Verdict,
    len_before: usize,
) -> Result<Verdict> {
    let vocab_size = model.vocab_size();
    check_growth(fed, len_before, model.len())?;
    if verdict.accepted > drafts {
        return Err(Error::InferenceError {
            reason: format!(
                "extend_sampled accepted {} drafts of {drafts}; it must accept at most the fed \
                 drafts",
                verdict.accepted
            ),
        });
    }
    if verdict.next as usize >= vocab_size {
        return Err(Error::InferenceError {
            reason: format!(
                "extend_sampled returned id {}, outside the model vocabulary of {vocab_size} \
                 tokens",
                verdict.next
            ),
        });
    }
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::draft::PromptLookup;
    use crate::inference::speculative::executor::generate::SpeculativeGenerator;
    use crate::inference::speculative::executor::source::DraftSource;
    use crate::inference::speculative::toy_lm::{
        HashDrafter, Oracle, RandomDrafter, ToyLm, plain_greedy, reference_probs, toy_drafter,
    };
    use crate::inference::speculative::types::{
        COUNTERS_PER_STEP, FinishReason, SpeculativeConfig, SpeculativeStats,
    };
    use crate::ops::traits::inference::FilterRule;

    const V8: usize = 8;
    const TOY_SEED: u64 = 0x70e5_eed5;
    const PROMPT: [u32; 5] = [1, 2, 3, 1, 2];
    /// Runs per distribution test; run `r` uses seed `r`.
    const RUNS: u64 = 1 << 14;
    /// The upper normal quantile at alpha = 1e-6.
    const Z: f64 = 4.753;

    fn filter(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> FilterRule {
        FilterRule {
            temperature,
            top_k,
            top_p,
            min_p,
        }
    }

    fn sampled(k: usize, max_tokens: usize, rule: SampleRule) -> SpeculativeConfig {
        SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens,
            eos_token_ids: Vec::new(),
            sampling: Some(rule),
        }
    }

    /// Grows its cache as told and returns a fixed verdict.
    struct Fixed {
        len: usize,
        verdict: Verdict,
    }

    impl SpeculativeModel for Fixed {
        fn vocab_size(&self) -> usize {
            10
        }
        fn len(&self) -> usize {
            self.len
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            self.len = len;
            Ok(())
        }
        fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
            self.len += tokens.len();
            Ok(vec![0; tokens.len()])
        }
        fn extend_sampled(&mut self, tokens: &[u32], _: &SampleRule, _: Draw) -> Result<Verdict> {
            self.len += tokens.len();
            Ok(self.verdict)
        }
    }

    fn run_fixed(accepted: usize, next: u32) -> Result<Verdict> {
        let mut model = Fixed {
            len: 0,
            verdict: Verdict { accepted, next },
        };
        let rule = SampleRule::new(FilterRule::temperature_only(1.0));
        accept(&mut model, &[1, 2, 3], &rule, Draw::for_step(0, 0), 0)
    }

    #[test]
    fn an_in_range_verdict_passes() {
        assert_eq!(
            run_fixed(2, 9).expect("valid"),
            Verdict {
                accepted: 2,
                next: 9
            }
        );
    }

    #[test]
    fn an_out_of_range_verdict_is_an_error() {
        let err = run_fixed(3, 1).expect_err("3 of 2 drafts");
        assert!(err.to_string().contains("accepted 3 drafts of 2"), "{err}");
        let err = run_fixed(0, 10).expect_err("id 10");
        assert!(err.to_string().contains("id 10"), "{err}");
    }

    /// Keeps the default `extend_sampled`, which refuses every call.
    struct GreedyOnly(ToyLm);

    impl SpeculativeModel for GreedyOnly {
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            self.0.truncate(len)
        }
        fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
            self.0.extend_greedy(tokens)
        }
    }

    /// Records the draw of every sampled step.
    struct Recording {
        toy: ToyLm,
        draws: Vec<Draw>,
    }

    impl SpeculativeModel for Recording {
        fn vocab_size(&self) -> usize {
            self.toy.vocab_size()
        }
        fn len(&self) -> usize {
            self.toy.len()
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            self.toy.truncate(len)
        }
        fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
            self.toy.extend_greedy(tokens)
        }
        fn extend_sampled(
            &mut self,
            tokens: &[u32],
            rule: &SampleRule,
            draw: Draw,
        ) -> Result<Verdict> {
            self.draws.push(draw);
            self.toy.extend_sampled(tokens, rule, draw)
        }
    }

    /// One sampled run on a toy of `vocab` tokens. Checks the cache invariant.
    fn run_toy<D: DraftSource<ToyLm>>(
        vocab: usize,
        seed: u64,
        drafter: D,
        config: SpeculativeConfig,
        prompt: &[u32],
    ) -> (Vec<u32>, FinishReason, SpeculativeStats) {
        let mut g =
            SpeculativeGenerator::new(ToyLm::new(vocab, seed), drafter, config).expect("valid");
        let result = g.generate(prompt).expect("generate");
        let tokens = g.tokens();
        assert_eq!(g.model().fed.as_slice(), &tokens[..tokens.len() - 1]);
        assert_eq!(result.2.steps, result.2.target_forwards);
        result
    }

    #[test]
    fn no_rule_or_temperature_zero_runs_the_greedy_path() {
        let greedy_rule = SampleRule {
            seed: Some(4),
            ..SampleRule::new(filter(0.0, 3, 0.5, 0.1))
        };
        for seed in 0..20u64 {
            let vocab = 30;
            let prompt = [(seed % 30) as u32, 4, 7, 4, 7];
            let plain = SpeculativeConfig {
                num_draft_tokens: 3,
                ..SpeculativeConfig::new(25)
            };
            let want = plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &plain).expect("plain");
            for sampling in [None, Some(greedy_rule)] {
                let config = SpeculativeConfig {
                    sampling,
                    ..plain.clone()
                };
                let model = GreedyOnly(ToyLm::new(vocab, seed));
                let mut g = SpeculativeGenerator::new(model, PromptLookup::default(), config)
                    .expect("valid");
                let (out, finish, _) = g.generate(&prompt).expect("no extend_sampled call");
                assert_eq!((out, finish), want, "seed {seed} sampling {sampling:?}");
            }
        }
        let config = sampled(3, 5, SampleRule::new(FilterRule::temperature_only(1.0)));
        let model = GreedyOnly(ToyLm::new(30, 1));
        let mut g =
            SpeculativeGenerator::new(model, PromptLookup::default(), config).expect("valid");
        let err = g.generate(&[1, 2]).expect_err("a greedy-only model");
        assert!(err.to_string().contains("extend_sampled"), "{err}");
    }

    #[test]
    fn top_k_one_equals_plain_greedy() {
        for seed in 0..12u64 {
            let vocab = 8 + (seed as usize * 7) % 40;
            let prompt: Vec<u32> = (0..6).map(|i| ((seed + i * 3) % 5) as u32).collect();
            let base = SpeculativeConfig {
                num_draft_tokens: 4,
                ..SpeculativeConfig::new(20)
            };
            let rule = SampleRule {
                seed: Some(seed),
                ..SampleRule::new(filter(1.0, 1, 1.0, 0.0))
            };
            let free = plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &base).expect("plain");
            let eos = SpeculativeConfig {
                eos_token_ids: vec![free.0[free.0.len() / 2]],
                ..base.clone()
            };
            for plain in [base, eos] {
                let want =
                    plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &plain).expect("plain");
                let config = SpeculativeConfig {
                    sampling: Some(rule),
                    ..plain
                };
                let what = format!("seed {seed} vocab {vocab} config {config:?}");
                let (out, finish, _) = run_toy(
                    vocab,
                    seed,
                    PromptLookup::default(),
                    config.clone(),
                    &prompt,
                );
                assert_eq!((out, finish), want.clone(), "{what}: prompt lookup");
                let (out, finish, stats) =
                    run_toy(vocab, seed, Oracle { seed, vocab }, config.clone(), &prompt);
                assert_eq!((out, finish), want.clone(), "{what}: oracle");
                assert_eq!(stats.accepted, stats.drafted, "{what}: {stats:?}");
                let random = RandomDrafter { state: seed, vocab };
                let (out, finish, _) = run_toy(vocab, seed, random, config, &prompt);
                assert_eq!((out, finish), want, "{what}: random");
            }
        }
    }

    /// The tokens and draws of one sampled run of 40 tokens.
    fn recorded_run(seed: Option<u64>) -> (Vec<u32>, Vec<Draw>) {
        let rule = SampleRule {
            seed,
            ..SampleRule::new(FilterRule::temperature_only(1.0))
        };
        let model = Recording {
            toy: ToyLm::new(V8, TOY_SEED),
            draws: Vec::new(),
        };
        let drafter = HashDrafter { salt: 5, vocab: V8 };
        let mut g = SpeculativeGenerator::new(model, drafter, sampled(3, 40, rule)).expect("valid");
        let (out, _, stats) = g.generate(&PROMPT).expect("generate");
        let (model, _) = g.into_parts();
        assert_eq!(model.draws.len(), stats.steps);
        (out, model.draws)
    }

    #[test]
    fn a_seed_fixes_the_output_and_steps_never_share_a_counter() {
        let (a, draws) = recorded_run(Some(0));
        assert_eq!(recorded_run(Some(0)), (a.clone(), draws.clone()));
        let (c, _) = recorded_run(Some(1));
        assert_ne!(a, c, "seeds 0 and 1 give the same 40 tokens");
        assert!(draws.len() >= 2, "{draws:?}");
        for (s, d) in draws.iter().enumerate() {
            assert_eq!(*d, Draw::for_step(0, s as u64));
        }
        for w in draws.windows(2) {
            assert!(w[0].counter + COUNTERS_PER_STEP <= w[1].counter, "{w:?}");
        }
    }

    #[test]
    fn an_unseeded_rule_follows_numr_with_seed() {
        let run = || numr::random::with_seed(42, || recorded_run(None));
        let (a, draws) = run();
        assert_eq!(run(), (a, draws.clone()));
        assert!(draws.iter().all(|d| d.key == draws[0].key), "{draws:?}");
    }

    #[test]
    fn a_restart_repeats_the_seeded_output() {
        let rule = SampleRule {
            seed: Some(3),
            ..SampleRule::new(FilterRule::temperature_only(1.0))
        };
        let drafter = HashDrafter { salt: 1, vocab: V8 };
        let mut g =
            SpeculativeGenerator::new(ToyLm::new(V8, TOY_SEED), drafter, sampled(2, 30, rule))
                .expect("valid");
        let first = g.generate(&PROMPT).expect("first");
        assert_eq!(g.generate(&PROMPT).expect("second"), first);
    }

    /// `P(t1, t2, t3)` of the first three tokens after `PROMPT`, enumerated exactly.
    fn joint_reference(rule: &SampleRule) -> Vec<f64> {
        let probs = |prefix: &[u32]| {
            let (p, margin) = reference_probs(TOY_SEED, V8, prefix, rule);
            assert!(
                margin > 1e-5,
                "{rule:?}: a filter threshold sits {margin:e} from a value after {prefix:?}; \
                 pick other filter values"
            );
            p
        };
        let mut joint = vec![0.0; V8 * V8 * V8];
        let mut prefix = PROMPT.to_vec();
        let p1 = probs(&prefix);
        for (t1, &a) in p1.iter().enumerate().filter(|(_, p)| **p > 0.0) {
            prefix.push(t1 as u32);
            let p2 = probs(&prefix);
            for (t2, &b) in p2.iter().enumerate().filter(|(_, p)| **p > 0.0) {
                prefix.push(t2 as u32);
                for (t3, &c) in probs(&prefix).iter().enumerate() {
                    joint[(t1 * V8 + t2) * V8 + t3] = a * b * c;
                }
                prefix.pop();
            }
            prefix.pop();
        }
        let total: f64 = joint.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "joint mass {total}");
        joint
    }

    /// The chi-square critical value at `df` by the Wilson-Hilferty approximation.
    fn wilson_hilferty(df: usize, z: f64) -> f64 {
        let k = df as f64;
        let a = 2.0 / (9.0 * k);
        k * (1.0 - a + z * a.sqrt()).powi(3)
    }

    /// Pearson chi-square of `observed` against `probs`. Cells under 5 expected pool together.
    fn chi_square_check(name: &str, observed: &[u64], probs: &[f64], runs: u64) {
        let n = runs as f64;
        let mut cells: Vec<(f64, f64)> = Vec::new();
        let (mut pooled_o, mut pooled_e) = (0.0, 0.0);
        for (i, (&o, &p)) in observed.iter().zip(probs).enumerate() {
            if p == 0.0 {
                assert_eq!(o, 0, "{name}: cell {i} has probability 0 and {o} draws");
                continue;
            }
            let e = n * p;
            if e < 5.0 {
                pooled_o += o as f64;
                pooled_e += e;
            } else {
                cells.push((o as f64, e));
            }
        }
        if pooled_e > 0.0 {
            match cells.iter_mut().min_by(|a, b| a.1.total_cmp(&b.1)) {
                Some(smallest) if pooled_e < 5.0 => {
                    smallest.0 += pooled_o;
                    smallest.1 += pooled_e;
                }
                _ => cells.push((pooled_o, pooled_e)),
            }
        }
        assert!(
            cells.len() >= 2,
            "{name}: {} cells; the test is vacuous",
            cells.len()
        );
        let df = cells.len() - 1;
        let stat: f64 = cells.iter().map(|&(o, e)| (o - e).powi(2) / e).sum();
        let critical = wilson_hilferty(df, Z);
        assert!(
            stat < critical,
            "{name}: chi-square {stat:.1} over {df} df exceeds {critical:.1} at alpha 1e-6"
        );
    }

    /// The first three tokens of `RUNS` sampled runs follow the exact joint distribution.
    ///
    /// `expect_accepts` asks for at least one accepted draft. A strict filter can give a
    /// deterministic drafter's token zero probability on every run, so only drafters
    /// and filters that keep it in the kept set set this.
    fn check_distribution<D: DraftSource<ToyLm>>(
        name: &str,
        rule: SampleRule,
        expect_accepts: bool,
        mut drafter: impl FnMut() -> D,
    ) {
        let want = joint_reference(&rule);
        let mut observed = vec![0u64; want.len()];
        let (mut drafted, mut accepted) = (0usize, 0usize);
        for r in 0..RUNS {
            let config = sampled(
                2,
                3,
                SampleRule {
                    seed: Some(r),
                    ..rule
                },
            );
            let (out, finish, stats) = run_toy(V8, TOY_SEED, drafter(), config, &PROMPT);
            assert_eq!(
                (out.len(), finish),
                (3, FinishReason::Length),
                "{name}: run {r}"
            );
            observed[(out[0] as usize * V8 + out[1] as usize) * V8 + out[2] as usize] += 1;
            drafted += stats.drafted;
            accepted += stats.accepted;
        }
        assert!(
            drafted > 0 && accepted < drafted && (accepted > 0 || !expect_accepts),
            "{name}: {accepted} of {drafted} drafts accepted; the expected branches must run"
        );
        chi_square_check(name, &observed, &want, RUNS);
    }

    #[test]
    fn prompt_lookup_output_follows_the_target_distribution() {
        let rule = SampleRule::new(filter(0.9, 6, 0.95, 0.02));
        check_distribution("prompt lookup", rule, false, PromptLookup::default);
    }

    #[test]
    fn prompt_lookup_accepts_drafts_under_an_open_filter() {
        let rule = SampleRule::new(filter(1.0, 0, 1.0, 0.0));
        check_distribution("prompt lookup open", rule, true, PromptLookup::default);
    }

    #[test]
    fn hash_drafter_output_follows_the_target_distribution() {
        let rule = SampleRule::new(filter(1.3, 0, 0.8, 0.0));
        check_distribution("hash", rule, true, || HashDrafter { salt: 9, vocab: V8 });
    }

    #[test]
    fn penalized_output_follows_the_penalized_distribution() {
        let rule = SampleRule {
            repeat_penalty: 1.3,
            frequency_penalty: 0.2,
            presence_penalty: 0.4,
            ..SampleRule::new(filter(1.0, 0, 1.0, 0.05))
        };
        check_distribution("penalties", rule, true, || HashDrafter {
            salt: 9,
            vocab: V8,
        });
    }

    #[test]
    fn windowed_penalties_follow_the_windowed_distribution() {
        for last_n in [2, 5] {
            let rule = SampleRule {
                repeat_penalty: 1.3,
                frequency_penalty: 0.2,
                presence_penalty: 0.4,
                penalty_last_n: last_n,
                ..SampleRule::new(filter(1.0, 0, 1.0, 0.05))
            };
            let whole = SampleRule {
                penalty_last_n: 0,
                ..rule
            };
            assert_ne!(
                joint_reference(&rule),
                joint_reference(&whole),
                "window {last_n} gives the whole-prefix distribution; the test is vacuous"
            );
            let name = format!("penalty window {last_n}");
            check_distribution(&name, rule, true, || HashDrafter { salt: 9, vocab: V8 });
        }
    }

    /// A draft model of another seed drafts from a q that is not one-hot.
    /// Every rejection then draws from the residual `max(0, p - q)`.
    const DRAFT_SEED: u64 = TOY_SEED ^ 0x0d2a_f75e;

    #[test]
    fn draft_model_output_follows_the_target_distribution() {
        let rule = SampleRule::new(filter(1.0, 0, 1.0, 0.0));
        check_distribution("draft model", rule, true, || toy_drafter(V8, DRAFT_SEED));
    }

    #[test]
    fn filtered_draft_model_output_follows_the_target_distribution() {
        let rule = SampleRule::new(filter(0.9, 6, 0.95, 0.02));
        check_distribution("draft model filtered", rule, true, || {
            toy_drafter(V8, DRAFT_SEED)
        });
    }

    #[test]
    fn penalized_draft_model_output_follows_the_penalized_distribution() {
        let rule = SampleRule {
            repeat_penalty: 1.3,
            frequency_penalty: 0.2,
            presence_penalty: 0.4,
            penalty_last_n: 2,
            ..SampleRule::new(filter(1.0, 0, 1.0, 0.05))
        };
        check_distribution("draft model penalties", rule, true, || {
            toy_drafter(V8, DRAFT_SEED)
        });
    }
}
