//! Generator state, start-up and the generation loop.

use super::accept::check_extend;
use super::step::StepOutput;
use crate::error::{Error, Result};
use crate::inference::speculative::draft::Drafter;
use crate::inference::speculative::model::SpeculativeModel;
use crate::inference::speculative::types::{FinishReason, SpeculativeConfig, SpeculativeStats};

/// Where the generator is in its life cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    /// No prompt, or the last call returned an error. Steps are refused.
    Idle,
    /// Steps emit tokens.
    Running,
    /// The output ended. Steps are refused.
    Finished(FinishReason),
}

/// Speculative generation over a target model and a drafter.
///
/// With a greedy rule, the output equals plain greedy decoding of the same
/// model token for token, penalties included. With a temperature above 0,
/// every emitted token follows the target's filtered distribution exactly.
/// Each [`step`](Self::step) costs one target forward and emits one or more
/// tokens.
///
/// # Randomness
///
/// [`start`](Self::start) fixes the Philox key: the rule's seed, or one
/// `numr::random::next_call_seed` value. So `manual_seed` and `with_seed`
/// make an unseeded run repeatable. Step `s` draws at
/// [`Draw::for_step`](crate::inference::speculative::Draw::for_step)`(key, s)`.
/// Two steps never share a Philox counter. A greedy rule draws no seed.
///
/// # Cache invariant
///
/// Between steps the model cache holds every committed token except the last.
/// Committed tokens are the prompt plus every emitted token. A step feeds
/// that last token and the drafts in one forward. It then truncates the cache
/// so that the cache holds the new committed tokens except the new last.
/// Rejected drafts and drafts past an EOS cut leave the cache that way.
///
/// The generator is synchronous and owns no thread. A server can run steps on
/// a blocking thread, stream each [`StepOutput`], and stop by dropping it.
pub struct SpeculativeGenerator<M, D> {
    pub(super) model: M,
    pub(super) drafter: D,
    pub(super) config: SpeculativeConfig,
    /// The prompt, then every emitted token.
    pub(super) tokens: Vec<u32>,
    pub(super) prompt_len: usize,
    pub(super) stats: SpeculativeStats,
    pub(super) phase: Phase,
    /// Reused buffer of the tokens one step feeds.
    pub(super) feed: Vec<u32>,
    /// The Philox key of the sampled path, fixed at start.
    pub(super) key: u64,
    /// Steps run since start. It picks each step's Philox counter.
    pub(super) step_index: u64,
}

impl<M: SpeculativeModel, D: Drafter> SpeculativeGenerator<M, D> {
    /// A generator over `model` and `drafter`. Call [`start`](Self::start) next.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the model has an empty vocabulary or
    /// [`SpeculativeConfig::check`] refuses `config` for it.
    pub fn new(model: M, drafter: D, config: SpeculativeConfig) -> Result<Self> {
        let vocab_size = model.vocab_size();
        if vocab_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "model",
                reason: "the model reports a vocabulary of 0 tokens".into(),
            });
        }
        config.check(vocab_size)?;
        Ok(Self {
            model,
            drafter,
            feed: Vec::with_capacity(config.num_draft_tokens + 1),
            config,
            tokens: Vec::new(),
            prompt_len: 0,
            stats: SpeculativeStats::default(),
            phase: Phase::Idle,
            key: 0,
            step_index: 0,
        })
    }

    /// Reset the model cache and prefill every prompt token except the last.
    ///
    /// The stats restart from zero. A one-token prompt needs no forward. On
    /// the sampled path the Philox key is fixed here. The step count
    /// restarts from zero.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an empty prompt or a prompt id outside
    /// the vocabulary. A model error from the reset or the prefill returns
    /// as is. After any error, steps are refused until a `start` succeeds.
    pub fn start(&mut self, prompt: &[u32]) -> Result<()> {
        self.phase = Phase::Idle;
        if prompt.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "prompt",
                reason: "the prompt has no tokens; pass at least one".into(),
            });
        }
        let vocab_size = self.model.vocab_size();
        if let Some((i, &id)) = prompt
            .iter()
            .enumerate()
            .find(|&(_, &id)| id as usize >= vocab_size)
        {
            return Err(Error::InvalidArgument {
                arg: "prompt",
                reason: format!(
                    "token {i} is id {id}, outside the model vocabulary of {vocab_size} tokens"
                ),
            });
        }
        self.model.truncate(0)?;
        let prefill = &prompt[..prompt.len() - 1];
        if !prefill.is_empty() {
            let ids = self.model.extend_greedy(prefill)?;
            check_extend(prefill.len(), ids.len(), 0, self.model.len())?;
        }
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.prompt_len = prompt.len();
        self.stats = SpeculativeStats::default();
        self.step_index = 0;
        if let Some(rule) = self.config.sampled_rule().filter(|rule| !rule.is_greedy()) {
            self.key = rule.seed.unwrap_or_else(numr::random::next_call_seed);
        }
        self.phase = Phase::Running;
        Ok(())
    }

    /// Start from `prompt` and step until the output ends.
    ///
    /// Returns the emitted tokens without the prompt, why they ended, and the
    /// stats.
    ///
    /// # Errors
    ///
    /// Every error of [`start`](Self::start) and [`step`](Self::step).
    pub fn generate(
        &mut self,
        prompt: &[u32],
    ) -> Result<(Vec<u32>, FinishReason, SpeculativeStats)> {
        self.start(prompt)?;
        let mut output = Vec::new();
        loop {
            let StepOutput { tokens, finish } = self.step()?;
            output.extend_from_slice(&tokens);
            if let Some(reason) = finish {
                return Ok((output, reason, self.stats));
            }
        }
    }

    /// The prompt, then every emitted token.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Every emitted token, without the prompt.
    pub fn generated(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    /// Why the output ended, or `None` while it runs.
    pub fn finish(&self) -> Option<FinishReason> {
        match self.phase {
            Phase::Finished(reason) => Some(reason),
            Phase::Idle | Phase::Running => None,
        }
    }

    /// Counters since the last [`start`](Self::start).
    pub fn stats(&self) -> SpeculativeStats {
        self.stats
    }

    /// The config the generator runs with.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// The target model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The drafter.
    pub fn drafter(&self) -> &D {
        &self.drafter
    }

    /// The model and the drafter, for reuse in another generator.
    pub fn into_parts(self) -> (M, D) {
        (self.model, self.drafter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::draft::PromptLookup;
    use crate::inference::speculative::toy_lm::{
        EmptyDrafter, OffByOne, Oracle, RandomDrafter, ToyLm, plain_greedy,
    };
    use crate::inference::speculative::types::MAX_NUM_DRAFT_TOKENS;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn config(k: usize, max_tokens: usize, eos: Vec<u32>) -> SpeculativeConfig {
        SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens,
            eos_token_ids: eos,
            sampling: None,
        }
    }

    fn arg_of(err: Error) -> &'static str {
        match err {
            Error::InvalidArgument { arg, .. } => arg,
            other => panic!("expected InvalidArgument, got {other}"),
        }
    }

    /// One generation setup and its plain greedy result.
    struct Case<'a> {
        vocab: usize,
        seed: u64,
        prompt: &'a [u32],
        config: &'a SpeculativeConfig,
        want: &'a [u32],
        finish: FinishReason,
    }

    impl Case<'_> {
        fn what(&self, drafter: &str) -> String {
            format!(
                "drafter {drafter}: vocab {} seed {} prompt {:?} config {:?}",
                self.vocab, self.seed, self.prompt, self.config
            )
        }

        /// Run `drafter` speculatively and check it against the plain result.
        fn check<D: Drafter>(&self, name: &str, drafter: D) -> SpeculativeStats {
            let what = self.what(name);
            let model = ToyLm::new(self.vocab, self.seed);
            let mut g = SpeculativeGenerator::new(model, drafter, self.config.clone())
                .expect("valid config");
            let (out, finish, stats) = g.generate(self.prompt).expect("generate");
            assert_eq!(out, self.want, "{what}: tokens");
            assert_eq!(finish, self.finish, "{what}: finish reason");
            assert_eq!(g.generated(), out.as_slice(), "{what}: generated");
            assert_eq!(stats.emitted, out.len(), "{what}: emitted");
            assert_eq!(
                stats.steps, stats.target_forwards,
                "{what}: one forward per step"
            );
            assert!(stats.target_forwards <= stats.emitted, "{what}: {stats:?}");
            assert!(stats.accepted <= stats.drafted, "{what}: {stats:?}");
            let tokens = g.tokens();
            assert_eq!(
                &tokens[..self.prompt.len()],
                self.prompt,
                "{what}: prompt kept"
            );
            assert_eq!(
                g.model().fed.as_slice(),
                &tokens[..tokens.len() - 1],
                "{what}: the cache holds every committed token except the last"
            );
            assert!(g.step().is_err(), "{what}: a step after the end is refused");
            stats
        }
    }

    /// Speculative output equals plain greedy output for every drafter, over
    /// seeded random models, prompts, draft counts, limits and EOS sets.
    #[test]
    fn speculative_output_equals_plain_greedy() {
        const CASES: usize = 320;
        let mut rng = Lcg(0x5bec_5eed);
        let mut runs = 0usize;
        let mut lookup_accepted = 0usize;
        let mut eos_endings = 0usize;
        let mut oracle_speedups = 0usize;
        for case in 0..CASES {
            let vocab = 8 + rng.below(293);
            let seed = rng.next();
            let prompt_len = 1 + rng.below(40);
            let prompt: Vec<u32> = if rng.below(3) == 0 {
                let period = 1 + rng.below(4);
                let pattern: Vec<u32> = (0..period).map(|_| rng.below(vocab) as u32).collect();
                (0..prompt_len).map(|i| pattern[i % period]).collect()
            } else {
                (0..prompt_len).map(|_| rng.below(vocab) as u32).collect()
            };
            let k = 1 + rng.below(8);
            let max_tokens = 1 + rng.below(60);

            let free_config = config(k, max_tokens, Vec::new());
            let (free, _) = plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &free_config)
                .expect("plain run");
            let emitted_id = free[rng.below(free.len())];
            let never: Vec<u32> = (0..vocab as u32)
                .filter(|t| !free.contains(t))
                .take(2)
                .collect();
            let mut eos_sets = vec![Vec::new(), vec![emitted_id]];
            if !never.is_empty() {
                eos_sets.push(never);
            }

            for eos in eos_sets {
                let run_config = config(k, max_tokens, eos);
                let (want, finish) =
                    plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &run_config)
                        .expect("plain run");
                if finish == FinishReason::Eos {
                    eos_endings += 1;
                }
                let c = Case {
                    vocab,
                    seed,
                    prompt: &prompt,
                    config: &run_config,
                    want: &want,
                    finish,
                };

                lookup_accepted += c.check("prompt-lookup", PromptLookup::default()).accepted;

                let oracle = c.check("oracle", Oracle { seed, vocab });
                let what = c.what("oracle");
                assert_eq!(oracle.accepted, oracle.drafted, "{what}: {oracle:?}");
                assert!(
                    oracle.target_forwards <= oracle.emitted.div_ceil(k + 1) + 1,
                    "{what}: {oracle:?}"
                );
                if k >= 2 && oracle.emitted >= k {
                    assert!(
                        oracle.target_forwards < oracle.emitted,
                        "{what}: {oracle:?}"
                    );
                    oracle_speedups += 1;
                }

                let random = RandomDrafter {
                    state: seed ^ case as u64,
                    vocab,
                };
                c.check("random", random);
                c.check("off-by-one", OffByOne { seed, vocab });

                let empty = c.check("empty", EmptyDrafter);
                let what = c.what("empty");
                assert_eq!(empty.target_forwards, empty.emitted, "{what}: {empty:?}");
                assert_eq!(empty.drafted, 0, "{what}: {empty:?}");
                runs += 5;
            }
        }
        assert!(runs >= CASES * 2 * 5, "{runs} runs");
        assert!(
            lookup_accepted > 0,
            "prompt lookup never matched; the test is vacuous"
        );
        assert!(eos_endings > 0, "no run ended at EOS; the test is vacuous");
        assert!(oracle_speedups > 0, "no oracle run saved a forward");
    }

    #[test]
    fn empty_prompt_is_refused() {
        let mut g =
            SpeculativeGenerator::new(ToyLm::new(10, 1), EmptyDrafter, config(2, 4, vec![]))
                .expect("valid");
        assert_eq!(arg_of(g.start(&[]).expect_err("empty")), "prompt");
        assert_eq!(arg_of(g.generate(&[]).expect_err("empty")), "prompt");
    }

    #[test]
    fn prompt_id_outside_the_vocab_is_refused() {
        let mut g =
            SpeculativeGenerator::new(ToyLm::new(10, 1), EmptyDrafter, config(2, 4, vec![]))
                .expect("valid");
        let err = g.start(&[1, 10]).expect_err("10 is outside");
        assert!(err.to_string().contains("token 1 is id 10"), "{err}");
    }

    #[test]
    fn one_token_prompt_needs_no_prefill() {
        let (vocab, seed) = (30, 9);
        let config = config(3, 10, vec![]);
        let mut g = SpeculativeGenerator::new(
            ToyLm::new(vocab, seed),
            Oracle { seed, vocab },
            config.clone(),
        )
        .expect("valid");
        g.start(&[4]).expect("start");
        assert!(g.model().is_empty());
        let (out, finish, _) = g.generate(&[4]).expect("generate");
        let (want, want_finish) =
            plain_greedy(&mut ToyLm::new(vocab, seed), &[4], &config).expect("plain");
        assert_eq!((out, finish), (want, want_finish));
    }

    #[test]
    fn invalid_configs_are_refused() {
        let new =
            |c: SpeculativeConfig| SpeculativeGenerator::new(ToyLm::new(10, 1), EmptyDrafter, c);
        assert_eq!(
            arg_of(new(config(0, 4, vec![])).err().expect("k = 0")),
            "num_draft_tokens"
        );
        assert_eq!(
            arg_of(
                new(config(MAX_NUM_DRAFT_TOKENS + 1, 4, vec![]))
                    .err()
                    .expect("k = 33")
            ),
            "num_draft_tokens"
        );
        assert_eq!(
            arg_of(new(config(2, 4, vec![10])).err().expect("eos 10")),
            "eos_token_ids"
        );
        assert_eq!(
            arg_of(new(config(2, 0, vec![])).err().expect("max 0")),
            "max_tokens"
        );
        let empty_vocab =
            SpeculativeGenerator::new(ToyLm::new(0, 1), EmptyDrafter, config(2, 4, vec![]));
        assert_eq!(arg_of(empty_vocab.err().expect("vocab 0")), "model");
    }

    #[test]
    fn start_resets_the_cache_and_the_stats() {
        let (vocab, seed) = (40, 3);
        let mut g = SpeculativeGenerator::new(
            ToyLm::new(vocab, seed),
            PromptLookup::default(),
            config(4, 20, vec![]),
        )
        .expect("valid");
        let first = g.generate(&[1, 2, 3, 1, 2]).expect("first");
        let second = g.generate(&[1, 2, 3, 1, 2]).expect("second");
        assert_eq!(first, second);
        let other = g.generate(&[7, 7]).expect("other prompt");
        let (want, _) =
            plain_greedy(&mut ToyLm::new(vocab, seed), &[7, 7], g.config()).expect("plain");
        assert_eq!(other.0, want);
        let (model, _) = g.into_parts();
        assert_eq!(model.fed.len(), 2 + want.len() - 1);
    }
}
