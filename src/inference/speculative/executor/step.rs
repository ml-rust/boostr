//! One speculative step: draft, one verify forward, accept, roll back.

use super::generate::{Phase, SpeculativeGenerator};
use crate::error::{Error, Result};
use crate::inference::speculative::draft::Drafter;
use crate::inference::speculative::model::SpeculativeModel;
use crate::inference::speculative::types::FinishReason;

/// The tokens one step emits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutput {
    /// One or more new tokens, in order. An EOS token ends the list.
    pub tokens: Vec<u32>,
    /// Set on the step that ends the output. No step follows it.
    pub finish: Option<FinishReason>,
}

/// One `extend_greedy` call over `fed` tokens returned `got` ids and moved the
/// cache from `len_before` to `len_after` positions.
///
/// # Errors
///
/// [`Error::InferenceError`] when the id count or the new cache length breaks
/// the [`SpeculativeModel`] contract.
pub(super) fn check_extend(
    fed: usize,
    got: usize,
    len_before: usize,
    len_after: usize,
) -> Result<()> {
    if got != fed {
        return Err(Error::InferenceError {
            reason: format!(
                "extend_greedy returned {got} ids for {fed} fed tokens; the model must return \
                 one id per fed token"
            ),
        });
    }
    if len_after != len_before + fed {
        return Err(Error::InferenceError {
            reason: format!(
                "the model cache holds {len_after} positions after {fed} tokens fed at \
                 {len_before}; it must hold {}",
                len_before + fed
            ),
        });
    }
    Ok(())
}

impl<M: SpeculativeModel, D: Drafter> SpeculativeGenerator<M, D> {
    /// Run one step and return the tokens it emits.
    ///
    /// The drafter proposes up to `num_draft_tokens` tokens. The count is
    /// clamped so that the step never emits past `max_tokens`. The proposal
    /// is cut at its first id outside the vocabulary. One forward feeds the
    /// last committed token `x` and the drafts `d_1..d_k`. It returns the
    /// target's ids `a_0..a_k`. The step keeps the longest prefix with
    /// `d_{i+1} == a_i`, of length `m`. It emits `d_1..d_m` and then `a_m`.
    /// The output ends after the first EOS token or at `max_tokens`. With no
    /// drafts the step is one plain greedy decode.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] before [`start`](Self::start), after the
    /// output ends, or when the model breaks its contract. A model error
    /// returns as is. After any error, steps are refused until a `start`
    /// succeeds.
    pub fn step(&mut self) -> Result<StepOutput> {
        match self.phase {
            Phase::Running => {}
            Phase::Idle => {
                return Err(Error::InferenceError {
                    reason: "no generation runs; call start with a prompt first".into(),
                });
            }
            Phase::Finished(reason) => {
                return Err(Error::InferenceError {
                    reason: format!(
                        "the output already ended ({reason:?}); call start for a new prompt"
                    ),
                });
            }
        }
        let result = self.run_step();
        if result.is_err() {
            self.phase = Phase::Idle;
        }
        result
    }

    fn run_step(&mut self) -> Result<StepOutput> {
        let vocab_size = self.model.vocab_size();
        let committed = self.tokens.len();
        let last = *self.tokens.last().ok_or_else(|| Error::InferenceError {
            reason: "the generator holds no committed token; call start first".into(),
        })?;
        let len_before = self.model.len();
        if len_before + 1 != committed {
            return Err(Error::InferenceError {
                reason: format!(
                    "the model cache holds {len_before} positions; the generator expects {}, \
                     every committed token except the last",
                    committed - 1
                ),
            });
        }
        let emitted_before = committed - self.prompt_len;
        let remaining = self.config.max_tokens.saturating_sub(emitted_before);
        let k = self
            .config
            .num_draft_tokens
            .min(remaining.saturating_sub(1));

        self.feed.clear();
        self.feed.push(last);
        if k > 0 {
            let proposal = self.drafter.propose(&self.tokens, k);
            self.feed.extend(
                proposal
                    .iter()
                    .take(k)
                    .take_while(|&&id| (id as usize) < vocab_size),
            );
        }
        let drafts = &self.feed[1..];

        let ids = self.model.extend_greedy(&self.feed)?;
        check_extend(self.feed.len(), ids.len(), len_before, self.model.len())?;
        if let Some((j, &id)) = ids
            .iter()
            .enumerate()
            .find(|&(_, &id)| id as usize >= vocab_size)
        {
            return Err(Error::InferenceError {
                reason: format!(
                    "extend_greedy returned id {id} at row {j}, outside the model vocabulary \
                     of {vocab_size} tokens"
                ),
            });
        }

        let accepted = drafts.iter().zip(&ids).take_while(|(d, a)| d == a).count();
        let next = *ids.get(accepted).ok_or_else(|| Error::InferenceError {
            reason: format!("extend_greedy returned no id for row {accepted}"),
        })?;
        let mut tokens = Vec::with_capacity(accepted + 1);
        tokens.extend_from_slice(&drafts[..accepted]);
        tokens.push(next);

        let mut finish = None;
        if let Some(pos) = tokens.iter().position(|&t| self.config.is_eos(t)) {
            tokens.truncate(pos + 1);
            finish = Some(FinishReason::Eos);
        } else if emitted_before + tokens.len() >= self.config.max_tokens {
            tokens.truncate(remaining);
            finish = Some(FinishReason::Length);
        }

        let keep = len_before + tokens.len();
        self.model.truncate(keep)?;
        if self.model.len() != keep {
            return Err(Error::InferenceError {
                reason: format!(
                    "the model cache holds {} positions after a truncate to {keep}",
                    self.model.len()
                ),
            });
        }

        self.stats.steps += 1;
        self.stats.target_forwards += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += accepted;
        self.stats.emitted += tokens.len();
        self.tokens.extend_from_slice(&tokens);
        if let Some(reason) = finish {
            self.phase = Phase::Finished(reason);
        }
        Ok(StepOutput { tokens, finish })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::draft::PromptLookup;
    use crate::inference::speculative::toy_lm::{
        EmptyDrafter, Oracle, Scripted, ToyLm, toy_continuation,
    };
    use crate::inference::speculative::types::SpeculativeConfig;

    const VOCAB: usize = 50;
    const PROMPT: [u32; 5] = [3, 1, 4, 1, 5];

    /// A seed whose free continuation of `PROMPT` satisfies `want`, and that
    /// continuation, 12 tokens long.
    fn seed_where(want: impl Fn(&[u32]) -> bool) -> (u64, Vec<u32>) {
        (0..1000u64)
            .map(|seed| (seed, toy_continuation(seed, VOCAB, &PROMPT, 12)))
            .find(|(_, free)| want(free))
            .expect("some seed satisfies the condition")
    }

    fn config(k: usize, max_tokens: usize, eos: Vec<u32>) -> SpeculativeConfig {
        SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens,
            eos_token_ids: eos,
        }
    }

    fn started<D: Drafter>(
        seed: u64,
        drafter: D,
        config: SpeculativeConfig,
        prompt: &[u32],
    ) -> SpeculativeGenerator<ToyLm, D> {
        let mut g = SpeculativeGenerator::new(ToyLm::new(VOCAB, seed), drafter, config)
            .expect("valid config");
        g.start(prompt).expect("start");
        g
    }

    /// The cache holds every committed token except the last.
    fn assert_cache_invariant<D: Drafter>(g: &SpeculativeGenerator<ToyLm, D>) {
        let tokens = g.tokens();
        assert_eq!(g.model().fed.as_slice(), &tokens[..tokens.len() - 1]);
    }

    #[test]
    fn eos_inside_accepted_drafts_cuts_right_after_it() {
        let (seed, free) = seed_where(|f| !f[..2].contains(&f[2]));
        let eos = free[2];
        let mut g = started(
            seed,
            Oracle { seed, vocab: VOCAB },
            config(5, 20, vec![eos]),
            &PROMPT,
        );
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..3]);
        assert_eq!(out.finish, Some(FinishReason::Eos));
        assert_eq!(g.model().len(), PROMPT.len() + 2);
        assert_cache_invariant(&g);
        let stats = g.stats();
        assert_eq!((stats.drafted, stats.accepted, stats.emitted), (5, 5, 3));
        assert_eq!(g.finish(), Some(FinishReason::Eos));
        let err = g.step().expect_err("the output ended");
        assert!(err.to_string().contains("call start"), "{err}");
        assert_eq!(
            g.model().len(),
            PROMPT.len() + 2,
            "a refused step changes nothing"
        );
    }

    #[test]
    fn eos_as_the_correction_token() {
        let (seed, free) = seed_where(|f| !f[..2].contains(&f[2]));
        let eos = free[2];
        let wrong = (free[2] + 1) % VOCAB as u32;
        let drafter = Scripted(vec![free[0], free[1], wrong, free[3]]);
        let mut g = started(seed, drafter, config(5, 20, vec![eos]), &PROMPT);
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..3]);
        assert_eq!(out.finish, Some(FinishReason::Eos));
        assert_eq!((g.stats().drafted, g.stats().accepted), (4, 2));
        assert_cache_invariant(&g);
    }

    #[test]
    fn eos_as_the_bonus_token_after_every_draft_matches() {
        let (seed, free) = seed_where(|f| !f[..3].contains(&f[3]));
        let mut g = started(
            seed,
            Oracle { seed, vocab: VOCAB },
            config(3, 20, vec![free[3]]),
            &PROMPT,
        );
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..4]);
        assert_eq!(out.finish, Some(FinishReason::Eos));
        assert_eq!((g.stats().drafted, g.stats().accepted), (3, 3));
        assert_cache_invariant(&g);
    }

    #[test]
    fn max_tokens_clamps_the_drafts() {
        let (seed, free) = seed_where(|_| true);
        let mut g = started(
            seed,
            Oracle { seed, vocab: VOCAB },
            config(5, 8, vec![]),
            &PROMPT,
        );
        let first = g.step().expect("step 1");
        assert_eq!(first.tokens, free[..6]);
        assert_eq!(first.finish, None);
        let second = g.step().expect("step 2");
        assert_eq!(second.tokens, free[6..8]);
        assert_eq!(second.finish, Some(FinishReason::Length));
        assert_eq!(g.stats().drafted, 6, "the second step drafts 1");
        assert_cache_invariant(&g);

        let mut g = started(
            seed,
            Oracle { seed, vocab: VOCAB },
            config(5, 3, vec![]),
            &PROMPT,
        );
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..3]);
        assert_eq!(out.finish, Some(FinishReason::Length));
        assert_eq!(g.stats().drafted, 2);
    }

    #[test]
    fn no_drafts_is_one_plain_decode() {
        let (seed, free) = seed_where(|_| true);
        let mut g = started(seed, EmptyDrafter, config(4, 3, vec![]), &PROMPT);
        for i in 0..3 {
            let out = g.step().expect("step");
            assert_eq!(out.tokens, free[i..=i]);
            assert_cache_invariant(&g);
        }
        assert_eq!(g.finish(), Some(FinishReason::Length));
        assert_eq!((g.stats().drafted, g.stats().target_forwards), (0, 3));
    }

    #[test]
    fn draft_ids_outside_the_vocab_cut_the_proposal() {
        let (seed, free) = seed_where(|_| true);
        let drafter = Scripted(vec![free[0], VOCAB as u32 + 5, free[1]]);
        let mut g = started(seed, drafter, config(5, 20, vec![]), &PROMPT);
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..2]);
        assert_eq!((g.stats().drafted, g.stats().accepted), (1, 1));
        assert_cache_invariant(&g);
    }

    #[test]
    fn a_proposal_longer_than_k_is_cut_to_k() {
        let (seed, free) = seed_where(|_| true);
        let mut g = started(
            seed,
            Scripted(free[..10].to_vec()),
            config(3, 20, vec![]),
            &PROMPT,
        );
        let out = g.step().expect("step");
        assert_eq!(out.tokens, free[..4]);
        assert_eq!(g.stats().drafted, 3);
    }

    #[test]
    fn prompt_lookup_drafts_are_checked_like_any_other() {
        let (seed, free) = seed_where(|_| true);
        let mut g = started(
            seed,
            PromptLookup::default(),
            config(4, 12, vec![]),
            &PROMPT,
        );
        let mut out = Vec::new();
        while g.finish().is_none() {
            out.extend(g.step().expect("step").tokens);
            assert_cache_invariant(&g);
        }
        assert_eq!(out, free);
    }

    #[test]
    fn step_before_start_is_refused() {
        let mut g =
            SpeculativeGenerator::new(ToyLm::new(VOCAB, 1), EmptyDrafter, config(2, 4, vec![]))
                .expect("valid");
        let err = g.step().expect_err("not started");
        assert!(err.to_string().contains("call start"), "{err}");
    }

    /// Returns one id fewer than it was fed.
    struct ShortLm(ToyLm);

    impl SpeculativeModel for ShortLm {
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
            let mut ids = self.0.extend_greedy(tokens)?;
            ids.pop();
            Ok(ids)
        }
    }

    /// Feeds one token fewer than it was given.
    struct LossyLm(ToyLm);

    impl SpeculativeModel for LossyLm {
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
            let mut ids = self.0.extend_greedy(tokens)?;
            self.0.fed.pop();
            ids.truncate(tokens.len());
            Ok(ids)
        }
    }

    #[test]
    fn a_wrong_length_model_output_is_an_error() {
        let model = ShortLm(ToyLm::new(VOCAB, 7));
        let mut g =
            SpeculativeGenerator::new(model, EmptyDrafter, config(2, 4, vec![])).expect("valid");
        g.start(&[1]).expect("a one-token prompt needs no forward");
        let err = g.step().expect_err("short output");
        assert!(err.to_string().contains("0 ids for 1 fed tokens"), "{err}");
        let err = g.step().expect_err("refused after an error");
        assert!(err.to_string().contains("call start"), "{err}");
    }

    #[test]
    fn a_cache_that_does_not_grow_is_an_error() {
        let model = LossyLm(ToyLm::new(VOCAB, 7));
        let mut g =
            SpeculativeGenerator::new(model, EmptyDrafter, config(2, 4, vec![])).expect("valid");
        let err = g.start(&[1, 2, 3]).expect_err("prefill loses a position");
        assert!(err.to_string().contains("must hold 2"), "{err}");
        assert!(
            g.step().is_err(),
            "a failed start leaves no running generation"
        );
    }
}
