//! A drafter that samples from a small draft model.

use super::prob_drafter::{DraftModel, ProbDraft, ProbDrafter};
use crate::error::{Error, Result};
use crate::inference::speculative::client::SpeculativeClient;
use crate::inference::speculative::executor::accept::check_growth;
use crate::inference::speculative::penalty::penalized_logits;
use crate::inference::speculative::types::{Draw, SampleRule};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::marker::PhantomData;

/// XORed into the step's Philox key to give the draft stream.
///
/// The draft draws and the verify uniforms of one step then come from
/// distinct Philox keys at the same counters, so they are independent.
pub const DRAFT_KEY_DOMAIN: u64 = 0x6472_6166_745f_7131;

/// Drafts by running a draft model one token at a time.
///
/// The drafter mirrors the tokens its model cache holds. Each proposal
/// first reconciles that cache with the context. It keeps the common prefix
/// of the cache and the context without its last token, truncates the rest,
/// and feeds the missing tokens together with the last one in one forward.
/// So a rejected draft never stays in the cache.
///
/// Then it drafts `k` tokens. Draft `i` comes from the logits row after
/// `context ++ drafts[..i]`, with the penalties of the rule over that prefix.
/// A sampled rule draws it from `q_i`, the canonical filter of `rule.filter`.
/// That is the filter the target's `p_i` runs. A greedy rule takes the
/// argmax, lowest id on ties. The last draft is not fed, so `k` drafts cost
/// `k` draft forwards.
///
/// The draw of draft `i` is uniform `i` of `philox_uniform` at key
/// `draw.key ^ DRAFT_KEY_DOMAIN` and counter `draw.counter`. A draft count up
/// to [`MAX_NUM_DRAFT_TOKENS`](crate::inference::speculative::MAX_NUM_DRAFT_TOKENS)
/// keeps every step on its own counters.
pub struct ModelDrafter<R, M, C> {
    model: M,
    client: C,
    /// The tokens the draft cache holds, in order.
    fed: Vec<u32>,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, M, C> ModelDrafter<R, M, C>
where
    R: Runtime<DType = DType>,
    M: DraftModel<R>,
    C: SpeculativeClient<R>,
{
    /// A drafter over `model`, running its device ops on `client`.
    ///
    /// The first proposal truncates the model cache to 0 and prefills it.
    pub fn new(model: M, client: C) -> Self {
        Self {
            model,
            client,
            fed: Vec::new(),
            _runtime: PhantomData,
        }
    }

    /// The draft model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The tokens the draft cache holds.
    pub fn fed(&self) -> &[u32] {
        &self.fed
    }

    /// The draft model and the client.
    pub fn into_parts(self) -> (M, C) {
        (self.model, self.client)
    }

    /// Bring the cache to `context[..len - 1]`, feed the last token, return its logits row.
    fn reconcile(&mut self, context: &[u32]) -> Result<Tensor<R>> {
        // After an error the mirror and the cache can differ in length; keep the shorter.
        let held = self.fed.len().min(self.model.len());
        let cached = &context[..context.len().saturating_sub(1)];
        let common = self.fed[..held]
            .iter()
            .zip(cached)
            .take_while(|(a, b)| a == b)
            .count();
        self.model.truncate(common)?;
        self.fed.truncate(common);
        if self.model.len() != common {
            return Err(Error::InferenceError {
                reason: format!(
                    "the draft cache holds {} positions after a truncate to {common}",
                    self.model.len()
                ),
            });
        }
        self.feed(&context[common..])
    }

    /// Feed `tokens` and return the `[1, 1, V]` logits row of the last one.
    fn feed(&mut self, tokens: &[u32]) -> Result<Tensor<R>> {
        let Some(last) = tokens.len().checked_sub(1) else {
            return Err(Error::InferenceError {
                reason: "a draft forward needs at least one token".into(),
            });
        };
        let before = self.model.len();
        let logits = self.model.forward_logits(tokens)?;
        check_growth(tokens.len(), before, self.model.len())?;
        self.fed.extend_from_slice(tokens);
        let shape = logits.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != tokens.len() || shape[2] == 0 {
            return Err(Error::InferenceError {
                reason: format!(
                    "forward_logits returned {shape:?} for {} tokens; expected [1, {}, vocab]",
                    tokens.len(),
                    tokens.len()
                ),
            });
        }
        Ok(logits.narrow(1, last, 1)?.contiguous()?)
    }

    /// The draft after `prefix` from its logits `row`, and its distribution when sampled.
    ///
    /// `uniform` is `[1, 1]` for a sampled rule and `None` for a greedy one.
    fn pick(
        &self,
        row: &Tensor<R>,
        prefix: &[u32],
        rule: Option<&SampleRule>,
        uniform: Option<&Tensor<R>>,
    ) -> Result<(u32, Option<Tensor<R>>)> {
        let vocab = row.shape().last().copied().unwrap_or(0);
        let penalized = match rule.filter(|r| r.has_penalties()) {
            Some(r) => Some(penalized_logits(&self.client, row, prefix, &[], r, vocab)?),
            None => None,
        };
        let scores = penalized.as_ref().unwrap_or(row);
        let (Some(rule), Some(uniform)) = (rule.filter(|r| !r.is_greedy()), uniform) else {
            // [1, 1, V] -> [1, 1] id on the device; one id crosses to the host.
            let ids = self
                .client
                .argmax(scores, 2, false)?
                .contiguous()?
                .try_to_vec::<i64>()?;
            return Ok((host_id(ids.first().copied(), vocab, "argmax")?, None));
        };
        let q = self.client.sampling_probs(scores, &rule.filter)?;
        // speculative_verify with no draft draws one token from row 0 by its uniform.
        let no_drafts = Tensor::<R>::from_slice(&[] as &[i32], &[1, 0], row.device())?;
        let pair = self
            .client
            .speculative_verify(&q, None, &no_drafts, uniform)?
            .contiguous()?
            .try_to_vec::<i32>()?;
        let next = pair.get(1).map(|&t| i64::from(t));
        Ok((host_id(next, vocab, "the draft draw")?, Some(q)))
    }
}

/// `id` as a token below `vocab`.
fn host_id(id: Option<i64>, vocab: usize, source: &str) -> Result<u32> {
    id.and_then(|id| u32::try_from(id).ok())
        .filter(|&t| (t as usize) < vocab)
        .ok_or_else(|| Error::InferenceError {
            reason: format!(
                "{source} returned {id:?}; expected one id below the draft vocabulary of {vocab}"
            ),
        })
}

impl<R, M, C> ProbDrafter<R> for ModelDrafter<R, M, C>
where
    R: Runtime<DType = DType>,
    M: DraftModel<R>,
    C: SpeculativeClient<R>,
{
    fn propose(
        &mut self,
        context: &[u32],
        k: usize,
        rule: Option<&SampleRule>,
        draw: Draw,
    ) -> Result<ProbDraft<R>> {
        if k == 0 || context.is_empty() {
            return Ok(ProbDraft::empty());
        }
        if let Some(rule) = rule {
            rule.check()?;
        }
        let sampled = rule.is_some_and(|r| !r.is_greedy());
        let uniforms = if sampled {
            let key = draw.key ^ DRAFT_KEY_DOMAIN;
            Some(
                self.client
                    .philox_uniform(&[1, k], key, draw.counter, DType::F32)?,
            )
        } else {
            None
        };

        let mut row = self.reconcile(context)?;
        let mut prefix = Vec::with_capacity(context.len() + k);
        prefix.extend_from_slice(context);
        let mut tokens = Vec::with_capacity(k);
        let mut rows = Vec::with_capacity(if sampled { k } else { 0 });
        for i in 0..k {
            if let Some(&prev) = tokens.last() {
                row = self.feed(&[prev])?;
            }
            let uniform = match &uniforms {
                Some(u) => Some(u.narrow(1, i, 1)?.contiguous()?),
                None => None,
            };
            let (id, q) = self.pick(&row, &prefix, rule, uniform.as_ref())?;
            tokens.push(id);
            prefix.push(id);
            rows.extend(q);
        }
        let probs = if sampled {
            let refs: Vec<&Tensor<R>> = rows.iter().collect();
            Some(self.client.cat(&refs, 1)?)
        } else {
            None
        };
        Ok(ProbDraft { tokens, probs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::executor::generate::SpeculativeGenerator;
    use crate::inference::speculative::model::{SpeculativeModel, SpeculativeProbModel};
    use crate::inference::speculative::toy_lm::{
        ToyLm, plain_greedy, toy_continuation, toy_drafter,
    };
    use crate::inference::speculative::types::{
        FinishReason, SpeculativeConfig, SpeculativeStats, Verdict,
    };
    use crate::ops::traits::inference::FilterRule;
    use crate::test_utils::Lcg;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    type ToyDrafter = ModelDrafter<CpuRuntime, ToyLm, CpuClient>;

    fn config(
        k: usize,
        max_tokens: usize,
        eos: Vec<u32>,
        sampling: Option<SampleRule>,
    ) -> SpeculativeConfig {
        SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens,
            eos_token_ids: eos,
            sampling,
        }
    }

    fn seeded(rule: SampleRule, seed: u64) -> SampleRule {
        SampleRule {
            seed: Some(seed),
            ..rule
        }
    }

    fn open_rule() -> SampleRule {
        SampleRule::new(FilterRule::temperature_only(1.0))
    }

    fn assert_mirror(d: &ToyDrafter, what: &str) {
        assert_eq!(
            d.fed(),
            d.model().fed.as_slice(),
            "{what}: the mirror equals the draft cache"
        );
    }

    /// One run over a toy target, step by step. Checks both caches after every step.
    fn run(
        vocab: usize,
        target_seed: u64,
        drafter: ToyDrafter,
        config: &SpeculativeConfig,
        prompt: &[u32],
    ) -> (Vec<u32>, FinishReason, SpeculativeStats) {
        let target = ToyLm::new(vocab, target_seed);
        let mut g = SpeculativeGenerator::new(target, drafter, config.clone()).expect("valid");
        g.start(prompt).expect("start");
        while g.finish().is_none() {
            g.step().expect("step");
            let tokens = g.tokens();
            assert_eq!(
                g.model().fed.as_slice(),
                &tokens[..tokens.len() - 1],
                "the target cache holds every committed token except the last"
            );
            assert_mirror(g.drafter(), "after a step");
        }
        let finish = g.finish().expect("finished");
        (g.generated().to_vec(), finish, g.stats())
    }

    /// A random case: vocab, target seed, prompt, draft count, token limit.
    fn random_case(rng: &mut Lcg) -> (usize, u64, Vec<u32>, usize, usize) {
        let vocab = 4 + rng.below(60);
        let seed = rng.below(1 << 30) as u64;
        let prompt = (0..1 + rng.below(12))
            .map(|_| rng.below(vocab) as u32)
            .collect();
        (vocab, seed, prompt, 1 + rng.below(6), 1 + rng.below(30))
    }

    #[test]
    fn a_draft_equal_to_the_target_matches_plain_greedy_and_accepts_every_draft() {
        let mut rng = Lcg(0xd2af_7001);
        let mut drafted = 0;
        for case in 0..60 {
            let (vocab, seed, prompt, k, max_tokens) = random_case(&mut rng);
            let cfg = config(k, max_tokens, vec![], None);
            let want = plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &cfg).expect("plain");
            let what = format!("case {case} vocab {vocab} seed {seed} prompt {prompt:?} k {k}");
            let (out, finish, stats) = run(vocab, seed, toy_drafter(vocab, seed), &cfg, &prompt);
            assert_eq!((out, finish), want, "{what}");
            assert_eq!(stats.accepted, stats.drafted, "{what}: {stats:?}");
            drafted += stats.drafted;
        }
        assert!(drafted > 0, "no case drafted; the test is vacuous");
    }

    #[test]
    fn a_draft_equal_to_the_target_accepts_every_sampled_draft() {
        let filtered = SampleRule {
            repeat_penalty: 1.2,
            presence_penalty: 0.3,
            penalty_last_n: 4,
            ..SampleRule::new(FilterRule {
                temperature: 0.8,
                top_k: 5,
                top_p: 0.9,
                min_p: 0.02,
            })
        };
        let mut rng = Lcg(0xd2af_7002);
        for (name, rule) in [("open", open_rule()), ("filtered", filtered)] {
            let mut drafted = 0;
            for s in 0..40u64 {
                let (vocab, seed, prompt, k, _) = random_case(&mut rng);
                let cfg = config(k, 24, vec![], Some(seeded(rule, s)));
                let what = format!("{name} run {s} vocab {vocab} seed {seed} k {k}");
                let (out, _, stats) = run(vocab, seed, toy_drafter(vocab, seed), &cfg, &prompt);
                assert_eq!(out.len(), 24, "{what}");
                assert_eq!(
                    stats.accepted, stats.drafted,
                    "{what}: q equals p: {stats:?}"
                );
                drafted += stats.drafted;
            }
            assert!(drafted > 0, "{name}: no run drafted; the test is vacuous");
        }
    }

    #[test]
    fn a_different_draft_still_matches_plain_greedy() {
        let mut rng = Lcg(0xd2af_7003);
        let (mut drafted, mut accepted) = (0, 0);
        for case in 0..60 {
            let (vocab, seed, prompt, k, max_tokens) = random_case(&mut rng);
            let free = plain_greedy(
                &mut ToyLm::new(vocab, seed),
                &prompt,
                &config(k, max_tokens, vec![], None),
            )
            .expect("plain");
            let eos = vec![free.0[rng.below(free.0.len())]];
            for cfg in [
                config(k, max_tokens, vec![], None),
                config(k, max_tokens, eos, None),
            ] {
                let want =
                    plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &cfg).expect("plain");
                let drafter = toy_drafter(vocab, seed ^ 0x0d2a_f75e);
                let what = format!("case {case} vocab {vocab} seed {seed} config {cfg:?}");
                let (out, finish, stats) = run(vocab, seed, drafter, &cfg, &prompt);
                assert_eq!((out, finish), want, "{what}");
                drafted += stats.drafted;
                accepted += stats.accepted;
            }
        }
        assert!(
            0 < accepted && accepted < drafted,
            "{accepted} of {drafted} drafts accepted; both branches must run"
        );
    }

    #[test]
    fn sampled_runs_with_a_different_draft_keep_both_caches_consistent() {
        let mut rng = Lcg(0xd2af_7005);
        let (mut drafted, mut accepted) = (0, 0);
        for s in 0..40u64 {
            let (vocab, seed, prompt, k, _) = random_case(&mut rng);
            let cfg = config(k, 40, vec![], Some(seeded(open_rule(), s)));
            let drafter = toy_drafter(vocab, seed ^ 0x0d2a_f75e);
            let (out, finish, stats) = run(vocab, seed, drafter, &cfg, &prompt);
            assert_eq!((out.len(), finish), (40, FinishReason::Length));
            drafted += stats.drafted;
            accepted += stats.accepted;
        }
        assert!(
            0 < accepted && accepted < drafted,
            "{accepted} of {drafted} drafts accepted; both branches must run"
        );
    }

    #[test]
    fn reconcile_after_any_divergence_equals_a_fresh_feed() {
        let (vocab, seed) = (12, 77);
        let rule = open_rule();
        let mut stale = toy_drafter(vocab, seed);
        let mut rng = Lcg(5);
        let mut context: Vec<u32> = vec![1, 2, 3];
        for step in 0..200u64 {
            // Keep a random prefix and append random tokens, so the cache diverges anywhere.
            context.truncate(1 + rng.below(context.len()));
            for _ in 0..1 + rng.below(4) {
                context.push(rng.below(vocab) as u32);
            }
            let k = 1 + rng.below(5);
            let draw = Draw::for_step(9, step);
            let r = (step % 3 != 0).then_some(&rule);
            let got = stale.propose(&context, k, r, draw).expect("stale");
            let mut fresh = toy_drafter(vocab, seed);
            let want = fresh.propose(&context, k, r, draw).expect("fresh");
            let what = format!("step {step} context {context:?} k {k}");
            assert_eq!(got.tokens, want.tokens, "{what}");
            let host = |p: &Option<Tensor<CpuRuntime>>| p.as_ref().map(|t| t.to_vec::<f32>());
            assert_eq!(host(&got.probs), host(&want.probs), "{what}: probs");
            if let Some(q) = &got.probs {
                assert_eq!(q.shape(), [1, k, vocab], "{what}");
            }
            assert_mirror(&stale, &what);
            assert_eq!(stale.fed(), fresh.fed(), "{what}");
            assert_eq!(stale.model().len(), context.len() + k - 1, "{what}");
            assert_eq!(&stale.fed()[..context.len()], context.as_slice(), "{what}");
            // Adopt a random number of drafts, as an accept does.
            context.extend_from_slice(&got.tokens[..rng.below(k + 1)]);
        }
    }

    #[test]
    fn greedy_proposals_are_the_draft_argmax_with_no_probs() {
        let mut d = toy_drafter(20, 5);
        let ctx = [1, 2, 3];
        let p = d
            .propose(&ctx, 4, None, Draw::for_step(0, 0))
            .expect("propose");
        let want = toy_continuation(5, 20, &ctx, 4);
        assert_eq!(p.tokens, want);
        assert!(p.probs.is_none());
        let mut fed = ctx.to_vec();
        fed.extend_from_slice(&want[..3]);
        assert_eq!(d.fed(), fed.as_slice(), "the last draft is not fed");
        assert!(
            d.propose(&ctx, 0, None, Draw::for_step(0, 1))
                .expect("k 0")
                .tokens
                .is_empty()
        );
    }

    #[test]
    fn eos_inside_the_drafts_ends_the_output_on_a_small_vocab() {
        let (vocab, eos) = (3, 2u32);
        let prompt = [0u32, 1];
        let mut cut_drafts = 0;
        for seed in 0..40u64 {
            let cfg = config(6, 30, vec![eos], None);
            let want = plain_greedy(&mut ToyLm::new(vocab, seed), &prompt, &cfg).expect("plain");
            for draft_seed in [seed, seed ^ 0xff] {
                let (out, finish, _) =
                    run(vocab, seed, toy_drafter(vocab, draft_seed), &cfg, &prompt);
                assert_eq!((out, finish), want, "greedy seed {seed} draft {draft_seed}");
            }
            let cfg = config(6, 30, vec![eos], Some(seeded(open_rule(), seed)));
            for draft_seed in [seed, seed ^ 0xff] {
                let drafter = toy_drafter(vocab, draft_seed);
                let (out, finish, stats) = run(vocab, seed, drafter, &cfg, &prompt);
                let first_eos = out.iter().position(|&t| t == eos);
                match finish {
                    FinishReason::Eos => assert_eq!(first_eos, Some(out.len() - 1), "{out:?}"),
                    FinishReason::Length => assert_eq!(first_eos, None, "{out:?}"),
                }
                // k never exceeds the remaining budget, so only an EOS cuts accepted drafts.
                if stats.emitted < stats.accepted + stats.steps {
                    cut_drafts += 1;
                }
            }
        }
        assert!(
            cut_drafts > 0,
            "no EOS fell inside accepted drafts; the test is vacuous"
        );
    }

    /// Keeps the default `extend_sampled_with_probs`, which refuses draft probs.
    struct NoProbs(ToyLm);

    impl SpeculativeModel for NoProbs {
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
        fn extend_sampled(
            &mut self,
            tokens: &[u32],
            rule: &SampleRule,
            draw: Draw,
        ) -> Result<Verdict> {
            self.0.extend_sampled(tokens, rule, draw)
        }
    }

    impl SpeculativeProbModel<CpuRuntime> for NoProbs {}

    #[test]
    fn a_target_without_draft_probs_refuses_a_sampled_draft_model() {
        let sampled = config(3, 10, vec![], Some(seeded(open_rule(), 1)));
        let mut g =
            SpeculativeGenerator::new(NoProbs(ToyLm::new(16, 3)), toy_drafter(16, 4), sampled)
                .expect("valid");
        let err = g.generate(&[1, 2]).expect_err("draft probs refused");
        assert!(
            err.to_string().contains("extend_sampled_with_probs"),
            "{err}"
        );
        let greedy = config(3, 10, vec![], None);
        let want = plain_greedy(&mut ToyLm::new(16, 3), &[1, 2], &greedy).expect("plain");
        let mut g =
            SpeculativeGenerator::new(NoProbs(ToyLm::new(16, 3)), toy_drafter(16, 4), greedy)
                .expect("valid");
        let (out, finish, _) = g.generate(&[1, 2]).expect("greedy needs no probs");
        assert_eq!((out, finish), want);
    }
}
