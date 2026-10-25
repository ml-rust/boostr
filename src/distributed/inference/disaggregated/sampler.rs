//! The per-request sampler prefill starts and decode resumes.
//!
//! Each token draws from `SamplingOps::logits_to_token` at Philox
//! `SampleSeed { key, counter: index }`, where `index` counts the request's
//! sampled tokens from 0. The draw is a pure function of the key, the index,
//! the logits and the context. So a request split across prefill and decode
//! samples the same tokens as one uninterrupted run with the same key.

use anyhow::{Result, anyhow, bail};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::protocol::{GenerationParams, TokenEvent};
use crate::inference::sampling::penalty_window;
use crate::ops::traits::inference::{SampleSeed, SamplingOps};

/// The sampler state of one request.
#[derive(Debug, Clone)]
pub struct RequestSampler {
    params: GenerationParams,
    key: u64,
    context: Vec<u32>,
    generated: u32,
}

impl RequestSampler {
    /// A sampler before the first token of `prompt`.
    ///
    /// `params.rule.seed == None` draws the key from `numr::random::next_call_seed`.
    /// [`Self::params`] then carries the resolved key.
    pub fn start(mut params: GenerationParams, prompt: &[u32]) -> Self {
        let key = *params
            .rule
            .seed
            .get_or_insert_with(numr::random::next_call_seed);
        Self {
            params,
            key,
            context: prompt.to_vec(),
            generated: 0,
        }
    }

    /// A sampler that continues after `generated` tokens, the tail of `context`.
    ///
    /// # Errors
    ///
    /// `params.rule.seed` is `None`, or `generated` is 0 or not below the context length.
    pub fn resume(params: GenerationParams, context: Vec<u32>, generated: u32) -> Result<Self> {
        let key = params.rule.seed.ok_or_else(|| {
            anyhow!("a resumed sampler needs the key prefill drew with, and the handoff has none")
        })?;
        if generated == 0 || generated as usize >= context.len() {
            bail!(
                "cannot resume after {generated} tokens in a context of {}",
                context.len()
            );
        }
        Ok(Self {
            params,
            key,
            context,
            generated,
        })
    }

    /// The params, with the seed resolved.
    pub fn params(&self) -> &GenerationParams {
        &self.params
    }

    /// The prompt, then every sampled token.
    pub fn context(&self) -> &[u32] {
        &self.context
    }

    /// Tokens sampled so far. It is also the index of the next token.
    pub fn generated(&self) -> u32 {
        self.generated
    }

    /// The token the next decode step feeds.
    pub fn last_token(&self) -> Option<u32> {
        self.context.last().copied()
    }

    /// The sequence position of [`Self::last_token`].
    pub fn last_position(&self) -> u32 {
        self.context.len().saturating_sub(1) as u32
    }

    /// Samples the next token from the last row of `logits` (`[1, seq_len, vocab]`).
    ///
    /// The penalties count the last `penalty_last_n` ids of the context, the
    /// prompt included. The result carries the finish reason the params give it.
    ///
    /// # Errors
    ///
    /// The sampler op's error, or a sampled id outside `u32`.
    pub fn sample<R>(&mut self, client: &R::Client, logits: &Tensor<R>) -> Result<TokenEvent>
    where
        R: Runtime<DType = DType>,
        R::Client: SamplingOps<R>,
    {
        let rule = &self.params.rule;
        let (ids, counts) = if rule.has_penalties() {
            penalty_window(&self.context, rule.penalty_last_n)
        } else {
            (Vec::new(), Vec::new())
        };
        let num_unique = ids.len();
        let device = logits.device();
        // The op reads no penalty entry when `num_unique == 0`, but it still
        // takes two tensors. One unused element stands in for them.
        let (ids, counts) = if num_unique == 0 {
            (vec![0i64], vec![0i32])
        } else {
            (ids, counts)
        };
        let ids_t = Tensor::<R>::from_slice(&ids, &[ids.len()], device)?;
        let counts_t = Tensor::<R>::from_slice(&counts, &[counts.len()], device)?;
        let out = client
            .logits_to_token(
                logits,
                &ids_t,
                &counts_t,
                num_unique,
                rule.repeat_penalty,
                rule.frequency_penalty,
                rule.presence_penalty,
                rule.filter.temperature,
                rule.filter.top_k,
                rule.filter.top_p,
                rule.filter.min_p,
                Some(SampleSeed::new(self.key, u64::from(self.generated))),
            )
            .map_err(|e| anyhow!("sampling token {} failed: {e}", self.generated))?;
        let raw = out
            .to_vec::<i64>()
            .first()
            .copied()
            .ok_or_else(|| anyhow!("the sampler returned no token"))?;
        let token = u32::try_from(raw)
            .map_err(|_| anyhow!("the sampler returned token id {raw}, outside u32"))?;
        let index = self.generated;
        self.context.push(token);
        self.generated += 1;
        Ok(TokenEvent {
            index,
            token,
            finish: self.params.finish_for(token, self.generated),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::SampleRule;
    use crate::ops::traits::inference::FilterRule;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const VOCAB: usize = 64;

    fn flat_logits(device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(&[0.0f32; VOCAB], &[1, 1, VOCAB], device).unwrap()
    }

    fn sampled(seed: Option<u64>, max_new_tokens: u32) -> GenerationParams {
        GenerationParams {
            rule: SampleRule {
                seed,
                ..SampleRule::new(FilterRule::temperature_only(1.0))
            },
            ..GenerationParams::greedy(max_new_tokens)
        }
    }

    fn run(s: &mut RequestSampler, steps: usize) -> Vec<u32> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (0..steps)
            .map(|_| {
                s.sample::<CpuRuntime>(&client, &flat_logits(&device))
                    .unwrap()
                    .token
            })
            .collect()
    }

    #[test]
    fn a_seed_fixes_the_stream_and_another_seed_changes_it() {
        let a = run(
            &mut RequestSampler::start(sampled(Some(7), 99), &[1, 2]),
            12,
        );
        let b = run(
            &mut RequestSampler::start(sampled(Some(7), 99), &[1, 2]),
            12,
        );
        let c = run(
            &mut RequestSampler::start(sampled(Some(8), 99), &[1, 2]),
            12,
        );
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn resume_continues_the_uninterrupted_stream() {
        let mut first = RequestSampler::start(sampled(None, 99), &[3]);
        assert!(
            first.params().rule.seed.is_some(),
            "start resolves the seed"
        );
        let key = first.params().rule.seed;
        let mut full = RequestSampler::start(sampled(key, 99), &[3]);
        let expect = run(&mut full, 8);
        let head = run(&mut first, 1);
        let mut rest =
            RequestSampler::resume(first.params().clone(), first.context().to_vec(), 1).unwrap();
        assert_eq!(rest.last_position(), 1);
        let tail = run(&mut rest, 7);
        assert_eq!([head, tail].concat(), expect);
    }

    #[test]
    fn greedy_picks_the_peak_and_reports_finish() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut row = [0.0f32; VOCAB];
        row[9] = 5.0;
        let logits = Tensor::<CpuRuntime>::from_slice(&row, &[1, 1, VOCAB], &device).unwrap();
        let mut p = GenerationParams::greedy(3);
        let mut s = RequestSampler::start(p.clone(), &[1]);
        let ev = s.sample::<CpuRuntime>(&client, &logits).unwrap();
        assert_eq!((ev.index, ev.token, ev.finish), (0, 9, None));
        p.eos_tokens = vec![9];
        let mut s = RequestSampler::start(p, &[1]);
        let ev = s.sample::<CpuRuntime>(&client, &logits).unwrap();
        assert_eq!(ev.finish, Some(super::super::protocol::FinishReason::Eos));
        assert_eq!(s.context(), &[1, 9]);
    }

    #[test]
    fn penalties_read_the_context() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut row = [0.0f32; VOCAB];
        row[4] = 2.0;
        row[5] = 1.9;
        let logits = Tensor::<CpuRuntime>::from_slice(&row, &[1, 1, VOCAB], &device).unwrap();
        let mut p = GenerationParams::greedy(5);
        p.rule.presence_penalty = 1.0;
        let mut s = RequestSampler::start(p, &[4]);
        let ev = s.sample::<CpuRuntime>(&client, &logits).unwrap();
        assert_eq!(ev.token, 5, "the prompt token 4 takes the presence penalty");
    }

    #[test]
    fn resume_refuses_a_missing_seed_and_a_bad_count() {
        let p = sampled(None, 4);
        assert!(RequestSampler::resume(p.clone(), vec![1, 2], 1).is_err());
        let seeded = sampled(Some(1), 4);
        assert!(RequestSampler::resume(seeded.clone(), vec![1, 2], 0).is_err());
        assert!(RequestSampler::resume(seeded, vec![1, 2], 2).is_err());
    }
}
