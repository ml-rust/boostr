//! Speculative decoding executor

use crate::error::Result;
use numr::runtime::Runtime;

use super::types::{
    DraftOutput, SpeculativeConfig, SpeculativeModel, SpeculativeStats, TargetOutput, TokenId,
    VerificationResult,
};

/// Speculative decoding executor
pub struct SpeculativeExecutor<R: Runtime, D, T>
where
    D: SpeculativeModel<R>,
    T: SpeculativeModel<R>,
{
    draft_model: D,
    target_model: T,
    config: SpeculativeConfig,
    pub(crate) current_depth: usize,
    pub stats: SpeculativeStats,
    rng_state: u64,
    _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime, D, T> SpeculativeExecutor<R, D, T>
where
    D: SpeculativeModel<R>,
    T: SpeculativeModel<R>,
{
    pub fn new(draft_model: D, target_model: T, config: SpeculativeConfig) -> Self {
        let seed = config.seed.unwrap_or(42);
        let current_depth = config.num_speculative_tokens;

        Self {
            draft_model,
            target_model,
            config,
            current_depth,
            stats: SpeculativeStats::default(),
            rng_state: seed,
            _runtime: std::marker::PhantomData,
        }
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        max_new_tokens: usize,
    ) -> Result<Vec<TokenId>> {
        self.draft_model.reset()?;
        self.target_model.reset()?;
        self.stats = SpeculativeStats::default();
        self.current_depth = self.config.num_speculative_tokens;

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut context = prompt_tokens.to_vec();

        while generated.len() < max_new_tokens {
            self.stats.iterations += 1;

            let draft_output = self.draft_speculate(&context)?;
            let target_output = self.target_verify(&context, &draft_output.tokens)?;
            let verification = self.verify_and_sample(&draft_output, &target_output)?;

            for &token in &verification.accepted_tokens {
                context.push(token);
                generated.push(token);
                self.stats.accepted_tokens += 1;
                self.stats.total_tokens += 1;

                if generated.len() >= max_new_tokens {
                    break;
                }
            }

            if let Some(bonus) = verification.bonus_token {
                if generated.len() < max_new_tokens {
                    context.push(bonus);
                    generated.push(bonus);
                    self.stats.bonus_tokens += 1;
                    self.stats.total_tokens += 1;
                }
            }

            let num_rejected = draft_output.tokens.len() - verification.num_accepted;
            self.stats.rejected_tokens += num_rejected;

            if self.config.adaptive_depth {
                self.adjust_depth();
            }

            if generated.last().is_some_and(|&t| t == 0 || t == 2) {
                break;
            }
        }

        Ok(generated)
    }

    fn draft_speculate(&mut self, context: &[TokenId]) -> Result<DraftOutput> {
        let vocab_size = self.draft_model.vocab_size();
        let mut tokens = Vec::with_capacity(self.current_depth);
        let mut log_probs = Vec::with_capacity(self.current_depth * vocab_size);

        let mut current_context = context.to_vec();

        for i in 0..self.current_depth {
            self.stats.draft_forward_passes += 1;

            let logits = self
                .draft_model
                .forward(&current_context, context.len() + i)?;

            let log_prob =
                self.apply_temperature_and_softmax(&logits, self.config.draft_temperature);

            let token = self.sample_from_logprobs(&log_prob, self.config.draft_top_p)?;

            tokens.push(token);
            log_probs.extend(log_prob);
            current_context.push(token);
        }

        Ok(DraftOutput {
            tokens,
            log_probs,
            vocab_size,
        })
    }

    fn target_verify(
        &mut self,
        context: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> Result<TargetOutput> {
        self.stats.target_forward_passes += 1;

        let vocab_size = self.target_model.vocab_size();
        let mut all_log_probs = Vec::with_capacity((draft_tokens.len() + 1) * vocab_size);

        let mut current_context = context.to_vec();

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let logits = self
                .target_model
                .forward(&current_context, context.len() + i)?;
            let log_prob =
                self.apply_temperature_and_softmax(&logits, self.config.target_temperature);
            all_log_probs.extend(log_prob);
            current_context.push(draft_token);
        }

        let logits = self
            .target_model
            .forward(&current_context, context.len() + draft_tokens.len())?;
        let log_prob = self.apply_temperature_and_softmax(&logits, self.config.target_temperature);
        all_log_probs.extend(log_prob);

        Ok(TargetOutput {
            log_probs: all_log_probs,
            vocab_size,
        })
    }

    fn verify_and_sample(
        &mut self,
        draft: &DraftOutput,
        target: &TargetOutput,
    ) -> Result<VerificationResult> {
        let vocab_size = draft.vocab_size;
        let mut accepted_tokens = Vec::new();
        let mut first_rejection_pos = -1i32;

        for (i, &draft_token) in draft.tokens.iter().enumerate() {
            let draft_prob = draft.log_probs[i * vocab_size + draft_token as usize].exp();
            let target_prob = target.log_probs[i * vocab_size + draft_token as usize].exp();

            let accept_prob = if draft_prob > 0.0 {
                (target_prob / draft_prob).min(1.0)
            } else {
                1.0
            };

            let r = self.random_uniform();

            if r < accept_prob {
                accepted_tokens.push(draft_token);
            } else {
                first_rejection_pos = i as i32;
                break;
            }
        }

        let bonus_token = if first_rejection_pos >= 0 {
            let i = first_rejection_pos as usize;
            let mut residual_probs = Vec::with_capacity(vocab_size);

            for t in 0..vocab_size {
                let draft_prob = draft.log_probs[i * vocab_size + t].exp();
                let target_prob = target.log_probs[i * vocab_size + t].exp();
                residual_probs.push((target_prob - draft_prob).max(0.0));
            }

            let sum: f32 = residual_probs.iter().sum();
            if sum > 1e-8 {
                for p in &mut residual_probs {
                    *p /= sum;
                }
                Some(self.sample_from_probs(&residual_probs)?)
            } else {
                let target_probs: Vec<f32> = (0..vocab_size)
                    .map(|t| target.log_probs[i * vocab_size + t].exp())
                    .collect();
                Some(self.sample_from_probs(&target_probs)?)
            }
        } else {
            let last_pos = draft.tokens.len();
            let target_probs: Vec<f32> = (0..vocab_size)
                .map(|t| target.log_probs[last_pos * vocab_size + t].exp())
                .collect();
            Some(self.sample_from_probs(&target_probs)?)
        };

        Ok(VerificationResult {
            num_accepted: accepted_tokens.len(),
            accepted_tokens,
            bonus_token,
            first_rejection_pos,
        })
    }

    pub(crate) fn apply_temperature_and_softmax(
        &self,
        logits: &[f32],
        temperature: f32,
    ) -> Vec<f32> {
        let temp = temperature.max(1e-8);
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum = max_val + exp_sum.ln();
        scaled.iter().map(|&x| x - log_sum).collect()
    }

    fn sample_from_logprobs(&mut self, log_probs: &[f32], top_p: f32) -> Result<TokenId> {
        let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();

        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let top_p_tokens: Vec<(usize, f32)> = indexed[..cutoff_idx].to_vec();
        let sum: f32 = top_p_tokens.iter().map(|(_, p)| p).sum();

        let r = self.random_uniform() * sum;
        let mut cumsum = 0.0;

        for (token_id, p) in top_p_tokens {
            cumsum += p;
            if r < cumsum {
                return Ok(token_id as TokenId);
            }
        }

        Ok(indexed[0].0 as TokenId)
    }

    fn sample_from_probs(&mut self, probs: &[f32]) -> Result<TokenId> {
        let sum: f32 = probs.iter().sum();
        if sum < 1e-8 {
            return Ok(0);
        }

        let r = self.random_uniform() * sum;
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as TokenId);
            }
        }

        Ok(probs.len() as TokenId - 1)
    }

    fn random_uniform(&mut self) -> f32 {
        const A: u64 = 48271;
        const M: u64 = 2147483647;
        self.rng_state = (A.wrapping_mul(self.rng_state)) % M;
        self.rng_state as f32 / M as f32
    }

    pub(crate) fn adjust_depth(&mut self) {
        let rate = self.stats.acceptance_rate();

        if rate < self.config.min_acceptance_rate && self.current_depth > 1 {
            self.current_depth = (self.current_depth - 1).max(1);
        } else if rate > 0.8 && self.current_depth < self.config.num_speculative_tokens {
            self.current_depth = (self.current_depth + 1).min(self.config.num_speculative_tokens);
        }
    }

    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
        self.current_depth = self.config.num_speculative_tokens;
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: SpeculativeConfig) {
        self.current_depth = config.num_speculative_tokens;
        self.config = config;
    }
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
