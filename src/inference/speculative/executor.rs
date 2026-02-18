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
    pub(crate) stats: SpeculativeStats,
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
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    struct MockModel {
        vocab_size: usize,
        name: String,
        fixed_logits: Option<Vec<f32>>,
    }

    impl MockModel {
        fn new(vocab_size: usize, name: &str) -> Self {
            Self {
                vocab_size,
                name: name.to_string(),
                fixed_logits: None,
            }
        }

        fn with_fixed_logits(mut self, logits: Vec<f32>) -> Self {
            self.fixed_logits = Some(logits);
            self
        }
    }

    impl SpeculativeModel<CpuRuntime> for MockModel {
        fn forward(&mut self, _input_tokens: &[TokenId], _position: usize) -> Result<Vec<f32>> {
            if let Some(ref logits) = self.fixed_logits {
                Ok(logits.clone())
            } else {
                Ok(vec![0.0; self.vocab_size])
            }
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_speculative_tokens, 4);
        assert!((config.draft_temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_stats() {
        let stats = SpeculativeStats {
            accepted_tokens: 80,
            rejected_tokens: 20,
            total_tokens: 100,
            target_forward_passes: 25,
            draft_forward_passes: 100,
            ..Default::default()
        };

        assert!((stats.acceptance_rate() - 0.8).abs() < 1e-6);
        assert!((stats.tokens_per_forward() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_executor_creation() {
        let draft = MockModel::new(100, "draft");
        let target = MockModel::new(100, "target");
        let config = SpeculativeConfig::default();

        let executor = SpeculativeExecutor::<CpuRuntime, _, _>::new(draft, target, config);

        assert_eq!(executor.current_depth(), 4);
        assert_eq!(executor.stats().total_tokens, 0);
    }

    #[test]
    fn test_speculative_executor_generate() {
        let mut logits = vec![-10.0; 100];
        logits[5] = 0.0;
        logits[2] = -100.0;

        let draft = MockModel::new(100, "draft").with_fixed_logits(logits.clone());
        let target = MockModel::new(100, "target").with_fixed_logits(logits);

        let config = SpeculativeConfig {
            num_speculative_tokens: 2,
            seed: Some(12345),
            ..Default::default()
        };

        let mut executor = SpeculativeExecutor::<CpuRuntime, _, _>::new(draft, target, config);

        let prompt = vec![1, 2, 3];
        let result = executor.generate(&prompt, 5).unwrap();

        assert!(!result.is_empty());
        assert!(result.len() <= 5);

        let stats = executor.stats();
        assert!(stats.total_tokens > 0);
        assert!(stats.target_forward_passes > 0);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            num_accepted: 3,
            accepted_tokens: vec![10, 20, 30],
            bonus_token: Some(40),
            first_rejection_pos: -1,
        };

        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.accepted_tokens.len(), 3);
        assert_eq!(result.bonus_token, Some(40));
    }

    #[test]
    fn test_apply_temperature_and_softmax() {
        let draft = MockModel::new(4, "draft");
        let target = MockModel::new(4, "target");
        let config = SpeculativeConfig::default();
        let executor = SpeculativeExecutor::<CpuRuntime, _, _>::new(draft, target, config);

        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let log_probs = executor.apply_temperature_and_softmax(&logits, 1.0);

        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        assert!(log_probs[3] > log_probs[2]);
        assert!(log_probs[2] > log_probs[1]);
        assert!(log_probs[1] > log_probs[0]);
    }

    #[test]
    fn test_adaptive_depth() {
        let draft = MockModel::new(100, "draft");
        let target = MockModel::new(100, "target");
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_depth: true,
            min_acceptance_rate: 0.3,
            ..Default::default()
        };

        let mut executor = SpeculativeExecutor::<CpuRuntime, _, _>::new(draft, target, config);

        executor.stats.accepted_tokens = 10;
        executor.stats.rejected_tokens = 90;
        executor.adjust_depth();
        assert!(executor.current_depth() < 4);

        executor.stats.accepted_tokens = 90;
        executor.stats.rejected_tokens = 10;
        executor.current_depth = 2;
        executor.adjust_depth();
        assert!(executor.current_depth() > 2);
    }
}
