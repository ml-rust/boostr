//! Executor state, the generation loop, and adaptive depth control.

use crate::error::Result;
use numr::runtime::Runtime;

use super::super::types::{SpeculativeConfig, SpeculativeModel, SpeculativeStats, TokenId};

/// Speculative decoding executor
pub struct SpeculativeExecutor<R: Runtime, D, T>
where
    D: SpeculativeModel<R>,
    T: SpeculativeModel<R>,
{
    pub(super) draft_model: D,
    pub(super) target_model: T,
    pub(super) config: SpeculativeConfig,
    pub(crate) current_depth: usize,
    pub stats: SpeculativeStats,
    pub(super) rng_state: u64,
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

            if let Some(bonus) = verification.bonus_token
                && generated.len() < max_new_tokens
            {
                context.push(bonus);
                generated.push(bonus);
                self.stats.bonus_tokens += 1;
                self.stats.total_tokens += 1;
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
pub(super) mod tests {
    use super::super::super::types::VerificationResult;
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    pub(in super::super) struct MockModel {
        vocab_size: usize,
        name: String,
        fixed_logits: Option<Vec<f32>>,
    }

    impl MockModel {
        pub(in super::super) fn new(vocab_size: usize, name: &str) -> Self {
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
