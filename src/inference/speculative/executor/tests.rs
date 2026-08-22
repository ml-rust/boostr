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
