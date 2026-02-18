//! Speculative decoding types, configuration, and model trait

use crate::error::Result;
use numr::runtime::Runtime;

/// Token ID type
pub type TokenId = u32;

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub num_speculative_tokens: usize,
    pub draft_temperature: f32,
    pub target_temperature: f32,
    pub draft_top_p: f32,
    pub target_top_p: f32,
    pub min_acceptance_rate: f32,
    pub adaptive_depth: bool,
    pub seed: Option<u64>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            draft_temperature: 0.7,
            target_temperature: 0.7,
            draft_top_p: 0.9,
            target_top_p: 0.9,
            min_acceptance_rate: 0.3,
            adaptive_depth: true,
            seed: None,
        }
    }
}

/// Statistics for speculative decoding performance
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeStats {
    pub total_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub bonus_tokens: usize,
    pub iterations: usize,
    pub draft_forward_passes: usize,
    pub target_forward_passes: usize,
}

impl SpeculativeStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.accepted_tokens + self.rejected_tokens == 0 {
            1.0
        } else {
            self.accepted_tokens as f32 / (self.accepted_tokens + self.rejected_tokens) as f32
        }
    }

    pub fn tokens_per_forward(&self) -> f32 {
        if self.target_forward_passes == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.target_forward_passes as f32
        }
    }

    pub fn estimated_speedup(&self) -> f32 {
        if self.target_forward_passes == 0 {
            1.0
        } else {
            let draft_cost = self.draft_forward_passes as f32 / 10.0;
            let total_cost = self.target_forward_passes as f32 + draft_cost;
            self.total_tokens as f32 / total_cost
        }
    }
}

/// Result of token verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub num_accepted: usize,
    pub accepted_tokens: Vec<TokenId>,
    pub bonus_token: Option<TokenId>,
    pub first_rejection_pos: i32,
}

/// Draft model output for a single speculative step
#[derive(Debug, Clone)]
pub struct DraftOutput {
    pub tokens: Vec<TokenId>,
    pub log_probs: Vec<f32>,
    pub vocab_size: usize,
}

/// Target model output for verification
#[derive(Debug, Clone)]
pub struct TargetOutput {
    pub log_probs: Vec<f32>,
    pub vocab_size: usize,
}

/// Trait for models that can be used as draft or target in speculative decoding
pub trait SpeculativeModel<R: Runtime>: Send + Sync {
    fn forward(&mut self, input_tokens: &[TokenId], position: usize) -> Result<Vec<f32>>;
    fn vocab_size(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn name(&self) -> &str;
}
