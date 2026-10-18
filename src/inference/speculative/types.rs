//! Speculative generation config, statistics and finish reason.

use crate::error::{Error, Result};

/// Drafts per step when the caller sets no other count.
pub const DEFAULT_NUM_DRAFT_TOKENS: usize = 5;

/// Largest accepted draft count per step.
pub const MAX_NUM_DRAFT_TOKENS: usize = 32;

/// Config of one speculative generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeConfig {
    /// Drafts proposed per step, `k`. Must be in `1..=MAX_NUM_DRAFT_TOKENS`.
    pub num_draft_tokens: usize,
    /// Tokens to emit at most, the prompt not counted. Must be at least 1.
    pub max_tokens: usize,
    /// Tokens that end generation. The EOS token is part of the output.
    pub eos_token_ids: Vec<u32>,
}

impl SpeculativeConfig {
    /// A config with `max_tokens`, the default draft count and no EOS ids.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            num_draft_tokens: DEFAULT_NUM_DRAFT_TOKENS,
            max_tokens,
            eos_token_ids: Vec::new(),
        }
    }

    /// Check every field against a model with `vocab_size` tokens.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the first field out of range.
    pub fn check(&self, vocab_size: usize) -> Result<()> {
        if !(1..=MAX_NUM_DRAFT_TOKENS).contains(&self.num_draft_tokens) {
            return Err(Error::InvalidArgument {
                arg: "num_draft_tokens",
                reason: format!(
                    "got {}; pass a draft count in 1..={MAX_NUM_DRAFT_TOKENS}",
                    self.num_draft_tokens
                ),
            });
        }
        if self.max_tokens == 0 {
            return Err(Error::InvalidArgument {
                arg: "max_tokens",
                reason: "got 0; pass a token limit of at least 1".into(),
            });
        }
        if let Some(&id) = self
            .eos_token_ids
            .iter()
            .find(|&&id| id as usize >= vocab_size)
        {
            return Err(Error::InvalidArgument {
                arg: "eos_token_ids",
                reason: format!(
                    "EOS id {id} is outside the model vocabulary of {vocab_size} tokens; pass \
                     ids below {vocab_size}"
                ),
            });
        }
        Ok(())
    }

    /// `token` ends generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishReason {
    /// The last emitted token is an EOS id.
    Eos,
    /// The output reached `max_tokens`.
    Length,
}

/// Counters of one generation. Prefill is not counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Steps run.
    pub steps: usize,
    /// Drafts fed to the target, after the clamp and the vocabulary cut.
    pub drafted: usize,
    /// Drafts equal to the target's own choice, before the EOS cut.
    pub accepted: usize,
    /// Tokens emitted to the caller.
    pub emitted: usize,
    /// Calls of [`SpeculativeModel::extend_greedy`](super::SpeculativeModel::extend_greedy)
    /// during steps. Each step makes exactly one.
    pub target_forwards: usize,
}

impl SpeculativeStats {
    /// `accepted / drafted`, or 0.0 when nothing was drafted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// `emitted / target_forwards`, or 0.0 before the first step.
    pub fn tokens_per_forward(&self) -> f64 {
        if self.target_forwards == 0 {
            0.0
        } else {
            self.emitted as f64 / self.target_forwards as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg_of(err: Error) -> &'static str {
        match err {
            Error::InvalidArgument { arg, .. } => arg,
            other => panic!("expected InvalidArgument, got {other}"),
        }
    }

    #[test]
    fn default_config_passes() {
        let config = SpeculativeConfig::new(16);
        assert_eq!(config.num_draft_tokens, DEFAULT_NUM_DRAFT_TOKENS);
        config.check(10).expect("valid");
    }

    #[test]
    fn draft_count_outside_range_names_the_field() {
        for k in [0, MAX_NUM_DRAFT_TOKENS + 1] {
            let config = SpeculativeConfig {
                num_draft_tokens: k,
                ..SpeculativeConfig::new(4)
            };
            assert_eq!(
                arg_of(config.check(10).expect_err("bad k")),
                "num_draft_tokens"
            );
        }
        let config = SpeculativeConfig {
            num_draft_tokens: MAX_NUM_DRAFT_TOKENS,
            ..SpeculativeConfig::new(4)
        };
        config.check(10).expect("k = 32 is valid");
    }

    #[test]
    fn zero_max_tokens_names_the_field() {
        let err = SpeculativeConfig::new(0).check(10).expect_err("zero");
        assert_eq!(arg_of(err), "max_tokens");
    }

    #[test]
    fn eos_outside_vocab_names_the_field() {
        let config = SpeculativeConfig {
            eos_token_ids: vec![3, 10],
            ..SpeculativeConfig::new(4)
        };
        let err = config.check(10).expect_err("10 is outside");
        assert!(err.to_string().contains("10"), "{err}");
        assert_eq!(arg_of(err), "eos_token_ids");
        config.check(11).expect("valid with a larger vocab");
    }

    #[test]
    fn rates_handle_zero_counts() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.tokens_per_forward(), 0.0);
        let stats = SpeculativeStats {
            steps: 2,
            drafted: 4,
            accepted: 3,
            emitted: 5,
            target_forwards: 2,
        };
        assert_eq!(stats.acceptance_rate(), 0.75);
        assert_eq!(stats.tokens_per_forward(), 2.5);
    }
}
