//! Speculative generation config.

use super::sample_rule::SampleRule;
use crate::error::{Error, Result};

/// Drafts per step when the caller sets no other count.
pub const DEFAULT_NUM_DRAFT_TOKENS: usize = 5;

/// Largest accepted draft count per step.
pub const MAX_NUM_DRAFT_TOKENS: usize = 32;

/// Config of one speculative generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeConfig {
    /// Drafts proposed per step, `k`. Must be in `1..=MAX_NUM_DRAFT_TOKENS`.
    pub num_draft_tokens: usize,
    /// Tokens to emit at most, the prompt not counted. Must be at least 1.
    pub max_tokens: usize,
    /// Tokens that end generation. The EOS token is part of the output.
    pub eos_token_ids: Vec<u32>,
    /// The acceptance rule. `None`, or temperature 0 without a penalty, runs the plain greedy path.
    pub sampling: Option<SampleRule>,
}

impl SpeculativeConfig {
    /// A greedy config with `max_tokens`, the default draft count and no EOS ids.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            num_draft_tokens: DEFAULT_NUM_DRAFT_TOKENS,
            max_tokens,
            eos_token_ids: Vec::new(),
            sampling: None,
        }
    }

    /// The rule that `extend_sampled` verifies with, or `None` when the plain greedy path runs.
    ///
    /// A greedy rule with a penalty returns `Some`: its argmax runs over penalized logits.
    pub fn sampled_rule(&self) -> Option<&SampleRule> {
        self.sampling
            .as_ref()
            .filter(|rule| !rule.is_greedy() || rule.has_penalties())
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
        if let Some(rule) = &self.sampling {
            rule.check()?;
        }
        Ok(())
    }

    /// `token` ends generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::inference::FilterRule;

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
        assert_eq!(config.sampling, None);
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
    fn a_bad_sample_rule_names_its_field() {
        let mut rule = SampleRule::new(FilterRule::temperature_only(1.0));
        rule.filter.top_p = 0.0;
        let config = SpeculativeConfig {
            sampling: Some(rule),
            ..SpeculativeConfig::new(4)
        };
        assert_eq!(arg_of(config.check(10).expect_err("top_p 0")), "top_p");
    }

    #[test]
    fn temperature_zero_selects_the_greedy_path() {
        let greedy = SpeculativeConfig {
            sampling: Some(SampleRule::new(FilterRule {
                temperature: 0.0,
                ..FilterRule::temperature_only(1.0)
            })),
            ..SpeculativeConfig::new(4)
        };
        greedy.check(10).expect("temperature 0 is valid");
        assert!(greedy.sampled_rule().is_none());
        let sampled = SpeculativeConfig {
            sampling: Some(SampleRule::new(FilterRule::temperature_only(0.7))),
            ..SpeculativeConfig::new(4)
        };
        assert!(sampled.sampled_rule().is_some());
        assert!(SpeculativeConfig::new(4).sampled_rule().is_none());
    }

    #[test]
    fn greedy_with_a_penalty_selects_the_verify_path() {
        let rule = SampleRule {
            repeat_penalty: 1.1,
            penalty_last_n: 64,
            ..SampleRule::new(FilterRule {
                temperature: 0.0,
                ..FilterRule::temperature_only(1.0)
            })
        };
        let config = SpeculativeConfig {
            sampling: Some(rule),
            ..SpeculativeConfig::new(4)
        };
        config
            .check(10)
            .expect("temperature 0 with a penalty is valid");
        assert_eq!(config.sampled_rule(), Some(&rule));
    }
}
