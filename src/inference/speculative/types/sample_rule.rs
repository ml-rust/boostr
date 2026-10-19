//! The sampled acceptance rule, the per-step draw and the verify verdict.

use super::config::MAX_NUM_DRAFT_TOKENS;
use crate::error::{Error, Result};
use crate::ops::traits::inference::FilterRule;

/// Uniforms one Philox counter yields.
const PHILOX_LANES: u64 = 4;

/// Philox counters one step reserves.
///
/// A step draws up to `MAX_NUM_DRAFT_TOKENS + 1` uniforms. Each counter
/// yields 4. So two steps never share a counter.
pub const COUNTERS_PER_STEP: u64 = (MAX_NUM_DRAFT_TOKENS as u64 + 1).div_ceil(PHILOX_LANES);

/// The sampling config of speculative generation.
///
/// The target distribution of each position is the canonical filter of
/// [`FilterRule`] over penalized logits. The penalties count every token of
/// the prefix that ends at that position, the prompt included.
///
/// Temperature 0 selects the greedy path. That path applies no penalty, so
/// [`check`](Self::check) refuses temperature 0 with an active penalty.
///
/// Mirostat, DRY, typical-p and grammar constraints have no field here. A
/// caller that needs them must decode through the plain sampler instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleRule {
    /// Temperature, top-k, top-p and min-p.
    pub filter: FilterRule,
    /// Divides a positive logit and multiplies a negative one. 1 disables it.
    pub repeat_penalty: f32,
    /// Subtracted once per occurrence. 0 disables it.
    pub frequency_penalty: f32,
    /// Subtracted once per distinct token. 0 disables it.
    pub presence_penalty: f32,
    /// The Philox key. `None` takes one seed from `numr::random` at start.
    pub seed: Option<u64>,
}

impl SampleRule {
    /// A rule with `filter`, no penalty and no fixed seed.
    pub const fn new(filter: FilterRule) -> Self {
        Self {
            filter,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }

    /// Temperature 0: the generator runs the greedy path.
    pub fn is_greedy(&self) -> bool {
        self.filter.temperature == 0.0
    }

    /// At least one penalty changes a logit.
    pub fn has_penalties(&self) -> bool {
        self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }

    /// Check every field against its range.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the first field out of range.
    pub fn check(&self) -> Result<()> {
        if self.is_greedy() {
            FilterRule {
                temperature: 1.0,
                ..self.filter
            }
            .check()?;
        } else {
            self.filter.check()?;
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "repeat_penalty",
                reason: format!(
                    "got {}; pass a finite value above 0, or 1.0 to disable it",
                    self.repeat_penalty
                ),
            });
        }
        let additive = [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ];
        if let Some(&(arg, value)) = additive.iter().find(|(_, v)| !v.is_finite()) {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!("got {value}; pass a finite value, or 0.0 to disable it"),
            });
        }
        if self.is_greedy() && self.has_penalties() {
            let arg = if self.repeat_penalty != 1.0 {
                "repeat_penalty"
            } else if self.frequency_penalty != 0.0 {
                "frequency_penalty"
            } else {
                "presence_penalty"
            };
            return Err(Error::InvalidArgument {
                arg,
                reason: "temperature 0 runs the greedy path, which applies no penalty; set a \
                         temperature above 0 or disable the penalty"
                    .into(),
            });
        }
        Ok(())
    }
}

/// The Philox position of one sampled step.
///
/// The step draws `K + 1` uniforms from counters `counter..counter + COUNTERS_PER_STEP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Draw {
    /// The Philox key of the generation.
    pub key: u64,
    /// The first Philox counter of the step.
    pub counter: u64,
}

impl Draw {
    /// The draw of step `step` in the stream `key` names.
    pub const fn for_step(key: u64, step: u64) -> Self {
        Self {
            key,
            counter: step.wrapping_mul(COUNTERS_PER_STEP),
        }
    }
}

/// The outcome of one sampled verify forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Verdict {
    /// Leading drafts the target accepted.
    pub accepted: usize,
    /// The token after the accepted drafts.
    pub next: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg_of(rule: SampleRule) -> &'static str {
        match rule.check() {
            Err(Error::InvalidArgument { arg, .. }) => arg,
            other => panic!("{rule:?}: expected InvalidArgument, got {other:?}"),
        }
    }

    fn base() -> SampleRule {
        SampleRule::new(FilterRule::temperature_only(0.8))
    }

    #[test]
    fn counters_cover_the_largest_step() {
        assert!(COUNTERS_PER_STEP * PHILOX_LANES > MAX_NUM_DRAFT_TOKENS as u64);
        assert!((COUNTERS_PER_STEP - 1) * PHILOX_LANES < MAX_NUM_DRAFT_TOKENS as u64 + 1);
        let (a, b) = (Draw::for_step(0, 3), Draw::for_step(0, 4));
        assert_eq!(b.counter - a.counter, COUNTERS_PER_STEP);
    }

    #[test]
    fn valid_rules_pass() {
        base().check().expect("plain");
        let rule = SampleRule {
            repeat_penalty: 1.3,
            frequency_penalty: -0.5,
            presence_penalty: 0.4,
            seed: Some(0),
            ..base()
        };
        rule.check().expect("penalties");
        let greedy = SampleRule::new(FilterRule {
            temperature: 0.0,
            ..FilterRule::temperature_only(1.0)
        });
        assert!(greedy.is_greedy());
        greedy.check().expect("greedy");
    }

    #[test]
    fn bad_fields_are_named() {
        let mut rule = base();
        rule.filter.temperature = -1.0;
        assert_eq!(arg_of(rule), "temperature");
        let mut rule = base();
        rule.filter.min_p = 2.0;
        assert_eq!(arg_of(rule), "min_p");
        for p in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                arg_of(SampleRule {
                    repeat_penalty: p,
                    ..base()
                }),
                "repeat_penalty"
            );
        }
        let rule = SampleRule {
            frequency_penalty: f32::NAN,
            ..base()
        };
        assert_eq!(arg_of(rule), "frequency_penalty");
        let rule = SampleRule {
            presence_penalty: f32::INFINITY,
            ..base()
        };
        assert_eq!(arg_of(rule), "presence_penalty");
    }

    #[test]
    fn greedy_with_a_penalty_is_refused() {
        let mut rule = SampleRule {
            presence_penalty: 0.5,
            ..base()
        };
        rule.filter.temperature = 0.0;
        assert_eq!(arg_of(rule), "presence_penalty");
        rule.filter.top_p = 0.0;
        assert_eq!(
            arg_of(rule),
            "top_p",
            "a greedy rule still checks the filter"
        );
    }
}
