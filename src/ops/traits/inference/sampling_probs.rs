//! The canonical sampling filter: logits in, filtered probabilities out.

use crate::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The temperature, top-k, top-p and min-p config of one sampling filter.
///
/// The filter runs in this order: temperature, softmax, top-k, top-p, min-p, renormalize.
///
/// - `temperature`: the logits divide by it. It must be finite and above 0.
/// - `top_k`: keeps every token with `p >= kth_largest`, so ties at the kth value stay.
///   0 or a value of at least the vocab size disables it.
/// - `top_p`: keeps every token with `p >= tau`. `tau` is the largest value
///   whose kept mass reaches `top_p` times the mass top-k left. It must be
///   finite and above 0. 1 or more disables it.
/// - `min_p`: keeps every token with `p >= min_p * max(p)`. It must be
///   finite and at most 1. 0 or less disables it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterRule {
    /// The logit divisor. Greedy decoding never calls the filter.
    pub temperature: f32,
    /// The rank whose probability is the top-k threshold.
    pub top_k: usize,
    /// The share of the remaining mass the top-p set must reach.
    pub top_p: f32,
    /// The min-p threshold as a fraction of the top probability.
    pub min_p: f32,
}

impl FilterRule {
    /// A config that applies only `temperature`, with every filter disabled.
    pub const fn temperature_only(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }

    /// Whether top-k removes any token from a row of `vocab` tokens.
    pub fn top_k_active(&self, vocab: usize) -> bool {
        self.top_k > 0 && self.top_k < vocab
    }

    /// Whether top-p removes any token.
    pub fn top_p_active(&self) -> bool {
        self.top_p < 1.0
    }

    /// Whether min-p removes any token.
    pub fn min_p_active(&self) -> bool {
        self.min_p > 0.0
    }

    /// Checks every field against its range.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` that names the first field out of range.
    pub fn check(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "temperature",
                reason: format!(
                    "sampling_probs: temperature must be finite and above 0, got {} — \
                     run greedy argmax for temperature 0 instead of the filter",
                    self.temperature
                ),
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "top_p",
                reason: format!(
                    "sampling_probs: top_p must be finite and above 0, got {} — \
                     pass 1.0 to disable top-p",
                    self.top_p
                ),
            });
        }
        if !self.min_p.is_finite() || self.min_p > 1.0 {
            return Err(Error::InvalidArgument {
                arg: "min_p",
                reason: format!(
                    "sampling_probs: min_p must be finite and at most 1, got {} — \
                     pass 0.0 to disable min-p",
                    self.min_p
                ),
            });
        }
        Ok(())
    }
}

/// The canonical sampling filter, on the device of the logits.
///
/// Every backend must give the same kept set as the generic implementation.
pub trait SamplingProbsOps<R: Runtime> {
    /// Filters `logits` by `rule` and returns the renormalized probabilities.
    ///
    /// # Layout
    ///
    /// - `logits`: `[.., vocab]` in any float dtype. The math runs in F32.
    /// - Output: `[.., vocab]` F32. Each row sums to 1. A removed token holds 0.
    ///
    /// The filter reads nothing back to the host.
    ///
    /// # Errors
    ///
    /// Returns an error if `rule` fails [`FilterRule::check`], `logits` is
    /// not a float dtype, or the vocab axis is empty.
    fn sampling_probs(&self, logits: &Tensor<R>, rule: &FilterRule) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(temperature: f32, top_p: f32, min_p: f32) -> FilterRule {
        FilterRule {
            temperature,
            top_k: 0,
            top_p,
            min_p,
        }
    }

    #[test]
    fn check_names_the_field_out_of_range() {
        let cases = [
            (rule(0.0, 1.0, 0.0), "temperature"),
            (rule(-1.0, 1.0, 0.0), "temperature"),
            (rule(f32::NAN, 1.0, 0.0), "temperature"),
            (rule(f32::INFINITY, 1.0, 0.0), "temperature"),
            (rule(1.0, 0.0, 0.0), "top_p"),
            (rule(1.0, f32::NAN, 0.0), "top_p"),
            (rule(1.0, 1.0, 1.5), "min_p"),
            (rule(1.0, 1.0, f32::NAN), "min_p"),
        ];
        for (r, field) in cases {
            match r.check() {
                Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, field, "{r:?}"),
                other => panic!("{r:?}: expected an error naming {field}, got {other:?}"),
            }
        }
    }

    #[test]
    fn check_accepts_the_disabled_values() {
        assert!(rule(1.0, 1.0, 0.0).check().is_ok());
        assert!(rule(0.5, 2.0, -1.0).check().is_ok());
        assert!(rule(2.0, 0.5, 1.0).check().is_ok());
        assert!(FilterRule::temperature_only(0.7).check().is_ok());
    }

    #[test]
    fn active_flags_follow_the_disable_values() {
        let r = FilterRule {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            min_p: 0.0,
        };
        assert!(r.top_k_active(5));
        assert!(!r.top_k_active(4));
        assert!(!r.top_p_active());
        assert!(!r.min_p_active());
        let off = FilterRule { top_k: 0, ..r };
        assert!(!off.top_k_active(5));
    }
}
